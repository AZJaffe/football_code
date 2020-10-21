import argparse
import datetime
from collections import defaultdict
import fire
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logger
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import pprint
from torch.utils.tensorboard import SummaryWriter


import sfmnet
from pair_frames_dataset import PairConsecutiveFramesDataset

log = logger.noop

def get_rank():
  return get_rank() if dist.is_initialized() else 0

def load(checkpoint_file, model, optimizer):
  # Returns the epoch number to start at, as well as loads the optimizer and model
  if checkpoint_file is None:
    return
  try:
    map_location = {'cuda:0': f'cuda:{get_rank()}' if torch.cuda.is_available() else 'cpu'} 
    checkpoint = torch.load(checkpoint_file, map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log.INFO(f'RANK {get_rank()}: Loaded from checkpoint at {checkpoint_file}')
    if 'epoch' in checkpoint:
      return checkpoint['epoch']
    else:
      return 0  
  except FileNotFoundError:
    return 0

def save(checkpoint_file, model, optimizer, epoch):
  if get_rank() is not 0:
    return
  if checkpoint_file is None:
    return
  tmp_file = checkpoint_file + '.tmp'
  torch.save({
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'epoch': epoch
    }, tmp_file)
  os.replace(tmp_file, checkpoint_file)
  log.INFO(f'Checkpoint saved at {checkpoint_file}')

def memory_summary(device):
  return torch.cuda.memory_summary(device) if torch.cuda.is_available() else 'NO CUDA DEVICE'

def noop_callback(*a, **k):
  pass

def train_loop(*,
  device,
  dl_train,
  dl_validation,
  vis_point=None,
  model,
  optimizer,
  forwbackw_data_augmentation=False, # If this is True, the batch size will be twice the size of the batch of dl_train
  flowreg_coeff=0.,
  maskreg_coeff=0.,
  displreg_coeff=0.,
  forwbackwreg_coeff=0.,
  mask_logit_noise_curriculum=None,
  num_epochs=1,
  start_at_epoch=0,
  log_metrics=noop_callback,
  using_ddp=False,
  checkpoint_file=None,
  checkpoint_freq=None,
):
  
  def get_mask_logit_noise(epoch):
    if mask_logit_noise_curriculum is None:
      return 0.
    return min(1., epoch/mask_logit_noise_curriculum)

  def run_step(im1, im2, labels, metrics):
    optimizer.zero_grad()
    im1, im2 = im1.to(device), im2.to(device)
    log.DEBUG(f'Start of train batch {step}:', memory_summary(device))
    batch_size, C, H, W = im1.shape
    forwardbatch = torch.cat((im1, im2), dim=1)
    if forwbackw_data_augmentation:
      backwardbatch = torch.cat((im2, im1), dim=1)
      input = torch.cat((forwardbatch, backwardbatch), dim=0)
      # The target of the first batch_size outputs is im2 and the target of the second B outputs is im1
      target = torch.cat((im2, im1), dim=0)
    else:
      input = forwardbatch
      target = im2
    output, mask, flow, displacement = model(input, mask_logit_noise_var=get_mask_logit_noise(e))
    log.DEBUG(f'After forward {step}:', memory_summary(device))

      
    recon_loss = sfmnet.dssim_loss(target, output)
    # backward forward regularization induces a prior on the output of the network
    # that encourages the output from going forward in time is consistent with
    # the output from going backward in time.
    # Precisely, the masks should be the same and the displacements should be the negations of each other 
    if forwbackw_data_augmentation is True:
      forwbackwreg = forwbackwreg_coeff * (
        torch.mean(torch.sum(torch.abs(mask[0:batch_size] - mask[batch_size: 2*batch_size]), dim=(1,)))
        + torch.mean(torch.sum(torch.square(displacement[0:batch_size] + displacement[batch_size: 2*batch_size]), dim=(1,2)))
      )
    else:
      forwbackwreg = 0.
    flowreg = flowreg_coeff * sfmnet.l1_flow_regularization(mask, displacement)
    maskreg = maskreg_coeff * sfmnet.l1_mask_regularization(mask)
    displreg = displreg_coeff * sfmnet.l2_displacement_regularization(displacement)

    loss = recon_loss + flowreg + maskreg + displreg + forwbackwreg
    if 'camera_translation' in labels:
      ct = labels['camera_translation'].to(device)
      mse = torch.mean(torch.sum(torch.square(displacement[:,0] * torch.tensor([W/2,H/2], device=device) - ct), dim=1))
      mse.backward()
      metrics['Label/CameraDisplMSE'] += mse.item() * batch_size
    log.DEBUG(f'Before backward {step}:', memory_summary(device))
    #loss.backward()
    log.DEBUG(f'After backward {step}:', memory_summary(device))
    optimizer.step()

    metrics['Loss'] += loss.item() * batch_size
    metrics['ReconLoss'] += recon_loss.item() * batch_size

  def run_validation(model, dl):
    model.eval()
    with torch.no_grad():
      log.DEBUG('Start of validation', memory_summary(device))
      assert(len(dl.dataset) > 0)
      m = defaultdict(int)
      for im1, im2, labels in dl:
        batch_size, C, H, W = im1.shape
        im1, im2 = im1.to(device), im2.to(device)
        input = torch.cat((im1, im2), dim=1)
        output, mask, flow, displacement = model(input)
        loss = sfmnet.dssim_loss(im2, output, reduction=torch.sum)

        # m['Loss/Recon']         += loss
        # m['Metric/MaskMass']    += torch.sum(torch.mean(mask, dim=(1,))) # Mask mass
        # m['Metric/DisplLength'] += torch.sum(torch.mean(torch.abs(displacement), dim=(1,))) # L1 displacements
        # m['Metric/MaskVar']     += torch.sum(torch.mean(mask * (1 - mask), dim=(1,))) # Mask var

        if 'camera_translation' in labels:
          ct = labels['camera_translation'].to(device)
          
          mse = torch.sum(torch.square(displacement[:,0] * torch.tensor([W/2, H/2], device=device) - ct))
          m['Label/CameraDisplMSE'] += mse
    model.train()
    for k,v in m.items():
      m[k] = v / len(dl.dataset)
    return m

  def reduce_metrics(metrics):
    nonlocal device
    t = torch.tensor((len(metrics.keys())), device=device)
    for i, v in enumerate(metrics.values()):
      t[i] = v
    dist.reduce(t)
    reduced = {}
    for i, k in enumerate(metrics.keys()):
      reduced[k] = t[i]
    return reduced

  step = 0
  for e in range(start_at_epoch, num_epochs):
    if isinstance(dl_train.sampler, torch.utils.data.DistributedSampler):
      dl_train.sampler.set_epoch(e)
    train_metrics = defaultdict(int)
    for im1, im2, labels in dl_train:
      run_step(im1, im2, labels, train_metrics)
      step += 1
    for k,v in train_metrics.items():
      train_metrics[k] = v / len(dl_train.dataset)

    validation_metrics = run_validation(model, dl_validation)
    if using_ddp:
      validation_metrics = reduce_metrics(validation_metrics)
      train_metrics = reduce_metrics(train_metrics)
    log_metrics(epoch=e, step=step, metric=validation_metrics, prefix='Validation/')
    log_metrics(epoch=e, step=step, metric=train_metrics, prefix='Train/')

    if checkpoint_file is not None and e % checkpoint_freq == 0:
      save(checkpoint_file, model, optimizer, e)

    if using_ddp:
      dist.barrier()

  return model

def train(*,
  data_dir,
  sliding_data=True,
  tensorboard_dir=None,
  checkpoint_file=None,
  checkpoint_freq=10,
  dl_num_workers=6,
  validation_split=0.1,
  seed=42,

  K=1,
  camera_translation=False,
  C=16,
  fc_layer_width=128,
  num_hidden_layers=1,
  conv_depth=2,

  lr=0.001,
  flowreg_coeff=0.,
  maskreg_coeff=0.,
  displreg_coeff=0.,
  forwbackwreg_coeff=0.,
  mask_logit_noise_curriculum=None,
  batch_size=16,

  num_epochs=1, 
  n_vis_point=None,
  vis_freq=50,

  using_ddp=False,
  debug=False,
):
  args = locals()
  assert(camera_translation == True)
  
  ds=PairConsecutiveFramesDataset(data_dir)
  im_channels, H, W = ds[0][0].shape
  
  model = sfmnet.SfMNet(H=H, W=W, im_channels=im_channels, \
    C=C, K=K, camera_translation=camera_translation, conv_depth=conv_depth, \
    hidden_layer_widths=[fc_layer_width]*num_hidden_layers \
  )
  n_params = model.total_params()
  
  if using_ddp:
    setup_dist()
    device = torch.device('cuda', 0)
    model = DDP(model.to(device), device_ids=[device])
  elif torch.cuda.is_available():
    device = torch.device('cuda', 0)
    model = model.to(device)
  else:
    device = torch.device('cpu')

  global log
  rank = get_rank()
  log = logger.logger(logger.LEVEL_INFO, rank)
  log.INFO('Initialized the model which has', n_params, 'parameters')
  if rank is 0:
    pprint.PrettyPrinter(indent=4).pprint(args)

  log.INFO('Training on', device)
  log.DEBUG(f'Inputs has size ({im_channels},{H},{W})')

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  start_at_epoch = 0
  if checkpoint_file is not None:
    start_at_epoch = load(checkpoint_file, model, optimizer)

  n_validation = int(len(ds) * validation_split)
  n_train = len(ds) - n_validation
  log.DEBUG(f'Validation size {n_validation}, train size {n_train}')
  ds_train, ds_validation = torch.utils.data.random_split(ds, [n_train, n_validation], generator=torch.Generator().manual_seed(seed))

  sampler_train = torch.utils.data.DistributedSampler(ds_train) if using_ddp else None
  sampler_validation = torch.utils.data.DistributedSampler(ds_validation, shuffle=False) if using_ddp else None
  dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, 
    shuffle=(sampler_train is None), sampler=sampler_train, num_workers=dl_num_workers, pin_memory=True)

  dl_validation = torch.utils.data.DataLoader(ds_validation, sampler=sampler_validation, 
    batch_size=batch_size, shuffle=False, num_workers=dl_num_workers, pin_memory=True)

  if tensorboard_dir is not None and rank is 0:
    writer = SummaryWriter(log_dir=tensorboard_dir)
    writer.add_text('model_summary', str(model))
  else:
    writer = None

  if n_vis_point is not None:
    vis_point = ds_validation[0:n_vis_point]
    vis_point = (vis_point[0].to(device), vis_point[1].to(device))
  else:
    vis_point = None

  best_validation = math.inf
  start_time = time.monotonic()
  def log_metrics(*, step, epoch, metric, prefix=''):
    nonlocal best_validation
    nonlocal start_time
    nonlocal rank
    if rank is not 0:
      return
    best_validation = min(best_validation, metric.get('Loss/Validation/Recon', math.inf))
    s = f'epoch: {epoch} step: {step} time_elapsed: {time.monotonic() - start_time:.2f}s '
    for k,v in metric.items():
      s += f'{prefix}{k}: {v:7f} '
    log.INFO(s)
    if writer is not None:
      for k,v in metric.items():
        writer.add_scalar(k, v, step)
    if writer is not None and vis_point is not None and epoch % vis_freq == 0:
      model.eval()
      fig = sfmnet.visualize(model, *vis_point)
      model.train()
      writer.add_figure(f'Visualization', fig, step)

  train_loop(
    device=device,
    model=model,
    dl_train=dl_train,
    dl_validation=dl_validation,
    optimizer=optimizer,
    flowreg_coeff=flowreg_coeff,
    maskreg_coeff=maskreg_coeff,
    displreg_coeff=displreg_coeff,
    forwbackwreg_coeff=forwbackwreg_coeff,
    mask_logit_noise_curriculum=mask_logit_noise_curriculum,
    num_epochs=num_epochs,
    start_at_epoch=start_at_epoch,
    log_metrics=log_metrics,
    using_ddp=using_ddp,
    checkpoint_file=checkpoint_file,
    checkpoint_freq=checkpoint_freq,
  )

  if writer is not None:
    writer.add_hparams({
      'lr':lr,
      'maskreg': maskreg_coeff,
      'displreg': displreg_coeff,
      'flowreg': flowreg_coeff,
      'forwbackwreg': forwbackwreg_coeff,
    },{
      'Validation/Recon': best_validation
    })

  if checkpoint_file is not None:
    save(checkpoint_file, model, optimizer, num_epochs)
  if using_ddp:
    cleanup_dist()
  # return model


def setup_dist():
  assert not torch.cuda.is_available or torch.cuda.device_count() == 1
  env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
  }
  print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
  dist.init_process_group(backend="nccl" if torch.cuda.is_available() else 'gloo', init_method='env://')
  print(
      f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
      + f"rank = {get_rank()}, backend={dist.get_backend()}"
  )

def cleanup_dist():
    dist.destroy_process_group()

if __name__=='__main__':
  fire.Fire(train)
  
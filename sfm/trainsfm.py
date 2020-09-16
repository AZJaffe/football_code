import argparse
import datetime
import fire
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import pprint
from torch.utils.tensorboard import SummaryWriter
import functools


import sfmnet
from pair_frames_dataset import PairConsecutiveFramesDataset

def load(checkpoint_file, model, optimizer, rank):
  if checkpoint_file is None:
    return
  try:
    map_location = {'cuda:0': f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'} 
    checkpoint = torch.load(checkpoint_file, map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Loaded from checkpoint at {checkpoint_file}')
    return
  except FileNotFoundError:
    return

def save(checkpoint_file, model, optimizer, rank):
  if rank != 0:
    return
  if checkpoint_file is None:
    return
  tmp_file = checkpoint_file + '.tmp'
  torch.save({
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }, tmp_file)
  os.replace(tmp_file, checkpoint_file)
  print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Checkpoint saved at {checkpoint_file}')

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
  maskvarreg_coeff=0.,
  num_epochs=1,
  epoch_callback=noop_callback,
  using_ddp=False,
):
  if forwbackw_data_augmentation is False and forwbackwreg_coeff > 0.:
    raise "bad args"

  step = 0

  for e in range(0, num_epochs):
    if isinstance(dl_train.sampler, torch.utils.data.DistributedSampler):
      print('setting epoch')
      dl_train.sampler.set_epoch(e)
    train_metrics = torch.zeros((2), dtype=torch.float32, device=device)
    model.train()
    for im1, im2 in dl_train:
      optimizer.zero_grad()
      im1, im2 = im1.to(device), im2.to(device)
      print(f'Start of train batch {step}:', torch.cuda.memory_summary(device))
      batch_size = im1.shape[0]
      forwardbatch = torch.cat((im1, im2), dim=1)
      if forwbackw_data_augmentation:
        backwardbatch = torch.cat((im2, im1), dim=1)
        input = torch.cat((forwardbatch, backwardbatch), dim=0)
        # The target of the first batch_size outputs is im2 and the target of the second B outputs is im1
        target = torch.cat((im2, im1), dim=0)
      else:
        input = forwardbatch
        target = im2
      output, mask, flow, displacement = model(input)
      print(f'After forward {step}:', torch.cuda.memory_summary(device))

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
      mask_var_reg = maskvarreg_coeff * sfmnet.mask_variance_regularization(mask)

      loss = recon_loss + flowreg + maskreg + displreg + forwbackwreg + mask_var_reg
      
      print(f'Before backward {step}:', torch.cuda.memory_summary(device))
      loss.backward()
      print(f'After backward {step}:', torch.cuda.memory_summary(device))
      optimizer.step()

      train_metrics[0] += loss.item() * input.shape[0]
      train_metrics[1] += recon_loss.item() * input.shape[0]
      step += 1
    
    with torch.no_grad():
      # Just evaluate the reconstruction loss for the validation set
      model.eval()
      print('Start of validation', torch.cuda.memory_summary(device))
      assert(len(dl_validation.dataset) > 0) # TODO allow no validation
      validation_metrics = torch.zeros((4), device=device, dtype=torch.float32)
      for im1,im2 in dl_validation:
        batch_size = im1.shape[0]
        im1, im2 = im1.to(device), im2.to(device)
        input = torch.cat((im1, im2), dim=1)
        output, mask, flow, displacement = model(input)
        loss = sfmnet.dssim_loss(im2, output, reduction=torch.sum)

        validation_metrics[0]   += loss
        validation_metrics[1]   += torch.sum(torch.mean(mask, dim=(1,)))
        validation_metrics[2]   += torch.sum(torch.mean(torch.abs(displacement), dim=(1,)))
        validation_metrics[3]   += 0 # TODO mask_var

    if using_ddp:
      dist.reduce(validation_metrics, 0)
      dist.reduce(train_metrics, 0)
    if not using_ddp or dist.get_rank() is 0:
      len_train_ds = len(dl_train.dataset) * (2 if forwbackwreg_coeff else 1)
      len_validate_ds = len(dl_validation.dataset)

      epoch_loss, epoch_recon_loss = train_metrics / len_train_ds
      avg_loss, avg_mask_mass, avg_displ_length, avg_mask_var = validation_metrics / len_validate_ds

      epoch_callback(epoch=e, step=step, metric={
        'Loss/Train/Recon': epoch_recon_loss,
        'Loss/Train/Total': epoch_loss,
        'Loss/Validation/Recon': avg_loss,
        'Metric/MaskMass': avg_mask_mass,
        'Metric/DisplLength': avg_displ_length,
        'Metric/MaskVar': avg_mask_var,
      })
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

  K=1,
  C=16,
  fc_layer_width=128,
  num_hidden_layers=1,
  conv_depth=2,

  lr=0.001,
  flowreg_coeff=0.,
  maskreg_coeff=0.,
  displreg_coeff=0.,
  forwbackwreg_coeff=0.,
  maskvarreg_coeff=0.,
  batch_size=16,

  num_epochs=1, 
  n_vis_point=None,
  vis_freq=50,

  using_ddp=False,
):
  args = locals()
  pprint.PrettyPrinter(indent=4).pprint(args)

  ds=PairConsecutiveFramesDataset(data_dir)
  im_channels, H, W = ds[0][0].shape
  model = sfmnet.SfMNet(H=H, W=W, im_channels=im_channels, \
    C=C, K=K, conv_depth=conv_depth, \
    hidden_layer_widths=[fc_layer_width]*num_hidden_layers \
  )
  print('Initialized the model which has', model.total_params(), 'parameters')

  if using_ddp:
    setup_dist()
    rank = dist.get_rank()
    device = torch.device('cuda', 0)
    model = DDP(model.to(device), device_ids=[device])
  elif torch.cuda.is_available():
    rank = 0
    device = torch.device('cuda', 0)
    model = model.to(device)
  else:
    rank = 0
    device = torch.device('cpu')

  print('Training on', device)
  print('At initialization:', torch.cuda.memory_summary(device))

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  if checkpoint_file is not None:
    load(checkpoint_file, model, optimizer, rank)

  n_validation = int(len(ds) * validation_split)
  n_train = len(ds) - n_validation
  print(f'Validation size {n_validation}, train size {n_train}')
  ds_train, ds_validation = torch.utils.data.random_split(ds, [n_train, n_validation], generator=torch.Generator().manual_seed(42))

  sampler_train = torch.utils.data.DistributedSampler(ds_train) if using_ddp else None
  sampler_validation = torch.utils.data.DistributedSampler(ds_validation, shuffle=False) if using_ddp else None
  dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, 
    shuffle=(sampler_train is None), sampler=sampler_train, num_workers=dl_num_workers, pin_memory=True)

  dl_validation = torch.utils.data.DataLoader(ds_validation, sampler=sampler_validation, 
    batch_size=2*batch_size, shuffle=False, num_workers=dl_num_workers, pin_memory=True)

  if tensorboard_dir is not None and rank is 0:
    sample_input = torch.cat((ds[0][0], ds[0][1]))
    writer = SummaryWriter(log_dir=tensorboard_dir)
    writer.add_graph(model, sample_input.to(device).unsqueeze(0))
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
  def epoch_callback(*, step, epoch, metric):
    nonlocal best_validation
    nonlocal start_time
    best_validation = min(best_validation, metric.get('Loss/Validation/Recon', math.inf))
    s = f'epoch: {epoch} time_elapsed: {time.monotonic() - start_time:.2f}s '
    for k,v in metric.items():
      s += f'{k}: {v:7f} '
    print(s)
    if writer is not None:
      for k,v in metric.items():
        writer.add_scalar(k, v, step)
    if checkpoint_file is not None and epoch % checkpoint_freq == 0:
      save(checkpoint_file, model, optimizer, rank)
    if writer is not None and vis_point is not None and epoch % vis_freq == 0:
      model.eval()
      fig = sfmnet.visualize(model, *vis_point)
      fig.show()
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
    num_epochs=num_epochs,
    epoch_callback=epoch_callback,
    using_ddp=using_ddp
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
    save(checkpoint_file, model, optimizer, rank)
  if using_ddp:
    cleanup_dist()
  # return model


def setup_dist():
  env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
  }
  print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
  dist.init_process_group(backend="nccl" if torch.cuda.is_available() else 'gloo', init_method='env://')
  print(
      f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
      + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
  )

def cleanup_dist():
    dist.destroy_process_group()

if __name__=='__main__':
  fire.Fire(train)
  
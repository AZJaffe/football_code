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
import kitti_dataset


import sfmnet
from pair_frames_dataset import PairConsecutiveFramesDataset

log = logger.noop


def get_rank():
  return dist.get_rank() if dist.is_initialized() else 0


def load(checkpoint_file, model, optimizer):
  # Returns the epoch number to start at, as well as loads the optimizer and model
  if checkpoint_file is None:
    return
  try:
    map_location = {
      'cuda:0': f'cuda:{get_rank()}' if torch.cuda.is_available() else 'cpu'}
    checkpoint = torch.load(checkpoint_file, map_location)
    model = sfmnet.SfMNet3D.load_from_params(checkpoint['model_hyperparams'], checkpoint['model_state_dict'])
    if model == None:
      model = sfmnet.SfMNet2D.load_from_params(checkpoint['model_hyperparams'], checkpoint['model_state_dict'])
    if model == None:
      raise Exception('Cannot load checkpoint', checkpoint_file)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log.INFO(
      f'RANK {get_rank()}: Loaded from checkpoint at {checkpoint_file}')
    if 'epoch' in checkpoint:
      return checkpoint['epoch']
    else:
      return 0
  except FileNotFoundError:
    return 0


def save(checkpoint_file, model, optimizer, epoch):
  if get_rank() != 0:
    return
  if checkpoint_file is None:
    return
  tmp_file = checkpoint_file + '.tmp'
  torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_hyperparams': model.get_params(),
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
         dl_validation=None,
         vis_point=None,
         train_model,
         validation_model,
         optimizer,
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

  def finalize_metrics(metrics):
    if using_ddp:
      metrics = reduce_metrics(metrics)
    if get_rank() == 0:
      metrics = normalize_metrics(metrics)
    return metrics

  def reduce_metrics(metrics):
    nonlocal device
    t = torch.empty(len(metrics.keys()), device=device)
    for i, v in enumerate(metrics.values()):
      t[i] = v
    dist.reduce(t, dst=0)
    reduced = {}
    for i, k in enumerate(metrics.keys()):
      reduced[k] = t[i]
    return reduced

  def normalize_metrics(m):
    for k, v in m.items():
      if k == 'num_samples':
        continue
      m[k] = v / m['num_samples']
    del m['num_samples']
    return m

  def update_metrics(metrics, *, labels, im_estimate, out, loss=None, recon_loss):
    mask = out['mask']
    N, K, H, W = mask.shape
    metrics['num_samples'] += N
    if loss is not None:
      metrics['Loss/Total'] += loss
    metrics['Loss/Recon'] += recon_loss
    # metrics['Metric/MaskMass']  += torch.sum(torch.mean(mask, dim=(1,))) # Mask mass
    # metrics['Metric/DisplLength'] += torch.mean(
    #   torch.sum(
    #     torch.abs(displacement * torch.tensor([W/2, H/2] if displacement.shape[-1] == 2 else displacement, device=device)),
    #   dim=(0,2,)
    #   ),
    # ) # Mean over the # of objects in the scene
    # metrics['Metric/MaskVar']   += torch.sum(torch.mean(mask * (1 - mask), dim=(1,))) # Mask var

    if 'camera_translation' in labels and out.get('displacement') != None:
      displacement = out.get('displacement')
      ct = labels['camera_translation'].to(device)
      H, W = tuple(mask.shape[2:4])
      # Need this in case using forwbackw and so batch size of displacement is 2*M = N
      M = ct.shape[0]
      ae = torch.sum(torch.abs(
        displacement[0:M, 0] * torch.tensor([W/2, H/2], device=device) - ct)) * N / M
      metrics['Label/CameraDisplAE'] += ae

  def run_step(im1, im2, labels, metrics):
    optimizer.zero_grad()
    im1, im2 = im1.to(device), im2.to(device)
    log.DEBUG(f'Start of train batch {step}:', memory_summary(device))
    batch_size, C, H, W = im1.shape
    total_loss, recon_loss, im2_estimate, out = train_model(
      im1, im2, mask_logit_noise_var=get_mask_logit_noise(e))
    log.DEBUG(f'After forward {step}:', memory_summary(device))
    total_loss.backward()
    log.DEBUG(f'After backward {step}:', memory_summary(device))
    optimizer.step()

    update_metrics(
      metrics,
      loss=total_loss.item() * batch_size,
      recon_loss=recon_loss.item() * batch_size,
      im_estimate=im2_estimate,
      labels=labels,
      out=out
    )

  def run_validation(model, dl):
    model.eval()
    with torch.no_grad():
      log.DEBUG('Start of validation', memory_summary(device))
      assert(len(dl.dataset) > 0)
      m = defaultdict(int)
      for im1, im2, labels in dl:
        N, C, H, W = im1.shape
        im1, im2 = im1.to(device), im2.to(device)
        total_loss, recon_loss, im2_estimate, out = validation_model(
          im1, im2, reduction=torch.sum)

        update_metrics(
          metrics=m,
          labels=labels,
          recon_loss=recon_loss,
          out=out,
          im_estimate=im2_estimate
        )

    model.train()
    return m

  step = 0
  for e in range(start_at_epoch, num_epochs):
    if isinstance(dl_train.sampler, torch.utils.data.DistributedSampler):
      dl_train.sampler.set_epoch(e)
    if dl_validation is not None:
      validation_metrics = run_validation(
        validation_model, dl_validation)
      validation_metrics = finalize_metrics(validation_metrics)

    train_metrics = defaultdict(int)
    for im1, im2, labels in dl_train:
      run_step(im1, im2, labels, train_metrics)
      step += 1
    train_metrics = finalize_metrics(train_metrics)

    if dl_validation is not None:
      log_metrics(epoch=e, step=step,
            metric=validation_metrics, prefix='Validation/')
    log_metrics(epoch=e, step=step, metric=train_metrics, prefix='Train/')

    if checkpoint_file is not None and e % checkpoint_freq == 0:
      save(checkpoint_file, train_model, optimizer, e)

    if using_ddp:
      dist.barrier()


def train(*,
      data_dir,
      dataset_type,
      rgb=True, # Used for kitti dataset

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

      depth_smooth_reg=0.,
      flow_smooth_reg=0.,
      mask_smooth_reg=0., 
      flowreg_coeff=0.,
      forwbackw_reg_coeff=0.,
      dimension=3, # Either 2dsfm or 3dsfm

      lr=0.001,
      mask_logit_noise_curriculum=None,
      batch_size=16,

      num_epochs=1,
      n_vis_point=None,
      vis_freq=50,

      using_ddp=False,
      debug=False,
      returning=False,
      ):
  args = locals()

  if dataset_type == 'consecutive':
    ds = PairConsecutiveFramesDataset(data_dir)
  if dataset_type == 'kitti_stereo':
    ds = kitti_dataset.CollectionKittiRawStereoDataset(data_dir, rgb)
  else:
    raise f'dataset_type {dataset_type} not supported'

  im_channels, H, W = ds[0][0].shape

  # sfm is the only model with parameters. The validation_model and train_model return the self-supervised
  # loss for training purposes.

  if dimension == 2:
    sfm = sfmnet.SfMNet2D(H=H, W=W, im_channels=im_channels,
                C=C, K=K, camera_translation=camera_translation, conv_depth=conv_depth,
                hidden_layer_widths=[
                  fc_layer_width]*num_hidden_layers
                )
  if dimension == 3:
      sfm = sfmnet.SfMNet3D(H=H, W=W, im_channels=im_channels,
              C=C, K=K, conv_depth=conv_depth,
              hidden_layer_widths=[
                fc_layer_width]*num_hidden_layers
              )

  validation_model = sfmnet.LossModule(sfm_model=sfm, 
    l1_flow_reg_coeff=flowreg_coeff,
    depth_smooth_reg=depth_smooth_reg,
    flow_smooth_reg=flow_smooth_reg,
    mask_smooth_reg=mask_smooth_reg, 
  )

  if forwbackw_reg_coeff != 0.:
    train_model = sfmnet.ForwBackwLoss(
      validation_model, forwbackw_reg_coeff)
  else:
    train_model = validation_model

  n_params = sfm.total_params()

  if using_ddp:
    setup_dist()
    device = torch.device('cuda', 0)
    model = DDP(train_model.to(device), device_ids=[device])
  elif torch.cuda.is_available():
    device = torch.device('cuda', 0)
    model = train_model.to(device)
  else:
    device = torch.device('cpu')

  global log
  rank = get_rank()
  if rank is 0:
    pprint.PrettyPrinter(indent=4).pprint(args)
  log = logger.logger(logger.LEVEL_INFO, rank)
  log.INFO('Initialized the model which has', n_params, 'parameters')
  log.INFO('Dataset has size', len(ds))
  
  log.INFO('Training on', device)
  log.DEBUG(f'Inputs has size ({im_channels},{H},{W})')

  optimizer = torch.optim.Adam(sfm.parameters(), lr=lr)
  start_at_epoch = 0
  if checkpoint_file is not None:
    start_at_epoch = load(checkpoint_file, model, optimizer)

  n_validation = int(len(ds) * validation_split)
  n_train = len(ds) - n_validation
  log.DEBUG(f'Validation size {n_validation}, train size {n_train}')
  ds_train, ds_validation = torch.utils.data.random_split(
    ds, [n_train, n_validation], generator=torch.Generator().manual_seed(seed))

  sampler_train = torch.utils.data.DistributedSampler(
    ds_train) if using_ddp else None
  sampler_validation = torch.utils.data.DistributedSampler(
    ds_validation, shuffle=False) if using_ddp else None
  dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,
                       shuffle=(sampler_train is None), sampler=sampler_train, num_workers=dl_num_workers, pin_memory=True)

  dl_validation = torch.utils.data.DataLoader(ds_validation, sampler=sampler_validation,
                        batch_size=batch_size, shuffle=False, num_workers=dl_num_workers, pin_memory=True)

  if tensorboard_dir is not None and rank is 0:
    writer = SummaryWriter(log_dir=tensorboard_dir)
    writer.add_text('model_summary', str(sfm))
  else:
    writer = None

  if n_vis_point is not None:
    vis_dl = torch.utils.data.DataLoader(ds_validation, batch_size=n_vis_point, shuffle=False)
    vis_point = next(iter(vis_dl))

  else:
    vis_point = None

  best_validation = math.inf
  start_time = time.monotonic()

  def log_metrics(*, step, epoch, metric, prefix=''):
    nonlocal best_validation
    nonlocal start_time
    nonlocal rank
    if rank != 0:
      return
    best_validation = min(best_validation, metric.get(
      'Loss/Validation/Recon', math.inf))
    s = f'epoch: {epoch} step: {step} time_elapsed: {time.monotonic() - start_time:.2f}s '
    for k, v in metric.items():
      s += f'{prefix}{k}: {v:7f} '
    log.INFO(s)
    if writer is not None:
      for k, v in metric.items():
        writer.add_scalar(prefix+k, v, step)

    if vis_point is not None and epoch % vis_freq == 0:
      validation_model.eval()
      vp = (vis_point[0].to(device), vis_point[1].to(device))
      fig = sfmnet.visualize(validation_model, *vp)
      validation_model.train()
      if writer is not None:
        writer.add_figure(f'Visualization', fig, step)
      else:
        pass
        # plt.show()

  train_loop(
    device=device,
    validation_model=validation_model,
    train_model=train_model,
    dl_train=dl_train,
    dl_validation=dl_validation,
    optimizer=optimizer,
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
      'lr': lr,
      'flowreg': flowreg_coeff,
    }, {
      'Validation/Recon': best_validation
    })

  if checkpoint_file is not None:
    save(checkpoint_file, model, optimizer, num_epochs)
  if using_ddp:
    cleanup_dist()

  if returning:
    return sfm


def setup_dist():
  assert not torch.cuda.is_available or torch.cuda.device_count() == 1
  env_dict = {
    key: os.environ[key]
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
  }
  print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
  dist.init_process_group(
    backend="nccl" if torch.cuda.is_available() else 'gloo', init_method='env://')
  print(
    f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
    + f"rank = {get_rank()}, backend={dist.get_backend()}"
  )


def cleanup_dist():
  dist.destroy_process_group()


if __name__ == '__main__':
  fire.Fire(train)

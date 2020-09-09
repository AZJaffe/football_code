import argparse
import datetime
import fire
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import math
import pprint
from torch.utils.tensorboard import SummaryWriter

import sfmnet
from pair_frames_dataset import PairConsecutiveFramesDataset

def load(checkpoint_file, model, optimizer):
  if checkpoint_file is None:
    return 0
  try:
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Loaded from checkpoint at {checkpoint_file}')
    return
  except FileNotFoundError:
    return

def save(checkpoint_file, model, optimizer):
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
  flowreg_coeff=0.,
  maskreg_coeff=0.,
  displreg_coeff=0.,
  forwbackwreg_coeff=0.,
  maskvarreg_coeff=0.,
  num_epochs=1,
  epoch_callback=noop_callback,
):

  step = 0
  for e in range(0, num_epochs):
    total_loss = 0.
    total_recon_loss = 0.
    for im1, im2 in dl_train:
      im1, im2 = im1.to(device), im2.to(device)
      batch_size = im1.shape[0]
      forwardbatch = torch.cat((im1, im2), dim=1)
      backwardbatch = torch.cat((im2, im1), dim=1)
      input = torch.cat((forwardbatch, backwardbatch), dim=0)
      output, mask, flow, displacement = model(input)

      # The target of the first B outputs is im2 and the target of the second B outputs is im1
      recon_loss = sfmnet.dssim_loss(torch.cat((im2, im1), dim=0), output)
      # backward forward regularization induces a prior on the output of the network
      # that encourages the output from going forward in time is consistent with
      # the output from going backward in time.
      # Precisely, the masks should be the same and the displacements should be the negations of each other 
      forwbackwreg = forwbackwreg_coeff * (
        torch.mean(torch.sum(torch.abs(mask[0:batch_size] - mask[batch_size: 2*batch_size]), dim=(1,)))
        + torch.mean(torch.sum(torch.square(displacement[0:batch_size] + displacement[batch_size: 2*batch_size]), dim=(1,2)))
      )
      flowreg = flowreg_coeff * sfmnet.l1_flow_regularization(mask, displacement)
      maskreg = maskreg_coeff * sfmnet.l1_mask_regularization(mask)
      displreg = displreg_coeff * sfmnet.l2_displacement_regularization(displacement)
      mask_var_reg = maskvarreg_coeff * sfmnet.mask_variance_regularization(mask)

      loss = recon_loss + flowreg + maskreg + displreg + forwbackwreg + mask_var_reg

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # 2 * because the forward model was ran thru with (im1,im2) and (im2, im1), twice the size of the batch
      total_loss       += loss * 2 * batch_size 
      total_recon_loss += recon_loss * 2 * batch_size

      step += 1
    
    len_train_ds = len(dl_train.dataset)
    len_validate_ds = len(dl_validation.dataset)
    epoch_recon_loss = total_recon_loss / (2 * len_train_ds)
    epoch_loss = total_loss / (2 * len_train_ds)
    with torch.no_grad():
      # Just evaluate the reconstruction loss for the validation set
      assert(len(dl_validation.dataset) > 0) # TODO allow no validation
      model.eval()
      total_loss = 0.
      mask_mass = 0.
      displ_length = 0.
      for im1,im2 in dl_validation:
        batch_size = im1.shape[0]
        input = torch.cat((im1, im2), dim=1)
        output, mask, flow, displacement = model(input)
        loss = sfmnet.dssim_loss(im2, output)

        total_loss    += loss * batch_size
        mask_mass     += torch.sum(torch.mean(mask, dim=(1,)))
        displ_length  += torch.sum(torch.mean(torch.abs(displacement), dim=(1,)))
      avg_loss = total_loss / len_validate_ds
      avg_mask_mass = mask_mass / len_validate_ds
      avg_displ_length = displ_length / len_validate_ds
      model.train()

    epoch_callback(epoch=e, step=step, metric={
      'Loss/Train/Recon': epoch_recon_loss,
      'Loss/Train/Total': epoch_loss,
      'Loss/Validation/Recon': avg_loss,
      'Mask_Mass/Validation': avg_mask_mass,
      'Displ_Length/Validation': avg_displ_length
    })

  return model

def train(*,
  data_dir,
  sliding_data=True,
  tensorboard_dir=None,
  checkpoint_file=None,
  checkpoint_freq=10,
  load_data_to_device=False,
  dl_num_workers=6,
  validation_split=0.1,
  disable_cuda=False,
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
):
  args = locals()
  pprint.PrettyPrinter(indent=4).pprint(args)
  if disable_cuda is False and torch.cuda.is_available():
    device = torch.device('cuda')
  else:                         
    device = torch.device('cpu')
  print('training on ' + device.type)

  ds=PairConsecutiveFramesDataset(data_dir, load_all=load_data_to_device, device=device)
  n_validation = int(len(ds) * validation_split)
  n_train = len(ds) - n_validation
  print(f'Validation size {n_validation}, train size {n_train}')
  ds_train, ds_validation = torch.utils.data.random_split(ds, [n_train, n_validation], generator=torch.Generator().manual_seed(42))
  dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=dl_num_workers, pin_memory=True)
  dl_validation = torch.utils.data.DataLoader(ds_validation, batch_size = 2 * batch_size, shuffle=False, num_workers=dl_num_workers, pin_memory=True)
  im_channels, H, W = ds[0][0].shape

  model = sfmnet.SfMNet(H=H, W=W, im_channels=im_channels, \
    C=C, K=K, conv_depth=conv_depth, \
    hidden_layer_widths=[fc_layer_width]*num_hidden_layers \
  )
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  
  if checkpoint_file is not None:
    load(checkpoint_file, model, optimizer)

  if tensorboard_dir is not None:
    sample_input = torch.cat((ds[0][0], ds[0][1]))
    writer = SummaryWriter(log_dir=tensorboard_dir)
    writer.add_graph(model, sample_input.unsqueeze(0))
    writer.add_text('model_summary', str(model))
  else:
    writer = None

  if n_vis_point is not None:
    vis_point = ds_validation[0:n_vis_point]
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
      save(checkpoint_file, model, optimizer)
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
    epoch_callback=epoch_callback
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
    save(checkpoint_file, model, optimizer)
  # return model

if __name__=='__main__':
  m = fire.Fire(train)

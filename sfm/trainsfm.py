import argparse
import datetime
import fire
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import math
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
  dl,
  vis_point=None,
  model,
  optimizer,
  flowreg_coeff=0.,
  maskreg_coeff=0.,
  displreg_coeff=0.,
  forwbackwreg_coeff=0.,
  num_epochs=1,
  epoch_callback=noop_callback
):

  sample_input = torch.cat((dl.dataset[0][0], dl.dataset[0][1]))
  step = 0
  len_ds = len(dl.dataset[0])
  start_time = time.monotonic()
  for e in range(0, num_epochs):
    epoch_start_time = time.monotonic()
    total_loss = 0.
    total_recon_loss = 0.
    for im1, im2 in dl:
      batch_size = im1.shape[0]
      forwardbatch = torch.cat((im1, im2), dim=1)
      backwardbatch = torch.cat((im2, im1), dim=1)
      input = torch.cat((forwardbatch, backwardbatch), dim=0)
      output, mask, flow, displacement = model(input)

      # The target of the first B outputs is im2 and the target of the second B outputs is im1
      recon_loss = sfmnet.l1_recon_loss(torch.cat((im2, im1), dim=0), output) 
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

      loss = recon_loss + flowreg + maskreg + displreg + forwbackwreg

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # 2 * because the forward model was ran thru with (im1,im2) and (im2, im1), twice the size of the batch
      total_loss       += loss * 2 * batch_size 
      total_recon_loss += recon_loss * 2 * batch_size

      step += 1
    epoch_recon_loss = total_recon_loss / len_ds
    epoch_loss = total_loss / len_ds
    print(f'epoch: {e} recon_loss: {epoch_recon_loss:.5f} loss: {epoch_loss:.5f} total_time: {time.monotonic() - start_time:.2f}s epoch_time: {time.monotonic() - epoch_start_time:.2f}s')

    epoch_callback(epoch=e, step=step, metric={
      'recon_loss': epoch_recon_loss,
      'total_loss': epoch_loss,
    })

  return model

def train(*,
  data_dir,
  sliding_data=True,
  tensorboard_dir=None,
  checkpoint_file=None,
  checkpoint_freq=None,
  load_data_to_device=True,
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
  batch_size=16, 
  num_epochs=1, 
  n_vis_point=None,
  vis_freq=50,
):
  print(locals())
  if disable_cuda is False and torch.cuda.is_available():
    device = torch.device('cuda')
  else:                         
    device = torch.device('cpu')
  print('training on ' + device.type)

  ds=PairConsecutiveFramesDataset(data_dir, load_all=load_data_to_device, device=device)
  dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
  im_channels, H, W = ds[0][0].shape

  model = sfmnet.SfMNet(H=H, W=W, im_channels=im_channels, \
    C=C, K=K, conv_depth=conv_depth, \
    hidden_layer_widths=[fc_layer_width]*num_hidden_layers \
  )
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
    vis_point = ds[0:n_vis_point]
  else:
    vis_point = None

  best_recon_loss = 100
  def epoch_callback(*, step, epoch, metric):
    nonlocal best_recon_loss
    if best_recon_loss > metric['recon_loss']:
      best_recon_loss = metric['recon_loss']
    if writer is not None:
      writer.add_scalars('Loss', {
        'reconstruction': metric['recon_loss'],
        'total': metric['total_loss'],
      }, step)
    if checkpoint_file is not None and epoch % checkpoint_freq == 0:
      save(checkpoint_file, model, optimizer)
    if vis_point is not None and epoch % vis_freq == 0:
      fig = sfmnet.visualize(model, *vis_point)
      writer.add_figure(f'Visualization', fig, step)

  train_loop(
    model=model,
    dl=dl,
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
      'hparam/reconloss': best_recon_loss
    })

  if checkpoint_file is not None:
    save(checkpoint_file, model, optimizer)

if __name__=='__main__':
  m = fire.Fire(train)

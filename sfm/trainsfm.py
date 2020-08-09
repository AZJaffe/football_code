import argparse
import datetime
import fire
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
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

def save(checkpoint_file, model, optimizer, e):
  if checkpoint_file is None:
    return
  tmp_file = checkpoint_file + '.tmp'
  torch.save({
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }, tmp_file)
  os.replace(tmp_file, checkpoint_file)
  print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Checkpoint saved at {checkpoint_file}')

def train_loop(*,
  dl,
  vis_point=None,
  model,
  optimizer,
  tensorboard_dir=None,
  checkpoint_file=None,
  checkpoint_freq=None,
  vis_freq=1000,
  flowreg_coeff=0.,
  maskreg_coeff=0.,
  displreg_coeff=0.,
  num_epochs=1,
):
  input_sample = dl.dataset[0]
  input_shape = input_sample.shape
  im_channels = input_shape[0] // 2
  def loss_fn(input, output, masks, flow, displacements):
    recon_loss = sfmnet.l1_recon_loss(input[:,im_channels:im_channels*2], output)
    flowreg = flowreg_coeff * sfmnet.l1_flow_regularization(masks, displacements)
    maskreg = maskreg_coeff * sfmnet.l1_mask_regularization(masks)
    displreg = displreg_coeff * sfmnet.l2_displacement_regularization(displacements)

    return recon_loss + flowreg + maskreg + displreg, recon_loss

  if vis_point is not None:
    cpu_vis_point = vis_point.cpu()

  with SummaryWriter(tensorboard_dir) as writer:
    writer.add_graph(model, input_sample.unsqueeze(0))
    writer.add_text('model_summary', str(model))
    # writer.add_scalars('hparams', {
    #   'lr':lr,
    #   'batch_size': batch_size,
    #   'flow_reg_coeff': flow_reg_coeff,
    #   'conv_depth': conv_depth,
    #   'fc_layer_width': fc_layer_width,
    #   'K': K,
    # })
    start_time = time.monotonic()
    step = 0
    for e in range(0, num_epochs):
      epoch_start_time = time.monotonic()
      total_loss = 0.
      total_recon_loss = 0.
      for batch in dl:
        batch_size = batch.shape[0]
        output, masks, flows, displacements = model(batch)
        loss, recon_loss = loss_fn(batch, output, masks, flows, displacements)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss       += loss * batch_size
        total_recon_loss += recon_loss * batch_size

        if checkpoint_file is not None and step % checkpoint_freq == 0:
          save(checkpoint_file, model, optimizer)
        if vis_point is not None and e % vis_freq == 0:
          with torch.no_grad():
            output, mask, flow, displacement = model(vis_point)
            output, mask, flow, displacement = output.cpu(), mask.cpu(), flow.cpu(), displacement.cpu()
            fig = sfmnet.visualize(cpu_vis_point, output, mask, flow, displacement)
            writer.add_figure(f'Visualization', fig, step)
        step += 1

      with torch.no_grad():
        print(f'epoch: {e} recon_loss: {total_recon_loss / len(dl.dataset):.5f} loss: {total_loss / len(dl.dataset):.5f} total_time: {time.monotonic() - start_time:.2f}s epoch_time: {time.monotonic() - epoch_start_time:.2f}s')
        writer.add_scalars('Loss', {
          'reconstruction': total_recon_loss / len(dl.dataset),
          'total': total_loss / len(dl.dataset)
        }, step)

  
  if checkpoint_file is not None:
    save(checkpoint_file, model, optimizer)
  return model

def train(*,
  data_dir,
  sliding_data=True,
  tensorboard_dir=None,
  checkpoint_file=None,
  checkpoint_freq=None,
  vis_freq=10,
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
  batch_size=16, 
  num_epochs=1, 
  n_vis_point=5,
):
  print(locals())
  if disable_cuda is False and torch.cuda.is_available():
    device = torch.device('cuda')
  else:                         
    device = torch.device('cpu')
  print('training on ' + device.type)

  ds=PairConsecutiveFramesDataset(data_dir, load_all=load_data_to_device, device=device)

  input_shape = ds[0].shape
  im_channels = ds[0].shape[0] // 2
  model = sfmnet.SfMNet(H=input_shape[1], W=input_shape[2], im_channels=im_channels, \
    C=C, K=K, conv_depth=conv_depth, \
    hidden_layer_widths=[fc_layer_width]*num_hidden_layers \
  )
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  
  if checkpoint_file is not None:
    load(checkpoint_file, model, optimizer)

  dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

  train_loop(
    model=model,
    vis_point=ds[0:n_vis_point],
    dl=dl,
    optimizer=optimizer,
    tensorboard_dir=tensorboard_dir,
    checkpoint_file=None,
    checkpoint_freq=None,
    vis_freq=10,
    flowreg_coeff=flowreg_coeff,
    maskreg_coeff=maskreg_coeff,
    displreg_coeff=displreg_coeff,
    num_epochs=1, 
  )

if __name__=='__main__':
  m = fire.Fire(train)

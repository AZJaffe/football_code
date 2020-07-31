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
    return checkpoint['epoch']
  except FileNotFoundError:
    return 0

def save(checkpoint_file, model, optimizer, e):
  if checkpoint_file is None:
    return
  tmp_file = checkpoint_file + '.tmp'
  torch.save({
      'epoch': e,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }, tmp_file)
  os.replace(tmp_file, checkpoint_file)
  print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Checkpoint saved at {checkpoint_file}')

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
  conv_depth=2,
  lr=0.001,
  flowreg_coeff=0.,
  maskreg_coeff=0.,
  displreg_coeff=0.,
  batch_size=16, 
  num_epochs=1, 
):
  print(locals())
  if disable_cuda is False and torch.cuda.is_available():
    device = torch.device('cuda')
  else:                         
    device = torch.device('cpu')
  print('training on ' + device.type)

  ds=PairConsecutiveFramesDataset(data_dir, load_all=load_data_to_device, device=device)

  input_shape = ds[0].shape
  model = sfmnet.SfMNet(H=input_shape[1], W=input_shape[2], im_channels=input_shape[0], C=C, K=K, conv_depth=conv_depth, fc_layer_width=fc_layer_width)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  start_epoch = 0
  
  if checkpoint_file is not None:
    start_epoch = load(checkpoint_file, model, optimizer)

  dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
  
  def loss_fn(input, output, masks, flow, displacements):
    recon_loss = sfmnet.l1_recon_loss(input[:,3:6], output)
    flowreg = flowreg_coeff * sfmnet.l1_flow_regularization(masks, displacements)
    maskreg = maskreg_coeff * sfmnet.l1_mask_regularization(masks)
    displreg = displreg_coeff * sfmnet.l2_displacement_regularization(displacements)

    return recon_loss + flowreg + maskreg + displreg, recon_loss

  start_time = time.monotonic()
  test_points = ds[0:5]
  cpu_test_points = test_points.cpu()
  print(cpu_test_points.shape)
  model.to(device)

  with SummaryWriter(tensorboard_dir) as writer:
    writer.add_graph(model, ds[0].unsqueeze(0))
    writer.add_text('model_summary', str(model))
    # writer.add_scalars('hparams', {
    #   'lr':lr,
    #   'batch_size': batch_size,
    #   'flow_reg_coeff': flow_reg_coeff,
    #   'conv_depth': conv_depth,
    #   'fc_layer_width': fc_layer_width,
    #   'K': K,
    # })
    for e in range(start_epoch, start_epoch+num_epochs):
      if e % vis_freq == 0:
        with torch.no_grad():
          output, mask, flow, displacement = model(test_points)
          output, mask, flow, displacement = output.cpu(), mask.cpu(), flow.cpu(), displacement.cpu()
          for i in range(len(test_points)):
            fig = sfmnet.visualize(cpu_test_points[i], output[i], mask[i], flow[i])
            writer.add_figure(f'Visualization/test_point_{i}', fig, e * len(ds))
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

      with torch.no_grad():
        print(f'epoch: {e} loss: {total_loss / len(ds):.7f} total_time: {time.monotonic() - start_time:.2f}s epoch_time: {time.monotonic() - epoch_start_time:.2f}s')
        writer.add_scalars('Loss', {
          'reconstruction': total_recon_loss / len(ds),
          'total': total_loss / len(ds)
        }, e * len(ds))

      if checkpoint_file is not None and e % checkpoint_freq == 0 and e > 0:
        save(checkpoint_file, model, optimizer, e)
  
  if checkpoint_file is not None:
    save(checkpoint_file, model, optimizer, start_epoch+num_epochs)
  #return model

if __name__=='__main__':
  fire.Fire(train)

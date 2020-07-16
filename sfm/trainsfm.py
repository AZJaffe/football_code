import argparse
import datetime
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
  tensorboard_dir,
  ds, 
  lr=0.001,
  num_epochs=1, 
  flow_reg_coeff=0.,
  batch_size=16, 
  checkpoint_freq=100,
  checkpoint_file,
  device=torch.device('cpu'),
  fc_layer_width=128,
  conv_depth=2,
  K=1
):
  if ds is None:
    raise 'Pass a dataset'

  input_shape = ds[0].shape
  model = sfmnet.SfMNet(H=input_shape[1], W=input_shape[2], K=K, conv_depth=conv_depth, fc_layer_width=fc_layer_width)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  start_epoch = 0
  
  if checkpoint_file is not None:
    start_epoch = load(checkpoint_file, model, optimizer)

  dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
  
  def loss_fn(input, output, masks, flow, displacements):
    recon_loss = sfmnet.l1_recon_loss(input[:,3:6], output)
    flow_regularization = sfmnet.l1_flow_regularization(masks, displacements)
    return recon_loss + flow_reg_coeff * flow_regularization, recon_loss, flow_regularization

  start_time = time.monotonic()
  test_points = ds[0:5]
  model.to(device)

  with SummaryWriter(tensorboard_dir) as writer:
    writer.add_graph(model, ds[0].unsqueeze(0))
    writer.add_text('model_summary', str(model))
    for e in range(start_epoch, start_epoch+num_epochs):
      epoch_start_time = time.monotonic()
      total_loss = 0.
      total_recon_loss = 0.
      total_flow_reg = 0.
      for batch in dl:
        batch_size = batch.shape[0]
        output, masks, flows, displacements = model(batch)
        loss, recon_loss, flow_reg = loss_fn(batch, output, masks, flows, displacements)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss       += loss * batch_size
        total_recon_loss += recon_loss * batch_size
        total_flow_reg   += flow_reg * batch_size

      with torch.no_grad():
        print(f'epoch: {e} loss: {total_loss / len(ds):.7f} total_time: {time.monotonic() - start_time:.2f}s epoch_time: {time.monotonic() - epoch_start_time:.2f}s')
        writer.add_scalars('Loss', {
          'reconstruction': total_recon_loss / len(ds),
          'flow_regularization': total_flow_reg / len(ds),
          'total': total_loss / len(ds)
        }, e * len(ds))
        with torch.no_grad():
          output, masks, flows, displacements = model(test_points)
          for i in range(len(test_points)):
            fig = sfmnet.visualize(test_points[i].cpu(), output[i].cpu(), masks[i].cpu(), flows[i].cpu())
            writer.add_figure(f'Visualization/test_point_{i}', fig, e)

      if e % checkpoint_freq == 0 and e > 0:
        save(checkpoint_file, model, optimizer, e)
  
  save(checkpoint_file, model, optimizer, start_epoch+num_epochs)

if __name__=='__main__':
  # Parse args
  parser = argparse.ArgumentParser(description='Train an sfm')
  parser.add_argument('--data_dir', required=True,
                      help='the directory containing sequential images to use as data')
  parser.add_argument('--tensorboard_dir',
                      help='the directory to save tensorboard events')
  parser.add_argument('--checkpoint_freq', type=int, default=100,
                      help='the frequency in epochs of saving the model')
  parser.add_argument('--checkpoint_file',
                      help='the directory to save checkpoints')
                      
  parser.add_argument('--lr', type=float, default=0.001,
                      help='the learning rate of the Adam optimizer')
  parser.add_argument('--flow_reg_coeff', type=float, default=0.,
                      help='the flow regularization coefficient')
  parser.add_argument('--batch_size', type=int, default=16,
                      help='the batch size used for training')
  parser.add_argument('--num_epochs', type=int, default=2,
                      help='the number of epochs to train for')
  parser.add_argument('--disable_cuda', type=bool, default=False)

  parser.add_argument('--fc_layer_width', type=int, default=128,
                      help='the width of the fully connected layers of the model')
  parser.add_argument('--conv_depth', type=int, default=2,
                      help='the number of downsampling steps (1/2 of total number of convolutional encoding layers)')
  parser.add_argument('--K', type=int, default=2,
                      help='the number of objects to predict in the scene')

  args = parser.parse_args()

  if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')                  
  else:                                                   
    args.device = torch.device('cpu')

  print(f'using device type {args.device.type}')
  print(args)

  train(ds=PairConsecutiveFramesDataset(args.data_dir, load_all=True, device=args.device),
        tensorboard_dir=args.tensorboard_dir,
        checkpoint_file=args.checkpoint_file,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq,
        flow_reg_coeff=args.flow_reg_coeff,
        device=args.device,
        fc_layer_width=args.fc_layer_width,
        conv_depth=args.conv_depth,
        K=args.K
  )

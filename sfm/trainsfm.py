import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import sfmnet
from pair_frames_dataset import PairConsecutiveFramesDataset


def train(*, 
  load_model_path,
  out_dir, 
  ds, 
  lr=0.001,
  num_epochs=1, 
  flow_reg_coeff=0.,
  batch_size=16, 
  save_frequency=100,
  device=torch.device('cpu'),
):
  if ds is None:
    raise 'Pass a dataset'
  input_shape = ds[0].shape
  model = sfmnet.SfMNet(H=input_shape[1], W=input_shape[2], K=1, fc_layer_width=128)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  start_epoch = 0

  if load_model_path is not None:
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']


  dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
  
  def loss_fn(input, output, masks, flow, displacements):
    recon_loss = sfmnet.l1_recon_loss(input[:,3:6], output)
    flow_regularization = sfmnet.l1_flow_regularization(masks, displacements)
    return recon_loss + flow_reg_coeff * flow_regularization, recon_loss, flow_regularization

  start_time = time.monotonic()
  test_points = ds[0:5]

  with SummaryWriter(out_dir) as writer:
    writer.add_graph(model, ds[0].unsqueeze(0).to(device))
    writer.add_text('model_summary', str(model))
    model.to(device)
    for e in range(start_epoch, start_epoch+num_epochs):
      epoch_start_time = time.monotonic()
      total_loss = 0.
      total_recon_loss = 0.
      total_flow_reg = 0.
      for batch in dl:
        batch.to(device)
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
      if out_dir is not None and e % save_frequency == 0 and e > 0:
        with torch.no_grad():
          output, masks, flows, displacements = model(test_points)
          for i in range(len(test_points)):
            fig = sfmnet.visualize(test_points[i], output[i], masks[i], flows[i])
            writer.add_figure(f'Visualization/test_point_{i}', fig, e)
        torch.save({
          'epoch': e,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
        }, f'{out_dir}/model_{e}.pt')

  if out_dir is not None:
    torch.save({
      'epoch': start_epoch+num_epochs,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
    }, f'{out_dir}/model_{start_epoch+num_epochs}.pt')

if __name__=='__main__':
  # Parse args
  parser = argparse.ArgumentParser(description='Train an sfm')
  parser.add_argument('--data_dir', required=True,
                      help='the directory containing sequential images to use as data')
  parser.add_argument('--batch_size', type=int, default=16,
                      help='the batch size used for training')
  parser.add_argument('--num_epochs', type=int, default=2,
                      help='the number of epochs to train for')
  parser.add_argument('--out_dir',
                      help='the directory to save model checkpoints and training stats to')
  parser.add_argument('--model',
                      help='the path to load a model to resume training with')
  parser.add_argument('--save_freq', type=int, default=100,
                      help='the frequency in epochs of saving the model')
  parser.add_argument('--lr', type=float, default=0.001,
                      help='the learning rate of the Adam optimizer')
  parser.add_argument('--flow_reg_coeff', type=float, default=0.,
                      help='the flow regularization coefficient')
  parser.add_argument('--disable_cuda', type=bool, default=False)
  args = parser.parse_args()

  if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')                  
  else:                                                   
    args.device = torch.device('cpu')

  print(f'using device type {args.device.type} with index {args.device.index}')

  train(load_model_path=args.model,
        ds=PairConsecutiveFramesDataset(args.data_dir, load_all=True, device=args.device),
        out_dir=args.out_dir,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_frequency=args.save_freq,
        flow_reg_coeff=args.flow_reg_coeff,
        device=args.device,
  )

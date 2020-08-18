import trainsfm
import sfmnet
import pair_frames_dataset
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import imageio
import datetime
import sys
import math
from torch.utils.tensorboard import SummaryWriter


im_channels = 1
H = 4
W = 6
data = torch.ones((8, 2*im_channels, H, W))
data[0, 0, 1, 1] = 0
data[0, 1, 0, 1] = 0
data[1, 0, 2, 4] = 0
data[1, 1, 1, 5] = 0
data[2, 0, 2, 3] = 0
data[2, 1, 2, 4] = 0
data[3, 0, 1, 1] = 0
data[3, 1, 2, 2] = 0
data[4, 0, 1, 1] = 0
data[4, 1, 2, 1] = 0
data[5, 0, 2, 2] = 0
data[5, 1, 3, 1] = 0
data[6, 0, 1, 4] = 0
data[6, 1, 1, 3] = 0
data[7, 0, 2, 4] = 0
data[7, 1, 1, 3] = 0
dataset = (data[:,0:1], data[:,1:2])
dl = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)


def train_model(*, lr, hidden_layer_widths, flowreg_coeff, maskreg_coeff, displreg_coeff,
                forwbackwreg_coeff):

    tensorboard_dir = sys.argv[1]
    num_epochs = 300
    model = sfmnet.SfMNet(H=H, W=W, im_channels=im_channels,
                          C=4, K=1, conv_depth=1,
                          hidden_layer_widths=hidden_layer_widths
                          )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    def epoch_callback(*, step, epoch, metric):
        writer.add_scalars('Loss', {
            'recon': metric['recon_loss'],
            'total': metric['total_loss'],
        }, step)

    trainsfm.train_loop(
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
    with torch.no_grad():
      output, _, _, _ = model(data)
      loss = sfmnet.l1_recon_loss(dataset[1], output, reduction=None)

    writer.add_hparams({
      'lr':lr,
      'maskreg': maskreg_coeff,
      'displreg': displreg_coeff,
      'flowreg': flowreg_coeff,
      'forwbackwreg': forwbackwreg_coeff,
    },{
      'hparam/reconloss': torch.mean(loss),
      'hparam/numgood': torch.sum(loss < 1 / 24)
    })


lrs = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.1]
regs = [
  (0.1, 0.1, 0., 0.),
  (0.01, 0.01, 0., 0.),
  (0.001, 0.001, 0., 0.),
  (0., 0., 0.1, 0.,),
  (0., 0., 0.01, 0.,),
  (0., 0., 0.001, 0.,),
  (0.1, 0.1, 0., 0.01),
  (0.01, 0.01, 0., 0.001),
  (0.001, 0.001, 0., 0.0001)
]
num_seeds = 3

for lr in lrs:
  for maskreg_coeff, displreg_coeff, flowreg_coeff, forwbackwreg_coeff in regs:
    for _ in range(num_seeds):
      train_model(lr=lr, hidden_layer_widths=[64,64],flowreg_coeff=flowreg_coeff,
      maskreg_coeff=maskreg_coeff, displreg_coeff=displreg_coeff, forwbackwreg_coeff=forwbackwreg_coeff)
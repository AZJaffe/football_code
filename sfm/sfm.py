import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import imageio
import os

class PairConsecutiveFramesDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir):
    self.num_images = len(os.listdir(root_dir))
    self.root_dir = root_dir

  def __len__(self):
    return self.num_images - 1 # -1 since we load pairs
  
  def __getitem__(self, idx):
    im_1 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx}.png'), dtype=torch.float32)
    im_2 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx+1}.png'), dtype=torch.float32)
    return torch.cat((im_1, im_2), dim=-1).transpose(-1,0)

class SfMNet(torch.nn.Module):
  """ SfMNet is a motion detected based off a paper

  This module is desgined for inputs with shape (6,288,512)
  The 6 input channels come from two 3 channel images concatenated
  along the 3rd dimension 

  H and W must be divisible by 2**num_conv_encode
  """
  def __init__(self, H,W, K=1, C=8, num_conv_encode=2, fc_layer_width=512):
    super(SfMNet, self).__init__()
    self.factor = num_conv_encode
    self.H, self.W, self.K, self.C = H,W,K,C
    self.fc_layer_width = fc_layer_width

    # This will be useful in forward to do the spatial transform
    self.grid = torch.cartesian_prod(
      torch.linspace(-1,1,steps=self.H), \
      torch.linspace(-1,1,step=self.W)   \
    ).reshape((self.H,self.W,2))


    ####################
    #     Encoder      #
    ####################
    conv_encode = nn.ModuleList([nn.Conv2d(6, self.C, kernel_size=3, stride=1, padding=1)])
    bns_encode = nn.ModuleList([nn.BatchNorm2d(self.C)])
    # Out channels is at most 2 ** (factor + 5) == 256 for factor == 3
    for i in range(self.factor):
      in_channels = self.C * (2 ** i)
      out_channels = self.C * (2 ** (i + 1))
      conv_encode.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
      bns_encode.append(nn.BatchNorm2d(out_channels))
      conv_encode.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
      bns_encode.append(nn.BatchNorm2d(out_channels))
    self.conv_encode = conv_encode
    self.bns_encode = bns_encode

    ####################
    #     Decoder      #
    ####################
    conv_decode = nn.ModuleList([]) 
    bns_decode = nn.ModuleList([])
    for i in range(self.factor):
      in_channels = int(self.C * 2 ** (self.factor - i - 1) * 1.5)
      out_channels = self.C * 2 ** (self.factor - i - 1)
      conv_decode.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
      bns_decode.append(nn.BatchNorm2d(out_channels))

    self.conv_decode = conv_decode
    self.bns_decode = bns_decode
    self.final_conv = nn.Conv2d(self.C, K, kernel_size=3, stride=1, padding=1)
    self.final_bn = nn.BatchNorm2d(K)


    #####################
    #     FC Layers     #
    #####################
    embedding_dim = self.C * H * W // (2 ** self.factor)
    self.fc1 = nn.Linear(embedding_dim, self.fc_layer_width)
    self.fc2 = nn.Linear(self.fc_layer_width, self.fc_layer_width)
    self.fc3 = nn.Linear(self.fc_layer_width, 2*self.K) # Predict 2d displacement for spatial transform

  def forward(self, input):
    xs = input
    batch_size = self.shape[0]
    # Compute the embedding using the encoder convolutional layers
    encodings = []
    for i, (conv, bn) in enumerate(zip(self.conv_encode, self.bns_encode)):
      if i % 2 == 1:
        encodings.append(xs)
      xs = F.relu(bn(conv(xs)))

    embedding = torch.reshape(xs, (xs.shape[0], -1)) # Reshape to a flat vector
    assert(len(encodings) == self.factor)

    # Compute object masks using convolutional decoder layers
    for i, (conv, bn) in enumerate(zip(self.conv_decode, self.bns_decode)):
      xs = F.pixel_shuffle(xs, 2)
      xs = torch.cat((xs, encodings[-1-i]), dim=1) # Cat on channel dimension
      xs = F.relu(bn(conv(xs)))

    masks = torch.sigmoid(self.final_bn(self.final_conv(xs)))

    # Compute the displacements starting from the embedding using FC layers
    embedding = F.relu(self.fc1(embedding))
    embedding = F.relu(self.fc2(embedding))
    displacements = self.fc3(embedding)
    # Displacements has shape (batch_size, self.K * 2)
    # reshape in order to broadcast with masks
    displacements = displacements.reshape((batch_size, self.K, 1, 1, 2))
    flow = torch.sum(displacements * masks.unsqueeze(-1), dim=1)
    # flow has size (batch_size, H, W, 2)

    grid = self.grid.clone() - flow
    out = torch.nn.functional.grid_sample(input[0:3], grid)
    
    return out, masks, displacements

ds = PairConsecutiveFramesDataset('../../data/single_red_ball_512x288')
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

model = SfMNet()
optim = torch.optim.Adam(model.parameters())

input = ds[0].unsqueeze(0)
output = model(input)
print(output.shape)

torch.dtype.float32
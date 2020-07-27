import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class SfMNet(torch.nn.Module):
  """ SfMNet is a motion detected based off a paper

  The 6 input channels come from two 3 channel images concatenated
  along the 3rd dimension 

  H and W must be divisible by 2**conv_depth
  """
  def __init__(self, *, H, W, K=1, C=16, conv_depth=2, fc_layer_width=512):
    super(SfMNet, self).__init__()
    self.factor = conv_depth
    self.H, self.W, self.K, self.C = H,W,K,C
    self.fc_layer_width = fc_layer_width
    # 2d affine transform
    self.register_buffer('identity_affine_transform', \
      torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32))

    ####################
    #     Encoder      #
    ####################
    conv_encode = nn.ModuleList([nn.Conv2d(6, self.C, kernel_size=3, stride=1, padding=1, bias=False)])
    bns_encode = nn.ModuleList([nn.BatchNorm2d(self.C)])
    # Out channels is at most 2 ** (factor + 5) == 256 for factor == 3
    for i in range(self.factor):
      in_channels = self.C * (2 ** i)
      out_channels = self.C * (2 ** (i + 1))
      conv_encode.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
      bns_encode.append(nn.BatchNorm2d(out_channels))
      conv_encode.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
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
      conv_decode.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
      bns_decode.append(nn.BatchNorm2d(out_channels))

    self.conv_decode = conv_decode
    self.bns_decode = bns_decode
    self.final_conv = nn.Conv2d(self.C, K, kernel_size=3, stride=1, padding=1, bias=False)
    self.final_bn = nn.BatchNorm2d(K)


    #####################
    #     FC Layers     #
    #####################
    embedding_dim = self.C * H * W // (2 ** self.factor)
    self.fc1 = nn.Linear(embedding_dim, self.fc_layer_width)
    self.fc2 = nn.Linear(self.fc_layer_width, self.fc_layer_width)
    self.fc3 = nn.Linear(self.fc_layer_width, 2*self.K) # Predict 2d displacement for spatial transform

  def get_params(self):
    """ Get a dictionary describing the configuration of the model """
    return {
      'H': self.H,
      'W': self.W,
      'K': self.K,
      'C': self.C,
      'conv_depth': self.conv_depth,
      'fc_layer_width': self.fc_layer_width,
    }

  def total_params(self):
    sum(p.numel() for p in self.parameters())

  def forward(self, input):
    xs = input
    batch_size = input.shape[0]
    # Compute the embedding using the encoder convolutional layers
    encodings = []
    for i, (conv, bn) in enumerate(zip(self.conv_encode, self.bns_encode)):
      if i % 2 == 1:
        encodings.append(xs)
      xs = F.relu(bn(conv(xs)))

    embedding = torch.flatten(xs, start_dim=1)
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
    displacements = self.fc3(embedding).reshape((batch_size, self.K, 2))

    # Reshape displacements and masks so they can be broadcast
    flow = torch.sum(displacements.unsqueeze(-2).unsqueeze(-2) * masks.unsqueeze(-1), dim=1)
    # flow has size (batch_size, H, W, 2)

    # identity is not a function of any of the forward parameters
    identity = F.affine_grid( \
      # Need to batchify identitiy_affine_transform
      self.identity_affine_transform.unsqueeze(0).repeat(batch_size, 1, 1), \
      (batch_size, 3, self.H, self.W), \
      align_corners=False
    )
  
    grid = identity + flow
    out = F.grid_sample(input[:,0:3], grid, align_corners=False)
    
    return out, masks, flow, displacements

def l1_recon_loss(p,q):
  """ Computes the mean L1 reconstructions loss of a batch

  p - The first image in the sequence
  q - The second image in the sequence

  p,q should both have shape NxCxHxW
  """

  return torch.mean(torch.sum(torch.abs(p - q), dim=(1,2,3)))

def l1_flow_regularization(masks, displacements):
  """ Computes the mean L1 norm of the flow across the batch

  This is a bit different than flow returned by the model.
  The flow reutrned by the model is the sum of the constituent flows
  of each object. The flow calculated here is the L1 norm of 
  the constituent flows.

  masks         - shape NxCxHxW where C is the number of objects, NOT image channels!
  displacements - shape NxCx2
  """

  # After the unsqueezes, the shape is NxCxHxWx1 * NxCx1x1x2. The sum is taken across C,H,W,2 then meaned across N
  return torch.mean( \
    torch.sum(torch.abs(masks.unsqueeze(-1) * displacements.unsqueeze(-2).unsqueeze(-2)), dim=(1,2,3,4)),
  )

def visualize(input, output, mask, flow, spacing=3):
  """ imgs should be size (6,H,W) """
  H = input.shape[1]
  W = input.shape[2]
  fig, ax = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(W/3,H/3))
  first = input[0:3].permute(1,2,0)
  second = input[3:6].permute(1,2,0)
  output = output.permute(1,2,0)
  ax[0][0].imshow(first)
  ax[0][0].set_title('First image')
  
  ax[1][0].imshow(second)
  ax[1][0].set_title('Second image')

  ax[0][1].imshow(output)
  ax[0][1].set_title('Predicted Second Image')

  # This one will throw if the v.f. is identically 0
  ax[1][1].imshow(mask.squeeze(), cmap='Greens')
  xflow = flow[:,:,0]
  yflow = flow[:,:,1]
  i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
  mask = np.logical_or((i % spacing != 0), (j % spacing != 0))
  mx = np.ma.masked_array(xflow, mask=mask)
  my = np.ma.masked_array(yflow, mask=mask)
  ax[1][1].quiver(mx * W/2, my * H/2, scale=1, scale_units='xy', angles='xy', color='red') 
  ax[1][1].set_title('Mask and Flow')
  return fig
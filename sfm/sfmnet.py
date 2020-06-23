import torch
import torch.nn.functional as F
import torch.nn as nn

class SfMNet(torch.nn.Module):
  """ SfMNet is a motion detected based off a paper

  The 6 input channels come from two 3 channel images concatenated
  along the 3rd dimension 

  H and W must be divisible by 2**num_conv_encode
  """
  def __init__(self, *, H, W, K=1, C=8, num_conv_encode=2, fc_layer_width=512):
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
    torch.nn.init.zeros_(self.fc3.weight)
    torch.nn.init.zeros_(self.fc3.bias)

  def forward(self, input):
    xs = input
    batch_size = self.shape[0]
    # Compute the embedding using the encoder convolutional layers
    encodings = []
    for i, (conv, bn) in enumerate(zip(self.conv_encode, self.bns_encode)):
      if i % 2 == 1:
        encodings.append(xs)
      xs = F.relu(bn(conv(xs)))

    embedding = torch.flatten(xs)
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
    flow = torch.sum(displacements.reshape((batch_size, self.K, 1, 1, 2)) * masks.unsqueeze(-1), dim=1)
    # flow has size (batch_size, H, W, 2)

    grid = self.grid.clone() + flow
    out = torch.nn.functional.grid_sample(input[0:3], grid)
    
    return out, masks, displacements

def l1_recon_loss(p,q):
  # p,q should both have shape NxCxHxW
  return torch.mean(torch.abs(p - q))

def l1_flow_regularization(masks, displacements):
  """ Computes the mean L1 of the flow across the batch

  masks         - shape NxCxHxW where C is the number of objects
  displacements - shape NxCx2
  """

  # This doesn't seem like it is invariant to upsampling.
  # If the input is upsampled 2x, then the displacement should be 2x larger, the masks should cover 4x the area
  # Thus this loss would be bigger. Does this matter? Probably not I guess...
  return torch.mean( \
    torch.sum(torch.abs(masks.unsqueeze(-1) * displacements.unsqueeze(-2).unsqueeze(-2)), dim=(1,2,3,4)), \
    dim=0 \
  )

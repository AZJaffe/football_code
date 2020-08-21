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
  def __init__(self, *, H, W, im_channels=3, K=1, C=16, conv_depth=2, hidden_layer_widths=[32]):
    """ fc_layer_spec is the number of fully connected layers BEFORE the output layer """
    super(SfMNet, self).__init__()
    self.factor = conv_depth
    self.H, self.W, self.K, self.C = H,W,K,C
    self.im_channels = im_channels
    # 2d affine transform
    self.register_buffer('identity_affine_transform', \
      torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32))

    ####################
    #     Encoder      #
    ####################
    conv_encode = nn.ModuleList([nn.Conv2d(im_channels*2, self.C, kernel_size=3, stride=1, padding=1, bias=False)])
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


    #####################
    #     FC Layers     #
    #####################
    embedding_dim = self.C * H * W // (2 ** self.factor)
    fc_layer_widths = [embedding_dim, *hidden_layer_widths, 2*K]
    self.fc_layers = nn.ModuleList([ \
      nn.Linear(fc_layer_widths[i], fc_layer_widths[i+1], bias=False) \
      for i in range(0, len(fc_layer_widths) - 1) \
    ])

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

    masks = torch.sigmoid(self.final_conv(xs))

    # Compute the displacements starting from the embedding using FC layers
    for i,fc in enumerate(self.fc_layers):
      if i != len(self.fc_layers) - 1:
        embedding = F.relu(fc(embedding))
      else:
        embedding = fc(embedding)
    displacements = embedding.reshape((batch_size, self.K, 2))

    # Reshape displacements and masks so they can be broadcast
    flow = torch.sum(displacements.unsqueeze(-2).unsqueeze(-2) * masks.unsqueeze(-1), dim=1)
    # flow has size (batch_size, H, W, 2)

    # identity is not a function of any of the forward parameters
    identity = F.affine_grid( \
      # Need to batchify identitiy_affine_transform
      self.identity_affine_transform.unsqueeze(0).repeat(batch_size, 1, 1), \
      (batch_size, self.im_channels, self.H, self.W), \
      align_corners=False
    )
  
    grid = identity + flow
    out = F.grid_sample(input[:,0:self.im_channels], grid, align_corners=False, padding_mode="border")
    
    return out, masks, flow, displacements

def l1_recon_loss(p,q, reduction=torch.mean):
  """ Computes the mean L1 reconstructions loss of a batch

  p - The first image in the sequence
  q - The second image in the sequence

  p,q should both have shape NxCxHxW
  """

  loss = torch.mean(torch.sum(torch.abs(p - q), dim=(1,)), dim=(1,2))
  if reduction is not None:
    return reduction(loss)
  else:
    return loss

def l1_flow_regularization(masks, displacements):
  """ Computes the mean L1 norm of the flow across the batch

  This is a bit different than flow returned by the model.
  The flow reutrned by the model is the sum of the constituent flows
  of each object. The flow calculated here is the L1 norm of 
  the constituent flows.

  masks         - shape NxCxHxW where C is the number of objects, NOT image channels!
  displacements - shape NxCx2
  """

  # After the unsqueezes, the shape is NxCxHxWx1 for masks NxCx1x1x2 for displacements. The sum is taken across C,H,W,2 then meaned across N
  return torch.mean( \
    torch.sum(torch.abs(masks.unsqueeze(-1) * displacements.unsqueeze(-2).unsqueeze(-2)), dim=(1,4),)
  )

def l1_mask_regularization(mask):
  """ Computes the mean L1 of the masks

  mask - shape NxCxHxW where C is the number of objects, NOT image channels!
  """

  return torch.mean( \
    torch.sum(torch.abs(mask), dim=(1,))
  )

def l2_displacement_regularization(displacement):
  """ Computes the mean L1 of the masks

  displacement - shape NxCx2 where C is the number of objects, NOT image channels!
  """

  return torch.mean( \
    torch.sum(torch.square(displacement), dim=(1,2,))
  )

def visualize(model, im1, im2, spacing=None):
  """ im1 and im2 should be size BxCxHxW """
  with torch.no_grad():
    forwardbatch = torch.cat((im1, im2), dim=1)
    backwardbatch = torch.cat((im2, im1), dim=1)
    input = torch.cat((forwardbatch, backwardbatch), dim=0)
    output, mask, flow, displacement = model(input)
    output, mask, flow, displacement = output.cpu(), mask.cpu(), flow.cpu(), displacement.cpu()
    loss = l1_recon_loss(torch.cat((im2, im1), dim=0), output, reduction=None)
    K = mask.shape[1]

  im1 = im1.cpu()
  im2 = im2.cpu()

  B,C,H,W = im1.shape
  fig, ax = plt.subplots(figsize=(9, B*4), nrows=2*B, ncols=(2+K), squeeze=False,)

  if spacing is None:
    spacing = [W//20, H//20]
  i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
  vmask = np.logical_or((i % spacing[0] != 0), (j % spacing[1] != 0))
  for b in range(B):
    first = im1[b].permute(1,2,0)
    second = im2[b].permute(1,2,0)
    predfirst = output[b+B].permute(1,2,0)
    predsecond = output[b].permute(1,2,0)
    if C == 1:
      first = first.squeeze(2)
      second = second.squeeze(2)
      predfirst = predfirst.squeeze(2)
      predsecond = predsecond.squeeze(2)

    ######### First Row ###############
    ax[2*b][0].imshow(first, vmin=0., vmax=1.)
    ax[2*b][0].set_title('1st Input')

    mx = np.ma.masked_array(flow[b+B,:,:,0], mask=vmask)
    my = np.ma.masked_array(flow[b+B,:,:,1], mask=vmask)
    ax[2*b][1].imshow(predfirst, vmin=0., vmax=1.)
    ax[2*b][1].quiver(mx * W/2, my * H/2, scale=1, scale_units='xy', angles='xy', color='red') 
    ax[2*b][1].set_title(f'Pred 1st w/ flow\n(l={loss[b+B]:.4f})' , wrap=True)

    for k in range(K):
      ax[2*b][2+k].imshow(predfirst, vmin=0., vmax=1.)
      ax[2*b][2+k].imshow(np.ma.masked_less(mask[b+B,k], 0.5), alpha=0.4, vmin=0., vmax=1., cmap='Oranges')
      ax[2*b][2+k].set_title('Pred 1st w/ mask %d\nd=(%.2f, %.2f)' % (k, displacement[b+B,k,0], displacement[b+B,k,1]))

    ######### Second Row ###############
    ax[2*b+1][0].imshow(second, vmin=0., vmax=1.)
    ax[2*b+1][0].set_title('2nd Input')

    mx = np.ma.masked_array(flow[b,:,:,0], mask=vmask)
    my = np.ma.masked_array(flow[b,:,:,1], mask=vmask)
    ax[2*b+1][1].imshow(predsecond, vmin=0., vmax=1.)
    ax[2*b+1][1].quiver(mx * W/2, my * H/2, scale=1, scale_units='xy', angles='xy', color='red') 
    ax[2*b+1][1].set_title(f'Pred 2nd w/ flow\n(l={loss[b]:.4f})', wrap=True)

    for k in range(K):
      ax[2*b+1][2+k].imshow(predsecond, vmin=0., vmax=1.)
      ax[2*b+1][2+k].imshow(np.ma.masked_less(mask[b,k], 0.5), alpha=0.4, vmin=0., vmax=1., cmap='Oranges')
      ax[2*b+1][2+k].set_title('Pred 2nd w/ mask %d\nd=(%.2f, %.2f)' % (k, displacement[b,k,0], displacement[b,k,1]))
    
  fig.tight_layout()
  return fig

import kornia
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

class ConvEncoder(torch.nn.Module):
  def __init__(self, *, H, W, im_channels, C=16, conv_depth=2):
    super(ConvEncoder, self).__init__()
    self.H, self.W, self.C = H,W,C
    self.im_channels = im_channels
    self.conv_depth = conv_depth

    conv_encode = nn.ModuleList([nn.Conv2d(im_channels*2, self.C, kernel_size=3, stride=1, padding=1, bias=False)])
    bns_encode = nn.ModuleList([nn.BatchNorm2d(self.C)])
    for i in range(self.conv_depth):
      in_channels = self.C * (2 ** i)
      out_channels = self.C * (2 ** (i + 1))
      conv_encode.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False))
      bns_encode.append(nn.BatchNorm2d(out_channels))
      conv_encode.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
      bns_encode.append(nn.BatchNorm2d(out_channels))
    self.conv_encode = conv_encode
    self.bns_encode = bns_encode

  def forward(self, input):
    xs = input
    # Compute the embedding using the encoder convolutional layers
    encodings = []
    for i, (conv, bn) in enumerate(zip(self.conv_encode, self.bns_encode)):
      if i % 2 == 1:
        encodings.append(xs)
      xs = F.relu(bn(conv(xs)))

    return xs, encodings

class ConvDecoder(torch.nn.Module):
  def __init__(self, *, C=16, conv_depth=2, K=1):
    super(ConvDecoder, self).__init__()
    self.conv_depth = conv_depth
    self.C = C
    conv_decode = nn.ModuleList([]) 
    bns_decode = nn.ModuleList([])
    for i in range(self.conv_depth):
      in_channels = int(self.C * 2 ** (self.conv_depth - i - 1) * 1.5)
      out_channels = self.C * 2 ** (self.conv_depth - i - 1)
      conv_decode.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
      bns_decode.append(nn.BatchNorm2d(out_channels))

    self.conv_decode = conv_decode
    self.bns_decode = bns_decode
    self.final_conv = nn.Conv2d(self.C, K, kernel_size=3, stride=1, padding=1, bias=False)

  def forward(self, input, encodings, mask_logit_noise_var=0.):
    xs = input
    # Compute object masks using convolutional decoder layers
    for i, (conv, bn) in enumerate(zip(self.conv_decode, self.bns_decode)):
      xs = F.pixel_shuffle(xs, 2)
      xs = torch.cat((xs, encodings[-1-i]), dim=1) # Cat on channel dimension
      xs = F.relu(bn(conv(xs)))
    logits = self.final_conv(xs)
    noise = torch.tensor(np.random.normal(scale=mask_logit_noise_var, size=logits.shape), dtype=torch.float32).to(logits.device)
    masks = torch.sigmoid(logits + noise)
    return masks

class Flow2D(torch.nn.Module):
  def __init__(self, *, C, H, W, camera_translation, max_batch_size=16):
    super(Flow2D, self).__init__()
    self.camera_translation = camera_translation
    self.H, self.W = H, W
  
  def forward(self, *, im, displacements, masks):
    B, K, _, _ = masks.shape
    if self.camera_translation:
      if K > 0:
        flow = torch.sum(displacements[:,1:].unsqueeze(-2).unsqueeze(-2) * masks.unsqueeze(-1), dim=1)
      else:
        flow = torch.zeros((B,self.H,self.W,2), device=displacements.device)
      flow = flow + displacements[:,0].unsqueeze(-2).unsqueeze(-2)
    else:
      flow = torch.sum(displacements.unsqueeze(-2).unsqueeze(-2) * masks.unsqueeze(-1), dim=1)
    return flow

class SfMNet2D(torch.nn.Module):
  """ SfMNet is a motion detected based off a paper

  The 6 input channels come from two 3 channel images concatenated
  along the 3rd dimension 

  H and W must be divisible by 2**conv_depth
  """
  def __init__(self, *, H, W, im_channels=3, K=1, C=16, conv_depth=2, hidden_layer_widths=[32], camera_translation=False):
    """ fc_layer_spec is the number of fully connected layers BEFORE the output layer """
    super(SfMNet2D, self).__init__()
    self.conv_depth = conv_depth
    self.H, self.W, self.K, self.C = H,W,K,C
    self.im_channels = im_channels
    self.camera_translation = camera_translation # whether or not to include camera_translation
    # 2d affine transform
    self.register_buffer('identity_affine_transform', \
      torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32))

    self.encoder = ConvEncoder(H=H,W=W,im_channels=im_channels,C=C, conv_depth=conv_depth)
    self.decoder = ConvDecoder(C=C, conv_depth=conv_depth, K=K) if K is not 0 else None
    self.flow_module = Flow2D(H=H,C=C,W=W, camera_translation=camera_translation)

    #####################
    #     FC Layers     #
    #####################
    embedding_dim = (self.C * H * W) // (2 ** self.conv_depth)
    fc_layer_widths = [embedding_dim, *hidden_layer_widths, 2*(K+camera_translation)]
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
    return sum(p.numel() for p in self.parameters())

  def forward(self, input, mask_logit_noise_var=0.):
    batch_size = input.shape[0]

    embedding, encodings = self.encoder(input)
    if self.decoder is not None:
      masks = self.decoder(embedding, encodings, mask_logit_noise_var)
    else:
      masks = torch.ones((batch_size, 0, self.H, self.W), device=input.device)

    xs = torch.flatten(embedding, start_dim=1)

    # Compute the displacements starting from the embedding using FC layers
    for i,fc in enumerate(self.fc_layers):
      if i != len(self.fc_layers) - 1:
        xs = F.relu(fc(xs))
      else:
        xs = fc(xs)
    displacements = xs.reshape((batch_size, self.K + self.camera_translation, 2))

    flow = self.flow_module(im=input[:,0:self.im_channels], displacements=displacements, masks=masks)
    
    return masks, flow, displacements
  

class LossModule(torch.nn.Module):

  def __init__(self, sfm_model, dssim_coeff=1., l1_photometric_coeff=0., l1_flow_reg_coeff=0., max_batch_size=16):
    super(LossModule, self).__init__()
    self.sfm_model = sfm_model
    self.C, self.H, self.W = sfm_model.im_channels, sfm_model.H, sfm_model.W
    self.dssim_coeff = dssim_coeff
    self.l1_photometric_coeff = l1_photometric_coeff
    self.l1_flow_reg_coeff = l1_flow_reg_coeff
    identity_affine = torch.tensor([[1,0,0],[0,1,0]], dtype=torch.float32)
    batched_identity = F.affine_grid( \
      # Need to batchify identitiy_affine_transform
      identity_affine.unsqueeze(0).repeat(max_batch_size, 1, 1), \
      (max_batch_size, self.C, self.H, self.W), \
      align_corners=False
    )
    self.register_buffer('batched_identity', batched_identity)

  def forward(self, im1, im2, mask_logit_noise_var=0., reduction=torch.mean):
    """ Returns the loss for the model 
      im1: Batch of first images in the temporal sequence
      im2: Batch of second images in the temporal sequence
      mask_logit_noise_var: Passed onto the model
      reduction: either 'mean' or None. 'mean' will return a scalar of the mean loss over the batch,
      while None will return the loss for each element of the batch.
    """
    B = im1.shape[0]
    inp = torch.cat((im1, im2), dim=1)
    mask, flow, displacement = self.sfm_model(inp, mask_logit_noise_var)

    grid = self.batched_identity[0:B] + flow
    im2_estimate = F.grid_sample(im1, grid, align_corners=False, padding_mode="zeros")

    dssim = dssim_loss(im2_estimate, im2, reduction=reduction) if self.dssim_coeff is not 0. else 0.
    l1_photometric = l1_photometric_loss(im2_estimate, im2, reduction=reduction) if l1_photometric_loss is not 0. else 0.
    flow_reg_loss = l1_flow_regularization(mask, displacement, reduction=reduction) if self.l1_flow_reg_coeff is not 0. else 0.

    photometric_loss = self.dssim_coeff * dssim + self.l1_photometric_coeff * l1_photometric
    total_loss =  self.l1_flow_reg_coeff * flow_reg_loss + photometric_loss
    return total_loss, photometric_loss, im2_estimate, mask, flow, displacement


class ForwBackwLoss(torch.nn.Module):
  def __init__(self, loss_module, forwbackw_coeff):
    super(ForwBackwLoss, self).__init__()
    self.loss_module = loss_module
    self.forwbackw_coeff = forwbackw_coeff

  def forward(self, im1, im2, mask_logit_noise_var=0.0, reduction=torch.mean):

    N = im1.shape[0]

    forw = torch.cat((im1, im2), dim=0)
    backw = torch.cat((im2, im1), dim=0)
    total_loss, photometric_loss, out, mask, flow, displacement = self.loss_module(forw, backw, mask_logit_noise_var, reduction=reduction)

    forwbackw_loss = forwbackw_displacement_loss(displacement[0:N],displacement[N:2*N], reduction=reduction)
    total_loss += self.forwbackw_coeff * forwbackw_loss
    # Note that out, mask, flow, displacement will have batch size 2*N !
    return total_loss, photometric_loss, out, mask, flow, displacement

def forwbackw_displacement_loss(forwdispl, backwdispl, reduction=torch.mean):
  loss = torch.sum(torch.abs(forwdispl - backwdispl), dim=(1,2))
  if reduction is not None:
    return reduction(loss)
  else:
    return loss

def l1_photometric_loss(p,q, reduction=torch.mean):
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

def dssim_loss(p,q, reduction=torch.mean):
  loss = torch.mean(kornia.losses.ssim(p, q, 11), dim=(1,2,3))
  if reduction is not None:
    return reduction(loss)
  else:
    return loss

def l1_flow_regularization(masks, displacements, reduction=None):
  """ Computes the mean L1 norm of the flow across the batch

  This is a bit different than flow returned by the model.
  The flow reutrned by the model is the sum of the constituent flows
  of each object. The flow calculated here is the L1 norm of 
  the constituent flows.

  masks         - shape NxCxHxW where C is the number of objects, NOT image channels!
  displacements - shape Nx(C+1)x2 or NxCx2 - The extra dim is for the camera (possibly)
  """

  # If there is camera translation, drop it and don't include it in the regularization
  N, C, H, W = masks.shape
  if displacements.shape[1] != C:
    #displacements = displacements[:,1:]
    masks = torch.cat((torch.ones(N,1,H,W, device=masks.device), masks), dim=1)
  # After the unsqueezes, the shape is NxCxHxWx1 for masks NxCx1x1x2 for displacements. The sum is taken across C,2 then meaned across H,W
  loss = torch.mean(torch.sum(torch.abs(masks.unsqueeze(-1) * displacements.unsqueeze(-2).unsqueeze(-2)), dim=(1,4)), dim=(1,2))
  if reduction is not None:
    return reduction(loss)
  else:
    return loss

def l1_mask_regularization(mask):
  """ Computes the mean L1 of the masks

  mask - shape NxCxHxW where C is the number of objects, NOT image channels!
  """
  if mask.shape[1] == 0:
    return 0

  return torch.mean( \
    torch.sum(mask, dim=(1,))
  )

def mask_variance_regularization(mask):
  if mask.shape[1] == 0:
    return 0

  return torch.mean( \
    torch.sum(mask * (1 - mask), dim=(1,))  
  )

def l2_displacement_regularization(displacement):
  """ Computes the mean L1 of the masks

  displacement - shape NxCx2 where C is the number of objects, NOT image channels!
  """
  # TODO doesn't account for camera translation
  return torch.mean( \
    torch.sum(torch.square(displacement), dim=(1,2,))
  )

def visualize(model, im1, im2):
  # TODO Figure out what to do with camera translation
  # TODO only show one row, not both forw and backw
  """ im1 and im2 should be size BxCxHxW """
  def rgb2gray(rgb):
    if len(rgb.shape) == 2 or rgb.shape[2] == 1:
      # In this case, rgb is actually only single channel not rgb.
      return rgb
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

  with torch.no_grad():
    total_loss, photometric_loss, output, mask, flow, displacement = model(im1, im2, reduction=None)
    output, mask, flow, displacement = output.cpu(), mask.cpu(), flow.cpu(), displacement.cpu()
    K = mask.shape[1]

  im1_cpu = im1.cpu()
  im2_cpu = im2.cpu()

  B,C,H,W = im1_cpu.shape

  fig, ax = plt.subplots(figsize=(9, B*4), nrows=2*B, ncols=(2+K), squeeze=False,)

  for b in range(B):
    second = im2_cpu[b].permute(1,2,0)
    predsecond = output[b].permute(1,2,0)
    if C == 1:
      second = second.squeeze(2)
      predsecond = predsecond.squeeze(2)

    ax[2*b][0].imshow(second, interpolation='none', vmin=0., vmax=1.)
    ax[2*b][0].set_title('2nd Input Image')


    ax[2*b][1].imshow(vis_flow(flow[b]), interpolation='none', vmin=0., vmax=1.)
    ax[2*b][1].set_title(f'F_21\n(photo_loss  ={photometric_loss[b]:.8f})', wrap=True)

    for k in range(K):
      ax[2*b][2+k].imshow(rgb2gray(predsecond), interpolation='none', cmap='gray', vmin=0., vmax=1.)
      ax[2*b][2+k].imshow(mask[b,k], interpolation='none', alpha=0.9, vmin=0., vmax=1., cmap='Reds')
      ax[2*b][2+k].set_title('Recon 2nd w/ mask %d\nd=(%.2f, %.2f)\nmass=%.2f' % (k, displacement[b,k,0], displacement[b,k,1], torch.sum(mask[b,k])))

  fig.tight_layout()
  return fig

def normalize(a):
    max = np.max(a)
    min = np.min(a)
    if max == min:
      return np.ones_like(a)
    return (a - min) / (max - min)

def vis_flow(flow):
  ### flow should be HxWx2
    angle = (np.arctan2(flow[...,1], -flow[...,0])) + 3 * math.pi / 2  # The sum rotates the colors
    angle = angle % (2 * math.pi) # Make sure that the values are in the range [0, 2pi]
    angle = angle / 2 / math.pi # Normalize to [0,1]
    mag = normalize(np.linalg.norm(flow, axis=2))
    ones = np.ones_like(angle)
    hsv = np.stack((angle, mag, ones), axis=2)
    return matplotlib.colors.hsv_to_rgb(hsv)

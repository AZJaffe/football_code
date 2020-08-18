import torch
import imageio
import os

class PairConsecutiveFramesDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, sliding=True, load_all=True, device=torch.device('cpu')):
    """ Initializes the dataset

    If load_all is true, load all the images from root_dir into memory and then send them to device
    """
    self.device = device
    self.num_images = len(os.listdir(root_dir))
    self.sliding = sliding
    if sliding is False and self.num_images % 2 != 0:
      raise ValueError('Expected an even amount of images, received', self.num_images)
    self.root_dir = root_dir

    if load_all == True:
      # Assume all images are the same size
      # Get a sample image for the size. imageio returns a tensor with shape HxWxC, and torch uses CxHxW
      im_0 = torch.tensor(imageio.imread(f'{self.root_dir}/image0.png'), dtype=torch.float32)
      if len(im_0.shape) == 2:
        im_0 = im_0.unsqueeze(2)
      images = torch.empty((self.num_images, im_0.shape[2], im_0.shape[0], im_0.shape[1]), dtype=torch.float32)
      for i in range(self.num_images):
        im = torch.tensor(imageio.imread(f'{self.root_dir}/image{i}.png'), dtype=torch.float32)
        if len(im.shape) == 2:
          im = im.unsqueeze(2)
        images[i] = im.permute(2, 0, 1) / 255
      self.images = images.to(device)

  def __len__(self):
    if self.sliding is True:
      return self.num_images - 1 # -1 since we load pairs
    else:
      return self.num_images // 2
  
  def __getitem__(self, idx):
    if isinstance(idx, slice):
      im1 = torch.cat([self[i][0].unsqueeze(0) for i in range(*idx.indices(len(self)))])
      im2 = torch.cat([self[i][1].unsqueeze(0) for i in range(*idx.indices(len(self)))])
      return (im1, im2)
    elif isinstance(idx, list):
      im1 = torch.cat([self[i][0].unsqueeze(0) for i in idx])
      im2 = torch.cat([self[i][1].unsqueeze(0) for i in idx])
      return (im1, im2)
    elif isinstance(idx, int):
      if idx >= len(self):
        raise IndexError
      if self.sliding is True:
        idx1 = idx
        idx2 = idx+1
      else:
        idx1 = idx*2
        idx2 = idx*2+1
      if self.images is None:
        im_1 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx1}.png'), dtype=torch.float32)
        im_2 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx2}.png'), dtype=torch.float32)
        if len(im_1.shape) == 2:
          im_1 = im_1.unsqueeze(2)
          im_2 = im_2.unsqueeze(2)
        im_1 = im_1.permute(2, 0, 1) / 255
        im_2 = im_2.permute(2, 0, 1) / 255
        return (im_1.to(self.device), im_2.to(self.device))
      else:
        return (self.images[idx1], self.images[idx2])
    else:
      raise TypeError("Invalid index operation")
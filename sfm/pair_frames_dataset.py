import torch
import imageio
import os
import numpy as np
import glob
import math

class PairConsecutiveFramesDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir):
    """ Initializes the dataset

    If load_all is true, load all the images from root_dir into memory and then send them to device
    """
    self.num_images = len(glob.glob(os.path.join(root_dir ,'image*.png')))
    self.digits = math.floor(math.log10(self.num_images - 1) + 1)
    self.root_dir = root_dir
    try:
      episodes = np.genfromtxt(os.path.join(self.root_dir, 'metadata.csv'), delimiter=',', skip_header=1, dtype=np.int)[:,1]
      self.index = []
      for i, (ep1, ep2) in enumerate(zip(episodes[:-1], episodes[1:])):
        if ep1 == ep2:
          self.index.append(i)
    except OSError:
      self.index = [i for i in range(self.num_images - 1)]

  def load_image(self, i):
    try:
      return torch.tensor(imageio.imread(os.path.join(self.root_dir ,f'image{str(i).zfill(self.digits)}.png')), dtype=torch.float32)
    except FileNotFoundError:
      return torch.tensor(imageio.imread(os.path.join(self.root_dir ,f'image{i}.png')), dtype=torch.float32)

  def __len__(self):
    return len(self.index)
  
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
      i = self.index[idx]
      im_1 = self.load_image(i)
      im_2 = self.load_image(i+1)
      if len(im_1.shape) == 2:
        im_1 = im_1.unsqueeze(2)
        im_2 = im_2.unsqueeze(2)
      im_1 = im_1.permute(2, 0, 1) / 255
      im_2 = im_2.permute(2, 0, 1) / 255
      return (im_1, im_2)
    else:
      raise TypeError("Invalid index operation")
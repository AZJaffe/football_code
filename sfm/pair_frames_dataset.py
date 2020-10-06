import torch
import imageio
import os
import numpy as np
import glob
import math
import pandas as pd

class PairConsecutiveFramesDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir):
    """ Initializes the dataset

    If load_all is true, load all the images from root_dir into memory and then send them to device
    """
    image_filenames = glob.glob(os.path.join(root_dir ,'image*.png')) 
    self.num_images = len(image_filenames)
    self.digits = len(os.path.basename(image_filenames[0])) - 9 # len('image.png') = 9
    self.root_dir = root_dir
    try:
      # metadata.csv contains the map between image number and which episode
      episodes = np.genfromtxt(os.path.join(self.root_dir, 'metadata.csv'), delimiter=',', skip_header=1, dtype=np.int)[:,1]
      self.index = []
      for i, (ep1, ep2) in enumerate(zip(episodes[:-1], episodes[1:])):
        if ep1 == ep2:
          self.index.append(i)
    except OSError:
      self.index = [i for i in range(self.num_images - 1)]

    try:
      # Try loading labels
      df = pd.read_csv(os.path.join(self.root_dir, 'labels.csv'))
      self.labels = {}
      if 'camera_translation_x' in df and 'camera_translation_y' in df:
        camera_translation = torch.tensor(df[['camera_translation_x', 'camera_translation_y']].to_numpy(), dtype=torch.float32)
        self.labels['camera_translation'] = camera_translation

    except OSError:
      self.labels = {}

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
      labels = {}
      for k,v in self.labels.items():
        labels[k] = v[idx]
      return (im1, im2, labels)
    elif isinstance(idx, list):
      im1 = torch.cat([self[i][0].unsqueeze(0) for i in idx])
      im2 = torch.cat([self[i][1].unsqueeze(0) for i in idx])
      labels = {}
      for k,v in self.labels.items():
        labels[k] = torch.cat([v[i].unsqueeze(0) for i in idx])
      return (im1, im2, labels)
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

      labels = {}
      for k,v in self.labels.items():
        labels[k] = v[idx]
      return (im_1, im_2, labels)
    else:
      raise TypeError("Invalid index operation")

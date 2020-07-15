import torch
import imageio
import os

class PairConsecutiveFramesDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, load_all=True, device=torch.device('cpu')):
    """ Initializes the dataset

    If load_all is true, load all the images from root_dir into memory and then send them to device
    """
    self.device = device
    self.num_images = len(os.listdir(root_dir))
    self.root_dir = root_dir

    if load_all == True:
      # Assume all images are the same size
      # Get a sample image for the size. imageio returns a tensor with shape HxWxC, and torch uses CxHxW
      im_0 = torch.tensor(imageio.imread(f'{self.root_dir}/image0.png'), dtype=torch.float32)
      self.images = torch.zeros((self.num_images, im_0.shape[2], im_0.shape[0], im_0.shape[1]), dtype=torch.float32)
      for i in range(self.num_images):
        self.images[i] = torch.tensor(imageio.imread(f'{self.root_dir}/image{i}.png'), dtype=torch.float32).permute(2, 0, 1) / 255
      self.images.to(device)

  def __len__(self):
    return self.num_images - 1 # -1 since we load pairs
  
  def __getitem__(self, idx):
    if isinstance(idx, slice):
      return torch.cat([self[i].unsqueeze(0) for i in range(*idx.indices(len(self)))])
    elif isinstance(idx, int):
      # imageio reads as WxH
      if self.images is None:
        im_1 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx}.png'), dtype=torch.float32).permute(2, 0, 1) / 255
        im_2 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx+1}.png'), dtype=torch.float32).permute(2, 0, 1) / 255
        return torch.cat((im_1, im_2), dim=0).to(self.device)
      else:
        return torch.cat((self.images[idx], self.images[idx+1]))
    else:

      raise TypeError("Invalid index operation")
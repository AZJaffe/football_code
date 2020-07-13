import torch
import imageio
import os

class PairConsecutiveFramesDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir):
    self.num_images = len(os.listdir(root_dir))
    self.root_dir = root_dir

  def __len__(self):
    return self.num_images - 1 # -1 since we load pairs
  
  def __getitem__(self, idx):
    if isinstance(idx, slice):
      return torch.cat([self[i].unsqueeze(0) for i in range(*idx.indices(len(self)))])
    elif isinstance(idx, int):
      # imageio reads as WxH
      im_1 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx}.png'), dtype=torch.float32)
      im_2 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx+1}.png'), dtype=torch.float32)
      return torch.cat((im_1 / 255, im_2 / 255), dim=-1).permute(2, 0, 1)
    else:
      raise TypeError("Invalid index operation")
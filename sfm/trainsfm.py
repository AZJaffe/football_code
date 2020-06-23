import torch
import imageio
import os
import argparse

import sfmnet

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


def train(*, data_dir, log_dir):
  ds = PairConsecutiveFramesDataset(data_dir)
  dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)


  input_shape = ds[0].shape

  model = sfmnet.SfMNet(H=input_shape[1], W=input_shape[2], K=1, fc_layer_width=128)
  print(model)
  optim = torch.optim.Adam(model.parameters())

  
  output = model(input)
  print(output.shape)



if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Train an sfm')
  parser.add_argument('data_dir',
                      help='the directory containing sequential images to use as data')
  parser.add_argument('log_dir',
                      help='the directory to store logs')
  args = parser.parse_args()
  train(data_dir=args.data_dir, log_dir=args.log_dir)

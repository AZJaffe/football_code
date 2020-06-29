import argparse
import os
import imageio
import torch
import sfmnet
import matplotlib.pyplot as plt

class PairConsecutiveFramesDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir):
    self.num_images = len(os.listdir(root_dir))
    self.root_dir = root_dir

  def __len__(self):
    return self.num_images - 1 # -1 since we load pairs
  
  def __getitem__(self, idx):
    #imageio reads as WxH
    im_1 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx}.png'), dtype=torch.float32)
    im_2 = torch.tensor(imageio.imread(f'{self.root_dir}/image{idx+1}.png'), dtype=torch.float32)
    return torch.cat((im_1 / 255, im_2 / 255), dim=-1).permute(2, 0, 1)

def train(*, data_dir, num_epochs=1, batch_size=8, lambda_flow_reg=0.2):
  ds = PairConsecutiveFramesDataset(data_dir)
  dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
  model = sfmnet.SfMNet(H=input_shape[1], W=input_shape[2], K=1, fc_layer_width=128)
  optimizer = torch.optim.Adam(model.parameters())

  for i in range(num_epochs):
    for batch in dl:
      optimizer.zero_grad()
      output, masks, flow, displacements = model(batch)
      loss = sfmnet.l1_recon_loss(batch[:,0:3], batch[:,3:6]) + \
        lambda_flow_reg * sfmnet.l1_flow_regularization(masks, displacements)
      loss.backward()
      optimizer.step()


def show_results(input, output, masks, flow):
  """ imgs should be size (6,H,W) """
  fig, ax = plt.subplots(nrows=3, ncols=2, squeeze=False)
  first = input[0:3].permute(1,2,0)
  second = input[3:6].permute(1,2,0)
  output = output.permute(1,2,0)
  ax[0][0].imshow(first)
  ax[0][0].set_title('First image')
  
  ax[1][0].imshow(second)
  ax[1][0].set_title('Second image')

  ax[2][0].imshow(output)
  ax[2][0].set_title('Predicted Second Image')

  ax[0][1].imshow(masks[0])
  ax[0][1].set_title('First Mask')

  # This one will throw if the v.f. is identically 0
  ax[1][1].quiver(flow[:,:,0], flow[:,:,1]) 
  ax[1][1].set_title('Flow')

  plt.show()

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Train an sfm')
  parser.add_argument('--data_dir',
                      help='the directory containing sequential images to use as data')
  # parser.add_argument('log_dir',
  #                     help='the directory to store logs')
  args = parser.parse_args()
  train(data_dir=args.data_dir)

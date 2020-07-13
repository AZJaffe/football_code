import argparse
import torch
import matplotlib.pyplot as plt

import sfmnet
from pair_frames_dataset import PairConsecutiveFramesDataset

if __name__=='__main__':
  # Parse args
  parser = argparse.ArgumentParser(description='Visualize a sfmnets prediction')
  parser.add_argument('--data_dir', required=True,
                      help='the directory containing sequential images to use as data')
  parser.add_argument('--model', required=True,
                      help='the path to load the sfm model')                    
  parser.add_argument('--index', default=0, 
                      help='the index of data to visualize')
  args = parser.parse_args()

  ds = PairConsecutiveFramesDataset(args.data_dir)
  # Construct model
  input = ds[args.index]
  input_shape = input.shape
  model = sfmnet.SfMNet(H=input_shape[1], W=input_shape[2], K=1, fc_layer_width=128)
  checkpoint = torch.load(args.model)
  model.load_state_dict(checkpoint['model_state_dict'])

  with torch.no_grad():
    output, masks, flow, displacements = model(input.unsqueeze(0))
    print(displacements[0])
    fig = sfmnet.visualize(input, output[0], masks[0], flow[0])
    plt.show()
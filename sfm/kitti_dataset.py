import torch
import imageio
import PIL
import os
import numpy as np
import glob
import math
import pandas as pd
from skimage import transform,io

def im_to_tensor(im):
    t = torch.tensor(im, dtype=torch.float32)
    if len(t.shape) == 2:
        t = t.unsqueeze(2)
    return t.permute(2,0,1) / 255.

H, W = 352, 1152 # Needs to be divisible by 32

class KittiRawStereoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, rgb=True):
        self.data_dir = data_dir
        if rgb:
            self.left_image_dir = os.path.join(data_dir, 'image_02', 'data')
            self.right_image_dir = os.path.join(data_dir, 'image_03', 'data')
        else:
            self.left_image_dir = os.path.join(data_dir, 'image_00', 'data')
            self.right_image_dir = os.path.join(data_dir, 'image_01', 'data')
        self.len = len(glob.glob(os.path.join(self.left_image_dir, '*.png')))

    def get_im_filename(self, idx):
        return str(idx).zfill(10) + '.png'

    def resize(self, im):
        return transform.resize(im, (H, W))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        if isinstance(i, slice):
            a = [self[j] for j in range(*i.indices(len(self)))]
            return torch.stack([l for l,r in a],dim=0), torch.stack([r for l,r in a],dim=0), {}

        left_im = io.imread(os.path.join(self.left_image_dir, self.get_im_filename(i)))
        right_im = io.imread(os.path.join(self.right_image_dir, self.get_im_filename(i)))
        return im_to_tensor(self.resize(left_im)), im_to_tensor(self.resize(right_im)), {}

# todo, make slicing work by wrapping ConcatDataset in a class that implements __getitem__
def CollectionKittiRawStereoDataset(dir_file, rgb=True):
    dirs = [ f.rstrip('\n') for f in open(dir_file)]
    ds = torch.utils.data.ConcatDataset([ KittiRawStereoDataset(d, rgb) for d in dirs])
    if len(ds) == 0:
        raise BaseException('No Kitti data found')
    return ds
    
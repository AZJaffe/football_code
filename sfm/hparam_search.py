import trainsfm
import sfmnet
import pair_frames_dataset
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio 
import datetime

im_channels=1
H=4
W=6
data=torch.ones((8,2*im_channels,H,W))
data[0,0,1,1] = 0
data[0,1,0,1] = 0
data[1,0,2,4] = 0
data[1,1,1,5] = 0
data[2,0,2,3] = 0
data[2,1,2,4] = 0
data[3,0,1,1] = 0
data[3,1,2,2] = 0
data[4,0,1,1] = 0
data[4,1,2,1] = 0
data[5,0,2,2] = 0
data[5,1,3,1] = 0
data[6,0,1,4] = 0
data[6,1,1,3] = 0
data[7,0,2,4] = 0
data[7,1,1,3] = 0
dl = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)

def train_model(lr=0.005, hidden_layer_widths=[16], num_epochs=1, maskreg=0., displreg=0.):
    model = sfmnet.SfMNet(
        H=H,
        W=W,
        im_channels=im_channels,
        K=1,
        C=4,
        conv_depth=1,
        hidden_layer_widths=hidden_layer_widths,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    time = datetime.datetime.now().strftime('%H:%M:%S')
    trainsfm.train_loop(
        model=model,
        vis_point=None,
        dl=dl,
        optimizer=optimizer,
        num_epochs=num_epochs,
        vis_freq=None,
        tensorboard_dir=f'./tb/08-09/{time}',
        maskreg_coeff=maskreg,
        displreg_coeff=displreg
    )
    return model

def eval_model(model):
    with torch.no_grad():
        output, mask, flow, displacement = model(data)
        loss = torch.mean(torch.abs(output - data[:,1:]), dim=(1,2,3))
        return loss
        

def vis_model(model):
    with torch.no_grad():
        output, mask, flow, displacement = model(data)
        loss = torch.mean(torch.abs(output - data[:,1:]), dim=(1,2,3))
        for l in loss:
            print(("Good " if l < 1/24 else "Bad ") + str(l))

        sfmnet.visualize(data, output, mask, flow, displacement,spacing=1)


lrs = [0.005, 0.01, 0.2, 0.0025]
hidden_layer_widths = [[32], [32,32]]
regs=[(0.1, 0.1), (0.01, 0.01), (0.001,0.001)]

hparams = []
for l in lrs:
    for h in hidden_layer_widths:
        for m,d in regs:
            hparams.append({
                'lr': l,
                'hidden_layer_widths': h,
                'maskreg': m,
                'displreg': d
            })
len(hparams)

num_runs = 5
result = [None] * len(hparams)
for k,hparam in enumerate(hparams):
    ms = [train_model(num_epochs=300, **hparam) for i in range(num_runs)]
    losses = np.array([eval_model(m).numpy() for m in ms])
    good = losses < 1/24
    mediangood = np.median(np.sum(good, axis=1))
    maxgood = np.max(np.sum(good, axis=1))
    mingood = np.max(np.sum(good, axis=1))
    result[k] = {}
    result[k]['loss'] = losses.tolist()
    result[k]['mean_loss'] = np.mean(losses)
    result[k]['good'] = good
    result[k]['mediangood'] = np.median(np.sum(good, axis=1))
for r in result:
    print(r['mediangood'])

import pickle
with open('./data.p', 'wb') as f:
    pickle.dump(result, f)
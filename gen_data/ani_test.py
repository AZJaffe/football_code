import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

save_path='data/football_clips/480x270_samples'

data = np.load('data/football_clips/480x270_colour.npy')
data = np.array(data, dtype='int')
fig = plt.figure()

for k in range(data.shape[0]):
  ims = []
  for j in range(4):
    ims.append([plt.imshow(data[k,:,:,3*j:3*(j+1)], animated=True)])

  ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                  repeat_delay=2000)
  ani.save(f'{save_path}_{k}.mp4')
  fig.clear()

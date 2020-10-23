import gym
import numpy as np
import imageio
import math
import os
import fire
import skimage
import skimage.transform
import skimage.color

def run(*, out_dir, env_name=None, env=None, num_images=100, H=None, W=None, gray=False, crop_top=None):
  if env_name is not None:
    env = gym.make(env_name)
  if env is None:
    raise 'Either env or env_name must be passed'
  digits = math.floor(math.log10(num_images - 1) + 1)
  e = 0
  os.makedirs(out_dir)
  index = np.empty((num_images), dtype=np.int)
  done = True
  e = -1
  for k in range(num_images):
    if k % 1000 == 0:
      print('step', k)
    if done:
      state = env.reset()
      done = False
      e += 1
    else:
      state, _, done, _ = env.step(env.action_space.sample())
    if crop_top is not None:
      # 50 for Breakout
      # 25 for Pong
      state = state[crop_top:]
    if gray is True:
      state = skimage.color.rgb2gray(state)
    if H is not None and W is not None:
      state = skimage.transform.resize(state, (H,W), order=0) # order=0 is nearest neighbour interpolation
    state = skimage.img_as_ubyte(state)
    imageio.imwrite(out_dir+'/image%s.png'%str(k).zfill(digits), state)
    index[k] = e
  with open(os.path.join(out_dir, 'metadata.csv'), 'w') as f:
    f.write('image_num,episode\n')
    for k,e in enumerate(index):
      f.write(','.join([str(k),str(e)]))
      f.write('\n')

if __name__ == '__main__':
  fire.Fire(run)
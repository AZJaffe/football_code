import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import imageio
import fire

class Frame:
  def __init__(self, shape):
    self.frame = 255 * np.ones((shape[1], shape[0]), dtype='uint8')

  def draw_circle(self, centre, radius, colour=0., gradient=0.2):
    """Draws a circle on the frame.

    Parameters:
    frame -- the frame to draw the circle on
    centre -- 2 value array which is the centre of the circle
    radius -- L1 radius of the circle 
    color -- the color of the circle
    """

    min_x = max(0, centre[0] - radius)
    max_x = min(self.frame.shape[0], centre[0] + radius)
    min_y = max(0, centre[1] - radius)
    max_y = min(self.frame.shape[1], centre[1] + radius)
    # Search the square with side lengths radius centred at centre.
    for i in range(min_x, max_x):
      for j in range(min_y, max_y):
        d = np.abs(i - centre[0]) + np.abs(j - centre[1])
        if d < radius:
          self.frame[i,j] = colour * (1 - d * gradient)
    return self.frame

def display_frames(frames):
  im = plt.imshow(frames[0])
  for i in range(frames.shape[0]):
    im.set_data(frames[i])
    plt.pause(0.02)
  plt.show()

# size should be np array of size (2,)
def generate_random_frames(out_dir, H=24, W=40, radius=3, speed=5, total_frames=1000,):
  size = np.array([W, H])
  try:
    os.makedirs(out_dir)
  except FileExistsError:
    s = input(f'The directory {out_dir} exists, do you wish to delete it? [y/n]:')
    if s == 'y':
      # TODO
      exit(1)
    else:
      exit(0)
  x = np.arange(speed,0,-1)
  y = np.arange(0,speed,1)
  top_right = np.column_stack((x,y))
  rot_90_cw = np.array([
    [0,-1],
    [1, 0]
  ]).T
  ds = np.concatenate((
    top_right, 
    top_right @ rot_90_cw, 
    top_right @ rot_90_cw @ rot_90_cw,
    top_right @ rot_90_cw @ rot_90_cw @ rot_90_cw,
  ))
  for i in range(total_frames):
    centre = np.random.randint(radius - 1 + speed, size - radius - speed + 1)
    d = ds[np.random.randint(0, ds.shape[0])]
    centre_new = centre + d
    if i % 10 == 0:
      print(f'i={i}')
    frame = Frame(size).draw_circle(centre, radius)
    frame2 = Frame(size).draw_circle(centre_new, radius)
    imageio.imwrite(out_dir+f"/image{2*i}.png", frame)
    imageio.imwrite(out_dir+f"/image{2*i+1}.png", frame2)


if __name__ == '__main__':
  fire.Fire(generate_random_frames)

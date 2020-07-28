import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import imageio

class Frame:
  def __init__(self, shape):
    self.frame = 255 * np.ones(tuple(shape) + (3,), dtype='uint8')

  def draw_circle(self, centre, radius, colour=np.array([255.,0.,0.]), gradient=0.2):
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
def generate_random_frames(size, out_dir, radius=3, total_frames=1000,):
  os.makedirs(out_dir)
  speed = 5
  top_right = np.array([
    [5,0], [4,1], [3,2], [2,3], [1,4],
  ])
  # top_right = np.array([
  #   [5,0], [4,1], [3,2], [2,3], [1,4],
  #   [0,5], [-1,4], [-2, 3], [-3,2], [-4,1],
  #   [-5,0], [-4,-1], [-3,-2], [-2,-3], [-1,-4],
  #   [0,-5], [1,-4],[2,-3],[3,-2],[4,-1],
  # ])
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
    centre = np.random.randint(radius + speed, size - radius - speed)
    d = ds[np.random.randint(0, ds.shape[0])]
    centre_new = centre + d
    if i % 10 == 0:
      print(f'i={i}')
    frame = Frame(size).draw_circle(centre, radius)
    frame2 = Frame(size).draw_circle(centre_new, radius)
    imageio.imwrite(out_dir+f"/image{2*i}.png", frame)
    imageio.imwrite(out_dir+f"/image{2*i+1}.png", frame2)


if __name__ == '__main__':
  import argparse  
  parser = argparse.ArgumentParser(description='Generates .pngs of circle bouncing')
  parser.add_argument('out_dir',
                      help='The directory to write to')
  parser.add_argument('height', type=int, default=288)
  parser.add_argument('width', type=int, default=512)
  parser.add_argument('--num_frames', type=int, default=1000)
  parser.add_argument('--radius', type=int, default=2 )

  args = parser.parse_args()
  print(f'Will save to {args.out_dir}')
  generate_random_frames(
    np.array([args.height, args.width]), 
    args.out_dir,
    radius=args.radius,
    total_frames=args.num_frames
  )

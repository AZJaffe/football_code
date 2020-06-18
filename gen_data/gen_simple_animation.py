import matplotlib.pyplot as plt
import numpy as np
import sys
import imageio

class Frame:
  def __init__(self, shape):
    self.frame = 255 * np.ones(tuple(shape) + (3,), dtype='uint8')

  def draw_circle(self, centre, radius, colour=[255,0,0]):
    """Draws a circle on the frame.

    Parameters:
    frame -- the frame to draw the circle on
    centre -- 2 value array which is the centre of the circle
    radius -- radius of the circle
    color -- the color of the circle
    """

    min_x = max(0, centre[0] - radius)
    max_x = min(self.frame.shape[0], centre[0] + radius)
    min_y = max(0, centre[1] - radius)
    max_y = min(self.frame.shape[1], centre[1] + radius)
    # Search the square with side lengths radius centred at centre.
    for i in range(min_x, max_x):
      for j in range(min_y, max_y):
        if np.sqrt((i - centre[0]) ** 2 + (j - centre[1]) ** 2) <= radius:
          self.frame[i,j] = colour
    return self.frame

def display_frames(frames):
  im = plt.imshow(frames[0])
  for i in range(frames.shape[0]):
    im.set_data(frames[i])
    plt.pause(0.02)
  plt.show()

# size should be np array of size (2,)
def generate_random_frames(size, radius, total_frames, speed, out_dir):
  n = np.random.normal([0,0], [1,1])
  d = np.array(speed * n / np.linalg.norm(n), dtype='int') # d ~ speed * Unif(S^2)
  loc = np.random.randint(radius, size - radius)
  print(d)
  for i in range(total_frames):
    if i % 10 == 0:
      print(f'i={i}')
    frame = Frame(size).draw_circle(loc, radius)
    imageio.imwrite(out_dir+f"/image{i}.png", frame)
    # Updates frames
    loc += d
    # Check for out of bounds
    if loc[0] + radius > size[0]: # out of bounds right
      a = loc[0] + radius - size[0]
      loc[0] -= 2*a
      d[0] *= -1
    if loc[0] - radius < 0: # out of bounds left
      a = radius - loc[0]
      loc[0] += 2*a
      d[0] *= -1
    if loc[1] + radius > size[1]: # out of bounds up
      a = loc[1] + radius - size[1]
      loc[1] -= 2*a
      d[1] *= -1
    if loc[1] - radius < 0: # out of bounds down
      a = radius - loc[1]
      loc[1] += 2*a
      d[1] *= -1


if __name__ == '__main__':
  import argparse  
  parser = argparse.ArgumentParser(description='Generates .pngs of circle bouncing')
  parser.add_argument('out_dir',
                      help='The directory to write to')
  parser.add_argument('height', type=int, default=288)
  parser.add_argument('width', type=int, default=512)
  parser.add_argument('--speed', type=int, default=6)
  parser.add_argument('--num_frames', type=int, default=1000)
  parser.add_argument('--radius', type=int, default=40)

  args = parser.parse_args()
  print(f'Will save to {args.out_dir}')
  generate_random_frames(
    np.array([args.height, args.width]), 
    args.radius, 
    args.num_frames, 
    args.speed, 
    args.out_dir,
  )

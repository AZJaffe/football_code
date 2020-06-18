import gfootball.env as fenv
import matplotlib.pyplot as plt
import random
import sys

assert len(sys.argv) == 3

is_gray = (sys.argv[1] == 'gray')
width = sys.argv[2]
height = sys.argv[3]
print(f'gray: {is_gray}, dimensions=({width},{height})')

env = fenv.create_environment(env_name='11_vs_11', 
  representation='pixels_gray' if is_gray else 'pixels', 
  render=True,
  channel_dimensions=(width, height)
)

random.seed(0) # Set seed to 0 to make sure we get the same input each time this is run
state = env.reset()
for _ in range(10):
  state, _, _, _, = env.step(env.action_space.sample())

if is_gray:
  plt.imshow(state.reshape(height, width), cmap='gray', vmin=0, vmax=255)
else:
  plt.imshow(state.reshape(height, width), cmap='gray', vmin=0, vmax=255)
plt.show()
file_name = f'{("gray" if is_gray else "colour")}_{width}_{height}.png'
plt.savefig(file_name)
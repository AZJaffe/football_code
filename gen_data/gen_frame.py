import gfootball.env as fenv
import matplotlib.pyplot as plt
from matplotlib import image
import skimage.transform as tr
from skimage import color


w = 1920
h = 1080

env = fenv.create_environment(env_name='11_vs_11_stochastic', 
  representation='pixels', 
  render=True,
  channel_dimensions=((w, h))
)

state = env.reset()
for _ in range(10):
  state, _, _, _, = env.step(env.action_space.sample())

fig, axes = plt.subplots(5,2, figsize=(2*w, 2*h))

for i in range(5):
  width = w // (2 ** i)
  height = h // (2 ** i)
  downsampled = tr.resize(state, (height, width))
  gray_downsampled = color.rgb2gray(downsampled)
  image.imsave(f'figures/football_sizes/{width}x{height}_color.png', downsampled)
  image.imsave(f'figures/football_sizes/{width}x{height}_gray.png', gray_downsampled, cmap='gray')


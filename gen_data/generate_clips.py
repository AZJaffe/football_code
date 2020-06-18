import numpy as np
import matplotlib.pyplot as plt
import gfootball.env as fenv


w = 480
h = 270
num_frames=100

output = np.zeros((num_frames,h,w,12))

out_file = f'data/football_clips/{w}x{h}_colour.npy'

env = fenv.create_environment(env_name='11_vs_11_stochastic', 
  representation='pixels', 
  render=True,
  stacked=True,
  channel_dimensions=((w, h))
)

done = True
for i in range(num_frames):
  for _ in range(100):
    if done == True:
      state = env.reset()
    state, _, done, _ = env.step(env.action_space.sample())
  
  output[i] = state

np.save(out_file,output)

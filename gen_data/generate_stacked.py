import numpy as np
import matplotlib.pyplot as plt
import gfootball.env as fenv
import cv2

def gen_stacked(w,h,out_file,num_frames=100):
  """Generate and save samples of the stacked pixel representation"""
  output = np.zeros((num_frames,w,h,12))

  env = fenv.create_environment(env_name='11_vs_11', 
    representation='pixels', 
    render=True,
    stacked=True,
    channel_dimensions=(h, w)
  )

  done = True
  for i in range(num_frames):
    for _ in range(100):
      if done == True:
        state = env.reset()
      state, _, done, _ = env.step(env.action_space.sample())
    
    output[i] = state

  np.save(out_file,output)


def gen_clips(w,h,num_frames):
  """Generate clips of football environment
  
  Params:
  w -- width of the images
  h -- height of the images
  num_frames -- number of frames to generate
  """

  output = np.zeros((num_frames,h,w,3), dtype="uint8")

  env = fenv.create_environment(env_name='11_vs_11_stochastic', 
    representation='pixels', 
    render=True,
    channel_dimensions=(w, h)
  )

  done = True
  for i in range(num_frames):
    if done == True:
      state, done = env.reset(), False
    else:
      state, _, done, _ = env.step(env.action_space.sample())
    output[i] = state
  return output

def write_mp4(vid, out_file):
  """Write 3D np ndarray vid to out_file"""
  out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (vid.shape[2], vid.shape[1]), True)
  for i in vid:
    out.write(cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
  out.release()
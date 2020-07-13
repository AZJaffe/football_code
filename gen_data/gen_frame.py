import gfootball.env as fenv
import argparse
import imageio
import os
import time


def gen_frames(*, out_dir, env_name, W, H, N):
  params = locals() # Call this before adding more to local scope
  env = fenv.create_environment(env_name=env_name, 
    representation='pixels', 
    render=True,
    channel_dimensions=((W, H))
  )

  path = f'{out_dir}/{env_name}_{W}x{H}'
  try:
    os.makedirs(path)
  except FileExistsError:
    pass

  with open(f'{path}/description.txt', 'w') as f:
    f.write(f'Parameters are {params}\n')

    done = True
    e = -1 # Start at -1 since we'll increment to 0 immediate
    start = time.monotonic()
    for i in range(N):
      if i % 100 == 0 and i > 0:
        print(f'frame: {i} episode: {e} total_time_s: {time.monotonic() - start} fps: {i/(time.monotonic() - start)}')
        f.flush()
      if done:
        state = env.reset()
        done = False
        e += 1
      else:
        state, _, done, _ = env.step(env.action_space.sample())
      imageio.imwrite(f"{path}/image_{i}.png", state)
      f.write(f'Image {i} is in episode {e}\n') #oops, bug haha
    print(f'total_episodes: {e} total_time_s: {time.monotonic() - start} fps: {N/(time.monotonic() - start)}')
    f.write(f'total_episodes: {e} total_time_s: {time.monotonic() - start} fps: {N/(time.monotonic() - start)}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate images from gfootball environment')
  parser.add_argument('--out_dir',
                      help='the directory to write the data')
  parser.add_argument('--env_name',
                       help='the name of the environment/scenario to run')
  parser.add_argument('--height', type=int,
                       help='the height of the state representation')
  parser.add_argument('--width', type=int,
                       help='the width of the state representation')
  parser.add_argument('--num_frames', type=int,
                       help='the number of frames to generate')
  args = parser.parse_args()
  gen_frames(out_dir=args.out_dir, env_name=args.env_name, H=args.height, W=args.width, N=args.num_frames)
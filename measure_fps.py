import gfootball.env as fenv
import time
import sys

env = fenv.create_environment(env_name='11_vs_11_stochastic', render=True, representation='pixels')

total_frames = int(sys.argv[1])
print(f'total_frames={total_frames}')

start = time.perf_counter()
done = True
for _ in range(total_frames):
  if done:
    env.reset()
  else:
    env.step(env.action_space.sample())

end = time.perf_counter()
print(f'FPS={total_frames/(end - start)}') 

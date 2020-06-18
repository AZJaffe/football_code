import numpy as np
from tensorflow import keras
import gfootball.env as fenv

path = "../models/may6_football_dqn/model_final.hd5"

model = keras.models.load_model(path)

env = fenv.create_environment(
  env_name="academy_empty_goal_close",
  representation="simple115",
  stacked=False,
  logdir='/tmp/football',
  write_goal_dumps=True,
  write_full_episode_dumps=True,
  render=True
)

for _ in range(10):
  state = env.reset()
  done = False
  reward = None
  step = 0
  while done is False:
    action = np.argmax(model.predict(np.array([state])))
    print(action)
    state, reward, done, _ = env.step(action)
    step += 1
  print(reward, step)
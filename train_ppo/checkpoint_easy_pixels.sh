#!/bin/bash

python3 -u -m gfootball.examples.run_ppo2 \
  --level academy_3_vs_1_with_keeper \
  --reward_experiment scoring,checkpoints \
  --policy impala_cnn \
  --cliprange 0.08 \
  --gamma 0.993 \
  --ent_coef 0.003 \
  --num_timesteps 5000000 \
  --max_grad_norm 0.64 \
  --lr 0.000343 \
  --num_envs 16 \
  --noptepochs 2 \
  --nminibatches 8 \
  --nsteps 128 \
  --state pixels \
  --render True \
  "$@"

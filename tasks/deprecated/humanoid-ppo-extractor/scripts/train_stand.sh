#!/bin/bash
cd ppo
python train_custom.py --env_name h1-stand-v0 --seed ${SEED:-42} --total_timesteps 1000000

#!/bin/bash
export WANDB_MODE=disabled
export MUJOCO_GL=osmesa

python custom_train.py \
    --env_name cube-single-play-singletask-v0 \
    --seed ${SEED:-0} \
    --offline_steps 1000000 \
    --eval_interval 100000 \
    --alpha 300.0

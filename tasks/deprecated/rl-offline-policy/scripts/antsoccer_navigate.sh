#!/bin/bash
export WANDB_MODE=disabled
export MUJOCO_GL=osmesa

python custom_train.py \
    --env_name antsoccer-arena-navigate-singletask-v0 \
    --seed ${SEED:-0} \
    --offline_steps 1000000 \
    --eval_interval 100000 \
    --discount 0.995 \
    --alpha 10.0

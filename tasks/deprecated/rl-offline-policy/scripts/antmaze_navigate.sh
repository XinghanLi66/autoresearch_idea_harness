#!/bin/bash
export WANDB_MODE=disabled
export MUJOCO_GL=osmesa

python custom_train.py \
    --env_name antmaze-large-navigate-singletask-v0 \
    --seed ${SEED:-0} \
    --offline_steps 1000000 \
    --eval_interval 100000 \
    --alpha 10.0 \
    --q_agg min

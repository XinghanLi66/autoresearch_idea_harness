#!/bin/bash
set -e
cd /workspace/dual-goal-representations
export PKG_DIR=/workspace/dual-goal-representations
python custom_train.py \
    --env_name=cube-single-noisy-v0 \
    --seed=${SEED:-0} \
    --train_steps=1000000 \
    --eval_interval=100000 \
    --log_interval=5000 \
    --eval_episodes=50 \
    --discount=0.99 \
    --alpha=10.0

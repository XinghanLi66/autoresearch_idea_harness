#!/bin/bash
export WANDB_MODE=disabled
python algorithms/small_scale/custom_nethack_pomdp.py \
    --character ran-hum-neu \
    --train_seed ${SEED:-42} \
    --checkpoints_path "${OUTPUT_DIR:-${SAVE_PATH:-/workspace/saves}/custom}/ran-hum-neu"

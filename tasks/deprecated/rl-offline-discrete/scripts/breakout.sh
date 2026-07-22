#!/bin/bash
python atari_offline/custom_atari.py \
    --game breakout \
    --fraction 0.01 \
    --seed ${SEED:-42} \
    --checkpoints_path "${OUTPUT_DIR:-${SAVE_PATH:-/workspace/saves}/custom}/breakout"

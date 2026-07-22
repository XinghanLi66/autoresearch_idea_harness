#!/bin/bash
# Evaluation script for cv-diffusion-efficiency — SD v2.0 base

WORKDIR_BASE="${OUTPUT_DIR:-examples/workdir}"
METHOD=${METHOD:-"ddim_cfg++"}
CFG_GUIDANCE=${CFG_GUIDANCE:-0.6}
SEED=${SEED:-42}
NUM_IMAGES=${NUM_IMAGES:-10000}
NGPU=${NGPU:-8}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

torchrun --nproc_per_node=$NGPU --master_port=$MASTER_PORT batch_eval.py \
    --model sd20 \
    --method "$METHOD" \
    --cfg_guidance "$CFG_GUIDANCE" \
    --NFE 20 \
    --seed "$SEED" \
    --num_images "$NUM_IMAGES" \
    --workdir "$WORKDIR_BASE/eval_sd20_${SEED}"

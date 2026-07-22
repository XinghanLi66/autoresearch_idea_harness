#!/bin/bash
# Evaluation script for cv-diffusion-cfg — SDXL

WORKDIR_BASE="${OUTPUT_DIR:-examples/workdir}"
METHOD=${METHOD:-"ddim_cfg++"}
CFG_GUIDANCE=${CFG_GUIDANCE:-0.6}
SEED=${SEED:-42}
NGPU=${NGPU:-8}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

torchrun --nproc_per_node=$NGPU --master_port=$MASTER_PORT batch_eval.py \
    --model sdxl \
    --method "$METHOD" \
    --cfg_guidance "$CFG_GUIDANCE" \
    --NFE 10 \
    --seed "$SEED" \
    --workdir "$WORKDIR_BASE/eval_sdxl_${SEED}"

#!/bin/bash

SEED=${SEED:-42}
# Train jointly on all 3 PDE datasets, evaluate on EULER2D (low-viscosity compressible NS)

VICON_WORKDIR=${VICON_WORKDIR:-/workspace/VICON}
VICON_DATA_ROOT=${VICON_DATA_ROOT:-/data/icon-data}
VICON_NUM_WORKERS=${VICON_NUM_WORKERS:-2}
VICON_OUTPUT_DIR=${OUTPUT_DIR:-${SAVE_PATH}/pde-foundation-icl}

cd "$VICON_WORKDIR"

python src/custom_train_eval.py \
    --eval-dataset EULER2D \
    --seed ${SEED:-42} \
    --output-dir "$VICON_OUTPUT_DIR" \
    --data-root "$VICON_DATA_ROOT" \
    --epochs 10 \
    --steps-per-epoch 5000 \
    --batch-size 30 \
    --lr 1e-4 \
    --warmup-steps 5000 \
    --dim-token 1024 \
    --nhead 8 \
    --dim-feedforward 2048 \
    --num-layers 10 \
    --num-workers "$VICON_NUM_WORKERS"

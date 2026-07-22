#!/bin/bash
# Single-GPU fallback (torchrun hangs in unicore container)
# PYTHONUNBUFFERED=1 ensures stdout is flushed immediately
PYTHONUNBUFFERED=1 python custom_mol3d.py \
    --dataset geom_drug --data-dir /data/geom_drug --dataset-type molecule \
    --epochs 150 --batch-size 8 --lr 1e-4 --max-train-hours 20 \
    --num-samples 10000 --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}

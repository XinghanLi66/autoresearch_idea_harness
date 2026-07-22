#!/bin/bash
python custom_mol3d.py \
    --dataset qm9 --data-dir /data/qm9 --dataset-type molecule \
    --epochs 1000 --batch-size 64 --lr 1e-4 \
    --num-samples 10000 --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}

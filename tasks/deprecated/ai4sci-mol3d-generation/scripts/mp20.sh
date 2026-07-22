#!/bin/bash
python custom_mol3d.py \
    --dataset mp20 --data-dir /data/mp20 --dataset-type crystal \
    --epochs 500 --batch-size 64 --lr 5e-5 --max-train-hours 20 \
    --num-samples 10000 --seed ${SEED:-42} --output-dir ${OUTPUT_DIR}

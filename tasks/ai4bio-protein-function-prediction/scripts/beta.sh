#!/bin/bash
# Train and evaluate protein encoder on Beta-lactamase activity prediction (regression)
cd /workspace

python DeepProtein/custom_protein.py \
    --dataset Beta \
    --data-dir /workspace/DeepProtein \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR}/${ENV}

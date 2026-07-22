#!/bin/bash
# Train and evaluate custom CDR model on the epitope_group split.

cd /workspace

python chimera-bench/custom_cdr.py \
    --split epitope_group \
    --data-root ${CHIMERA_DATA_ROOT:-/data/chimera-bench-v1.0} \
    --seed ${SEED:-42} \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --output-dir ${OUTPUT_DIR:-./output} \
    --gpu 0 \

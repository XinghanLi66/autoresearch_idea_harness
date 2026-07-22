#!/bin/bash
# Zero-shot scoring on HER2 antigen (4d5_her2 dataset, ~2.1k rows)
cd /workspace

python AbBiBench/custom_abscore.py \
    --antigen-key 4d5_her2 \
    --dataset-label 4d5_her2 \
    --data-dir /data \
    --metadata-file /data/metadata.json \
    --batch-size 4 --max-rows 0 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR}/${ENV}

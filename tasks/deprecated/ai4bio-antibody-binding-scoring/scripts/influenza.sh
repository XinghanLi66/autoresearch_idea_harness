#!/bin/bash
# Zero-shot scoring on influenza hemagglutinin H1 (3gbn_h1 dataset, ~1.9k rows)
cd /workspace

python AbBiBench/custom_abscore.py \
    --antigen-key 3gbn \
    --dataset-label 3gbn_h1 \
    --data-dir /data \
    --metadata-file /data/metadata.json \
    --batch-size 4 --max-rows 0 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR}/${ENV}

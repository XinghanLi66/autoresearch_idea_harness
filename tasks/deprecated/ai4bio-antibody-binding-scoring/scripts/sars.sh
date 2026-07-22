#!/bin/bash
# Zero-shot scoring on SARS-related hemagglutinin H1 (4fqi_h1 dataset, ~65k rows, subsampled to 5000)
cd /workspace

python AbBiBench/custom_abscore.py \
    --antigen-key 4fqi \
    --dataset-label 4fqi_h1 \
    --data-dir /data \
    --metadata-file /data/metadata.json \
    --batch-size 4 --max-rows 5000 \
    --seed ${SEED:-42} \
    --output-dir ${OUTPUT_DIR}/${ENV}

#!/bin/bash
# Train and evaluate speech enhancement/separation on LibriMix 2-speaker

cd /workspace

DATA_DIR=/data/speech/se/librimix \
N_LAYERS=4 D_MODEL=256 N_HEAD=4 \
MAX_EPOCHS=50 BATCH_SIZE=4 LEARNING_RATE=1e-3 \
python speechbrain/custom_speech_enhancement.py

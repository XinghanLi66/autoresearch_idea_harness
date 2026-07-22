#!/bin/bash
# Train and evaluate speech enhancement on DNS Challenge

cd /workspace

DATA_DIR=/data/speech/se/dns-challenge \
N_LAYERS=4 D_MODEL=256 N_HEAD=4 \
MAX_EPOCHS=50 BATCH_SIZE=4 LEARNING_RATE=1e-3 \
python speechbrain/custom_speech_enhancement.py

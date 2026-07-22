#!/bin/bash
# Train and evaluate ASR encoder on AISHELL-1 (Mandarin)

cd /workspace

DATA_DIR=/data/speech/asr/aishell-1 \
N_LAYERS=4 N_HEAD=4 D_MODEL=256 D_FFN=1024 KERNEL_SIZE=31 \
MAX_EPOCHS=30 BATCH_SIZE=16 LEARNING_RATE=1e-3 WARMUP_STEPS=5000 \
python speechbrain/custom_asr_encoder.py

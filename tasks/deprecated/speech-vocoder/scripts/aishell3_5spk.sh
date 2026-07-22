#!/bin/bash
# Train vocoder on AISHELL-3 5-speaker subset (multi-speaker Chinese)
cd /workspace
DATA_DIR=/data/speech/tts/aishell3-5spk \
UPSAMPLE_RATES=8,8,2,2 D_MODEL=512 \
MAX_EPOCHS=100 BATCH_SIZE=16 LEARNING_RATE=2e-4 \
python speechbrain/custom_vocoder.py

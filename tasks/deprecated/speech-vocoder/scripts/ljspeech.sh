#!/bin/bash
# Train vocoder on LJSpeech (single speaker English)
cd /workspace
DATA_DIR=/data/speech/tts/ljspeech \
UPSAMPLE_RATES=8,8,2,2 D_MODEL=512 \
MAX_EPOCHS=100 BATCH_SIZE=16 LEARNING_RATE=2e-4 \
python speechbrain/custom_vocoder.py

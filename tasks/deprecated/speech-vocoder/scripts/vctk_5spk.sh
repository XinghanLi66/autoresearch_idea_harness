#!/bin/bash
# Train vocoder on VCTK 5-speaker subset (multi-speaker English)
# VCTK-5spk has ~79k utterances vs LJSpeech ~12k, so fewer epochs
# to keep total gradient steps comparable and fit within time limit.
cd /workspace
DATA_DIR=/data/speech/tts/vctk-5spk \
UPSAMPLE_RATES=8,8,2,2 D_MODEL=512 \
MAX_EPOCHS=15 BATCH_SIZE=16 LEARNING_RATE=2e-4 \
python speechbrain/custom_vocoder.py

#!/bin/bash
# Train and evaluate speech enhancement on VoiceBank-DEMAND

cd /workspace

DATA_DIR=/data/speech/se/voicebank-demand \
N_LAYERS=4 D_MODEL=256 N_HEAD=4 \
MAX_EPOCHS=20 BATCH_SIZE=4 LEARNING_RATE=1e-3 \
python speechbrain/custom_speech_enhancement.py

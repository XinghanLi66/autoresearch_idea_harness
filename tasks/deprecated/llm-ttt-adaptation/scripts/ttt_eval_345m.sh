#!/bin/bash
# TTT adaptation + evaluation for GPT-2 Medium (345M).
# Loads real GPT-2 Medium weights from HuggingFace (pre-downloaded to
# /data/gpt2-medium), converts to nanoGPT format, runs TTT adaptation,
# and evaluates on validation data + benchmark datasets.
# Single GPU, no pretraining needed.
set -e
python custom_ttt_eval.py

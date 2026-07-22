#!/bin/bash
# Run feature selection evaluation on 20newsgroups (TF-IDF text features)
set -e
cd /workspace

ENV=20newsgroups SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    SKLEARN_DATA_HOME=/data/sklearn \
    python -u scikit-learn/custom_featsel.py

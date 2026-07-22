#!/bin/bash
# Run feature selection evaluation on MNIST (784 pixel features)
set -e
cd /workspace

ENV=mnist SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    SKLEARN_DATA_HOME=/data/sklearn \
    python -u scikit-learn/custom_featsel.py

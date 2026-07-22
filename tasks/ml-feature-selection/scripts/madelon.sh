#!/bin/bash
# Run feature selection evaluation on Madelon (500 features, 20 informative)
set -e
cd /workspace

ENV=madelon SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    SKLEARN_DATA_HOME=/data/sklearn \
    python -u scikit-learn/custom_featsel.py

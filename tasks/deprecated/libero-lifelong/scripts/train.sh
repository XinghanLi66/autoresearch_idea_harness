#!/bin/bash
cd LIBERO
python -m libero.lifelong.main \
    seed=${SEED:-42} \
    benchmark_name=LIBERO_SPATIAL \
    lifelong=custom \
    use_wandb=false \
    eval.eval=true

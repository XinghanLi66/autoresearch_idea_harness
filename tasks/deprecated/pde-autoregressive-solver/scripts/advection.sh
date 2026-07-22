#!/bin/bash

SEED=${SEED:-42}

python run.py \
  --gpu 0 \
  --data_path /data/PDEBench/1D/Advection/1D_Advection_Sols_beta1.0.hdf5 \
  --loader pdebench_autoregressive_flat \
  --geotype structured_1D \
  --task dynamic_autoregressive \
  --teacher_forcing 0 \
  --lr 0.0005 \
  --weight_decay 1e-4 \
  --scheduler StepLR \
  --space_dim 1 \
  --fun_dim 10 \
  --out_dim 1 \
  --model Custom \
  --n_hidden 64 \
  --n_heads 8 \
  --n_layers 8 \
  --mlp_ratio 2 \
  --slice_num 32 \
  --modes 12 \
  --unified_pos 0 \
  --ref 8 \
  --batch-size 20 \
  --epochs 100 \
  --eval 0 \
  --max_grad_norm 1.0 \
  --seed $SEED \
  --save_name advection_Custom_s${SEED}

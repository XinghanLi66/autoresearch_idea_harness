#!/bin/bash

SEED=${SEED:-42}

python run.py \
  --gpu 0 \
  --data_path /data/fno/airfoil/naca \
  --loader airfoil \
  --geotype structured_2D \
  --task steady \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --space_dim 2 \
  --fun_dim 2 \
  --out_dim 1 \
  --model Custom \
  --n_hidden 128 \
  --n_heads 8 \
  --n_layers 8 \
  --mlp_ratio 2 \
  --slice_num 64 \
  --modes 12 \
  --unified_pos 0 \
  --ref 8 \
  --batch-size 8 \
  --epochs 500 \
  --eval 0 \
  --max_grad_norm 0.1 \
  --seed $SEED \
  --save_name airfoil_Custom_s${SEED}

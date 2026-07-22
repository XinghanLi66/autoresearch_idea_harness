#!/bin/bash

SEED=${SEED:-42}

python run.py \
  --gpu 0 \
  --data_path /data/fno \
  --loader ns \
  --geotype structured_2D \
  --task dynamic_autoregressive \
  --lr 0.0012 \
  --weight_decay 1e-5 \
  --space_dim 2 \
  --fun_dim 10 \
  --out_dim 1 \
  --model Custom \
  --n_hidden 256 \
  --n_heads 8 \
  --n_layers 8 \
  --mlp_ratio 2 \
  --slice_num 32 \
  --modes 12 \
  --unified_pos 1 \
  --ref 8 \
  --batch-size 8 \
  --epochs 200 \
  --eval 0 \
  --max_grad_norm 1.0 \
  --seed $SEED \
  --save_name ns_Custom_s${SEED}

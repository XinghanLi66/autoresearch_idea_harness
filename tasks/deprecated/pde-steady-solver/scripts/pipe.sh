#!/bin/bash

SEED=${SEED:-42}

python run.py \
  --gpu 0 \
  --data_path /data/fno/pipe \
  --loader pipe \
  --geotype structured_2D \
  --task steady \
  --normalize 1 \
  --lr 0.001 \
  --weight_decay 1e-5 \
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
  --batch-size 4 \
  --epochs 500 \
  --eval 0 \
  --seed $SEED \
  --save_name pipe_Custom_s${SEED}

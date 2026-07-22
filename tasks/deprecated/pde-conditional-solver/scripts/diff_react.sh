#!/bin/bash

SEED=${SEED:-42}

python run.py \
  --gpu 0 \
  --data_path /data/PDEBench/2D/DiffReact/2D_diff-react_NA_NA.h5 \
  --loader pdebench_conditional \
  --geotype structured_2D \
  --task dynamic_conditional \
  --ntrain 900 \
  --ntest 100 \
  --T_out 20 \
  --time_input 1 \
  --space_dim 2 \
  --fun_dim 2 \
  --out_dim 2 \
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
  --epochs 30 \
  --normalize 1 \
  --downsamplex 2 \
  --downsampley 2 \
  --eval 0 \
  --seed $SEED \
  --save_name diff_react_Custom_s${SEED}

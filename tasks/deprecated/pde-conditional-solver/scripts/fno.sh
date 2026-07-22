#!/bin/bash

SEED=${SEED:-42}

case "${ENV}" in
  Plasticity)
    python run.py --gpu 0 --data_path /data/fno/ --loader plas \
      --geotype structured_2D --task dynamic_conditional \
      --ntrain 900 --ntest 80 --T_out 20 --time_input 1 \
      --space_dim 2 --fun_dim 1 --out_dim 4 \
      --model FNO --n_hidden 128 --n_heads 8 --n_layers 8 --modes 12 \
      --unified_pos 0 --ref 8 --batch-size 8 --epochs 500 --eval 0 \
      --seed $SEED \
      --save_name plas_FNO_s${SEED} ;;
  SWE)
    python run.py --gpu 0 \
      --data_path /data/PDEBench/2D/SWE/2D_rdb_NA_NA.h5 \
      --loader pdebench_conditional --geotype structured_2D \
      --task dynamic_conditional \
      --ntrain 900 --ntest 100 --T_out 20 --time_input 1 \
      --space_dim 2 --fun_dim 1 --out_dim 1 \
      --model FNO --n_hidden 128 --n_heads 8 --n_layers 8 --modes 12 \
      --unified_pos 0 --ref 8 --batch-size 8 --epochs 300 --eval 0 \
      --seed $SEED \
      --save_name swe_FNO_s${SEED} ;;
  DiffReact)
    python run.py --gpu 0 \
      --data_path /data/PDEBench/2D/DiffReact/2D_diff-react_NA_NA.h5 \
      --loader pdebench_conditional --geotype structured_2D \
      --task dynamic_conditional \
      --ntrain 900 --ntest 100 --T_out 20 --time_input 1 \
      --space_dim 2 --fun_dim 2 --out_dim 2 \
      --model FNO --n_hidden 128 --n_heads 8 --n_layers 8 --modes 12 \
      --unified_pos 0 --ref 8 --batch-size 8 --epochs 300 --eval 0 \
      --seed $SEED \
      --save_name diff_react_FNO_s${SEED} ;;
esac

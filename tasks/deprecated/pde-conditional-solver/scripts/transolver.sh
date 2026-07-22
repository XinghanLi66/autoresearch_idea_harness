#!/bin/bash

SEED=${SEED:-42}

case "${ENV}" in
  Plasticity)
    python run.py --gpu 0 --data_path /data/fno/ --loader plas \
      --geotype structured_2D --task dynamic_conditional \
      --ntrain 900 --ntest 80 --T_out 20 --time_input 1 \
      --space_dim 2 --fun_dim 1 --out_dim 4 \
      --model Transolver --n_hidden 128 --n_heads 8 --n_layers 8 \
      --slice_num 64 --unified_pos 0 --ref 8 --batch-size 8 --epochs 160 --eval 0 \
      --seed $SEED \
      --save_name plas_Transolver_s${SEED} ;;
  SWE)
    python run.py --gpu 0 \
      --data_path /data/PDEBench/2D/SWE/2D_rdb_NA_NA.h5 \
      --loader pdebench_conditional --geotype structured_2D \
      --task dynamic_conditional \
      --ntrain 900 --ntest 100 --T_out 20 --time_input 1 \
      --space_dim 2 --fun_dim 1 --out_dim 1 \
      --model Transolver --n_hidden 128 --n_heads 8 --n_layers 8 \
      --slice_num 64 --unified_pos 0 --ref 8 --batch-size 8 --epochs 50 --eval 0 \
      --seed $SEED \
      --save_name swe_Transolver_s${SEED} ;;
  DiffReact)
    python run.py --gpu 0 \
      --data_path /data/PDEBench/2D/DiffReact/2D_diff-react_NA_NA.h5 \
      --loader pdebench_conditional --geotype structured_2D \
      --task dynamic_conditional \
      --ntrain 900 --ntest 100 --T_out 20 --time_input 1 \
      --space_dim 2 --fun_dim 2 --out_dim 2 \
      --model Transolver --n_hidden 128 --n_heads 8 --n_layers 8 \
      --slice_num 64 --unified_pos 0 --ref 8 --batch-size 8 --epochs 50 --eval 0 \
      --seed $SEED \
      --save_name diff_react_Transolver_s${SEED} ;;
esac

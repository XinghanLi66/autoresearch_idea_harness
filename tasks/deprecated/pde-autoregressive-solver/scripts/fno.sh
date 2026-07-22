#!/bin/bash

SEED=${SEED:-42}

case "${ENV}" in
  NS)
    python run.py --gpu 0 --data_path /data/fno --loader ns \
      --geotype structured_2D --task dynamic_autoregressive \
      --teacher_forcing 0 --lr 0.0005 --weight_decay 1e-4 --scheduler StepLR \
      --space_dim 2 --fun_dim 10 --out_dim 1 \
      --model FNO --n_hidden 64 --n_heads 8 --n_layers 8 --modes 12 \
      --unified_pos 0 --ref 8 --batch-size 20 --epochs 500 --eval 0 \
      --seed $SEED \
      --save_name ns_FNO_s${SEED} ;;
  DiffSorp)
    python run.py --gpu 0 \
      --data_path /data/PDEBench/1D/diffusion-sorption/1D_diff-sorp_NA_NA.h5 \
      --loader pdebench_autoregressive --geotype structured_1D \
      --task dynamic_autoregressive --teacher_forcing 0 \
      --lr 0.0005 --weight_decay 1e-4 --scheduler StepLR \
      --space_dim 1 --fun_dim 10 --out_dim 1 \
      --model FNO --n_hidden 64 --n_heads 8 --n_layers 8 --modes 12 \
      --unified_pos 0 --ref 8 --batch-size 20 --epochs 100 --eval 0 \
      --seed $SEED \
      --save_name diff_sorp_FNO_s${SEED} ;;
  Burgers)
    python run.py --gpu 0 \
      --data_path /data/PDEBench/1D/Burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
      --loader pdebench_autoregressive_flat --geotype structured_1D \
      --task dynamic_autoregressive --teacher_forcing 0 \
      --lr 0.0005 --weight_decay 1e-4 --scheduler StepLR \
      --space_dim 1 --fun_dim 10 --out_dim 1 \
      --model FNO --n_hidden 64 --n_heads 8 --n_layers 8 --modes 12 \
      --unified_pos 0 --ref 8 --batch-size 20 --epochs 200 --eval 0 \
      --seed $SEED \
      --save_name burgers_FNO_s${SEED} ;;
esac

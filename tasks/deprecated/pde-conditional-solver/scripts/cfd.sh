#!/bin/bash

SEED=${SEED:-42}

python run.py \
  --gpu 0 \
  --data_path /data/PDEBench/2D/CFD/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 \
  --loader pdebench_conditional_cfd \
  --geotype structured_2D \
  --task dynamic_conditional \
  --ntrain 900 \
  --ntest 100 \
  --T_out 20 \
  --time_input 1 \
  --space_dim 2 \
  --fun_dim 4 \
  --out_dim 4 \
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
  --downsamplex 2 \
  --downsampley 2 \
  --eval 0 \
  --seed $SEED \
  --save_name cfd_Custom_s${SEED}

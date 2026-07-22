#!/bin/bash
set -e
cd /workspace/CleanDiffuser
export MUJOCO_GL=osmesa
python train_custom.py --env lift --seed ${SEED:-0} --device cuda:0 --save_dir ${SAVE_PATH:-/tmp/robot_imitation_lift}

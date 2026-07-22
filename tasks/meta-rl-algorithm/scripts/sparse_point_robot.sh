#!/bin/bash
set -e
cd "${MLSBENCH_PKG_DIR:-oyster}"
python custom_meta_rl.py --env sparse-point-robot --gpu 0 --seed ${SEED:-42}

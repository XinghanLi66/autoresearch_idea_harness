#!/bin/bash
set -e
cd "${MLSBENCH_PKG_DIR:-oyster}"
python custom_meta_rl.py --env cheetah-vel --gpu 0 --seed ${SEED:-42}

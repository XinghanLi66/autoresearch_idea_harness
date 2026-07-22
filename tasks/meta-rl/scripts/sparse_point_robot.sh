#!/bin/bash
set -e
cd "${MLSBENCH_PKG_DIR:-oyster}"
python launch_custom.py --env sparse-point-robot --gpu 0 --seed ${SEED:-42}

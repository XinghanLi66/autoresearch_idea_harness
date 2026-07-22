#!/bin/bash
# Select environment based on ENV label injected by harness
case "${ENV}" in
  lift_ph)   TASK="lift" ;;
  can_ph)    TASK="can" ;;
  square_ph) TASK="square" ;;
  *)         TASK="can" ;;
esac

cd /workspace/CleanDiffuser
python custom_conditioning.py \
  --env "$TASK" \
  --seed "${SEED:-42}" \
  --gradient-steps 300000 \
  --batch-size 256 \
  --lr 1e-4 \
  --eval-freq 50000 \
  --eval-episodes 50 \
  --dataset-root /data/robomimic/datasets

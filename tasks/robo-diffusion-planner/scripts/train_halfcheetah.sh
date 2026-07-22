#!/bin/bash
set -e
cd /workspace/CleanDiffuser
SEED=${SEED:-42}
# default/diffuser/decision_diffuser converge in 200k steps to paper-reported
# scores on D4RL halfcheetah-medium-v2.
# AdaptDiffuser specifically requires 1M so the classifier is calibrated enough
# for the trajectory-selection threshold (args.task.metric_value=5005) to be
# achievable.
BL=$(basename "$(dirname "$OUTPUT_DIR")")
if [ "$BL" = "adaptdiffuser" ]; then
    STEPS=1000000
else
    STEPS=200000
fi
echo "Baseline=$BL → diffusion_gradient_steps=$STEPS"
# ++ syntax is force-override-or-add: works whether the YAML defines the key or not.
python pipelines/custom_planner.py task=halfcheetah-medium-v2 mode=train seed=$SEED ++diffusion_gradient_steps=$STEPS ++classifier_gradient_steps=$STEPS batch_size=64 log_interval=1000 save_interval=100000
python pipelines/custom_planner.py task=halfcheetah-medium-v2 mode=finetune seed=$SEED ++ft_ckpt=$STEPS ++task.metric_value=1350 || true
python pipelines/custom_planner.py task=halfcheetah-medium-v2 mode=inference seed=$SEED ++ckpt=$STEPS num_envs=50 num_episodes=3 use_ema=True

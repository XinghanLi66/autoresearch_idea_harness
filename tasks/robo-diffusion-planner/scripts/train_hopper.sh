#!/bin/bash
set -e
cd /workspace/CleanDiffuser
SEED=${SEED:-42}
# Most baselines (default/diffuser/decision_diffuser) converge in 200k steps to
# the paper-reported normalized scores on D4RL-MuJoCo medium-v2.
# AdaptDiffuser specifically requires the full 1M-step recipe so the classifier
# is calibrated enough for the trajectory-selection threshold (args.task.metric_value)
# to be achievable; with 200k the classifier under-estimates returns and the
# self-evolving loop can never select trajectories.
BL=$(basename "$(dirname "$OUTPUT_DIR")")
if [ "$BL" = "adaptdiffuser" ]; then
    STEPS=1000000
else
    STEPS=200000
fi
echo "Baseline=$BL → diffusion_gradient_steps=$STEPS"
# ++ syntax is force-override-or-add: works whether the YAML defines the key or not
# (Decision Diffuser config has no classifier_gradient_steps, so plain '=' fails Hydra).
python pipelines/custom_planner.py task=hopper-medium-v2 mode=train seed=$SEED ++diffusion_gradient_steps=$STEPS ++classifier_gradient_steps=$STEPS batch_size=64 log_interval=1000 save_interval=100000
# Finetune step: no-op for default/diffuser/decision_diffuser (template has 'pass'),
# runs the AdaptDiffuser self-evolving phase for the adaptdiffuser baseline.
python pipelines/custom_planner.py task=hopper-medium-v2 mode=finetune seed=$SEED ++ft_ckpt=$STEPS ++task.metric_value=700 || true
python pipelines/custom_planner.py task=hopper-medium-v2 mode=inference seed=$SEED ++ckpt=$STEPS num_envs=50 num_episodes=3 use_ema=True

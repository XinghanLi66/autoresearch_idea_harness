# Robo-Diffusion: Trajectory Planner Design

## Objective
Design one improved trajectory-level diffusion planner for offline robot
control. The research question is: given a fixed D4RL MuJoCo dataset and fixed
diffusion-model training budget, can a planner generate and select better
future trajectories at inference time?

## What You Can Modify
- Trajectory diffusion architecture and conditioning logic
- Classifier guidance / return conditioning inside the planner
- Finetuning logic for planner-specific self-improvement
- Inference-time trajectory sampling and selection

## What Is Fixed
- D4RL datasets and environment names
- Evaluation seeds, rollout count, and hidden-env split
- Top-level training budget and checkpoint numbers in `scripts/train_*.sh`
- Environment stepping and D4RL normalized-score computation

## Evaluation
Evaluated on three D4RL MuJoCo environments:
1. **hopper-medium-v2**
2. **walker2d-medium-v2**
3. **halfcheetah-medium-v2**

Metrics: normalized_score, episode_reward, planning_time

### Training / Eval Protocol
The task intentionally truncates the upstream 1M-step CleanDiffuser recipes to
200k steps so that all baselines fit the benchmark walltime while preserving
the expected relative ordering:

- `diffusion_gradient_steps = 200_000`, `classifier_gradient_steps = 200_000`
- `batch_size = 64`
- Eval rollouts: `num_envs = 50`, `num_episodes = 3` (150 episodes per env)

This protocol applies to `default`, `diffuser`, `decision_diffuser`, and
`adaptdiffuser`. Scores are therefore expected to be in the same ballpark as
published D4RL medium-v2 results, but not an exact 1M-step reproduction.

## Baselines

### diffuser
Planning with Diffusion - Original planner

### decision_diffuser
Decision Diffuser - Return-conditioned planning

### adaptdiffuser
AdaptDiffuser - Adaptive planning with online refinement

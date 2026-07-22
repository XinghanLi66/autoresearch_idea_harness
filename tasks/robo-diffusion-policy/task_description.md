# Robo-Diffusion: Policy Algorithm Design

## Objective
Design a single model-free offline RL policy algorithm that uses a diffusion
actor for action generation and improves D4RL MuJoCo control performance.

This task is intentionally separate from trajectory-diffusion planning. The
agent should modify the policy-level actor/critic learning rule, Q/value
estimation, or inference-time action selection for a Markov policy. It should
not turn the solution into a trajectory planner, classifier-guided planner, or
environment-specific evaluation shortcut.

## What You Can Modify
- Policy algorithm core logic
- Q-function design (if used)
- Action generation strategy
- Training objective
- Actor-critic architecture

## What Is Fixed
- D4RL dataset construction, environment names, hidden split, and evaluation loop
- Random seeds, episode count, vectorized environment count, and checkpoint names
- The overall offline RL setup: train from fixed D4RL buffers, then evaluate a
  policy that maps current observation to one action

## Evaluation
Evaluated on three D4RL MuJoCo environments:
1. **hopper-medium-v2**
2. **walker2d-medium-v2**
3. **halfcheetah-medium-v2**

Metrics: normalized_score, episode_reward, training_time

`halfcheetah-medium-v2` is hidden during intermediate agent tests. Final scores
use a geometric mean over the three environment-specific normalized-score terms.

## Baselines

### default
Diffusion Q-Learning (DQL) — the unmodified template ports cleandiffuser's
`dql_d4rl_mujoco.py` line-for-line (diffusion actor + twin Q critic, BC + Q
loss). This serves as the paper-level DQL reference
(Wang et al., 2022, https://arxiv.org/abs/2208.06193).

### idql
Implicit Diffusion Q-Learning - decoupled actor/critic with τ-expectile IQL
critic and softmax(adv * β) action reweighting at inference
(Hansen-Estruch et al., 2023, https://arxiv.org/abs/2304.10573).

### diffusion_policy
Diffusion Policy — pure behavior cloning with a diffusion actor (no critic,
single-action sampling at inference, Chi et al., 2023,
https://arxiv.org/abs/2303.04137).

## Evaluation Protocol

To keep the protocol fixed across all baselines/agents:

- `gradient_steps = 1,000,000` for every method. CleanDiffuser's package configs
  (`configs/{dql,idql}/mujoco/mujoco.yaml`) and the DQL paper both use 2M, but at
  1M our DQL default already reproduces CleanDiffuser's published numbers
  (hopper 0.997 vs ~1.00, walker2d 0.904 vs ~0.87, halfcheetah 0.513 vs ~0.51),
  so 1M is the better walltime/quality tradeoff for this benchmark. The model
  may shorten training inside its own edits but cannot lengthen it.
- `num_candidates = 50` at inference for every method that uses Q-reranking
  (DQL, IDQL). This matches CleanDiffuser's DQL reference repro and the
  "DQL+selection" variant; CleanDiffuser's IDQL package config defaults to 256
  but is reduced to 50 here so all reranking baselines see the same compute.
  `diffusion_policy` ignores `num_candidates` (sample-1 inference, no critic).
- `num_envs = 50`, `num_episodes = 3`, `use_ema = True` at inference.

Seed=42 is the primary seed; multi-seed (123, 456) is run for the `default`
baseline to confirm hopper-medium-v2 is not cherry-picked.

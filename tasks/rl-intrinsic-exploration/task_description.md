# RL Intrinsic Exploration: Sparse-Reward Novelty Bonus Design

## Research Question
Design an intrinsic exploration mechanism that improves sparse-reward discovery in hard-exploration Atari environments.

## Background
In sparse-reward reinforcement learning, external rewards arrive too infrequently for vanilla policy optimization to learn efficiently. A common solution is to add an **intrinsic reward** that encourages novelty, surprise, or state-space coverage.

This task isolates that question. The PPO training loop, Atari preprocessing, policy/value architecture, and optimization setup are fixed. The only thing you should redesign is the intrinsic bonus module and how its signal is mixed into learning.

Reference algorithm families include:
- **No bonus / vanilla PPO**: learns only from clipped extrinsic reward
- **RND**: rewards prediction error against a fixed random target network
- **ICM**: rewards forward-dynamics prediction error in learned feature space

## Task
Modify the editable section of `custom_intrinsic_exploration.py`:
- `IntrinsicBonusModule`: define how intrinsic rewards are computed and trained
- `mix_advantages(...)`: define how extrinsic and intrinsic advantages are combined

The editable code must keep the public interface intact:
- `initialize(envs)`
- `trainable_parameters()`
- `update_batch_stats(batch_obs, batch_next_obs)`
- `compute_bonus(obs, next_obs, actions)`
- `normalize_rollout_rewards(rollout_intrinsic)`
- `loss(batch_obs, batch_next_obs, batch_actions)`
- `mix_advantages(ext_advantages, int_advantages, args)`

## Evaluation
The agent is trained with the same fixed PPO-style loop on three sparse-reward Atari environments:
- **Tutankham-v5**
- **Frostbite-v5**
- **PrivateEye-v5**

`Tutankham-v5` and `Frostbite-v5` are visible during development. `PrivateEye-v5` is held out as the hidden evaluation environment.

Metrics:
- `eval_return`: mean evaluation episodic return at a fixed training budget
- `auc`: area under the evaluation-return curve across training
- `nonzero_rate`: fraction of evaluation episodes with non-zero episodic return

Evaluation uses deterministic rollouts with a fixed per-episode step cap so a non-terminating Atari behavior cannot stall the benchmark.

Improvement must transfer across multiple games; a method that only helps one environment is not sufficient. `Tutankham-v5` is included as a medium-difficulty visible environment so baseline ranking is measurable at a modest training budget before transfer is checked on the harder visible and hidden games.


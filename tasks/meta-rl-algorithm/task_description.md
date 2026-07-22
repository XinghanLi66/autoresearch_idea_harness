# Meta-RL Algorithm Design

## Objective
Design a complete meta-reinforcement learning algorithm for fast adaptation to new tasks from limited interaction data. You must implement both the **agent** (how to encode context and condition the policy) and the **training algorithm** (how to meta-train the agent across tasks).

## Background
Meta-RL algorithms learn to learn: they train across a distribution of tasks so that at test time, the agent can quickly adapt to a new, unseen task from just a few interactions. The key challenge is designing:

1. **Task inference**: How to encode past experience (context) into a compact task representation
2. **Policy conditioning**: How to condition the policy on this task representation
3. **Meta-training**: How to optimize the agent across tasks so it generalizes to new ones

Different approaches exist: PEARL uses a probabilistic encoder with product-of-Gaussians aggregation; FOCAL uses contrastive learning for task embeddings; VariBAD uses a recurrent encoder with reward prediction.

## Your Task
Modify the `CustomMetaRLAgent` and `CustomMetaRLAlgorithm` classes in `custom_meta_rl.py`. The template provides fixed infrastructure (environment setup, evaluation, replay buffers, network building blocks) — you design the algorithm.

### Agent Interface (`CustomMetaRLAgent`)
Your agent must implement:
- `get_action(obs, deterministic=False)` -> `(action_np, agent_info)` — sample action conditioned on task belief
- `update_context(transition_tuple)` -> `None` — accumulate online experience (called during rollout)
- `adapt()` -> `None` — perform task inference from collected context (called after exploration)
- `clear_context(num_tasks=1)` -> `None` — reset context and task belief
- `infer_posterior(context_tensor)` -> `None` — encode context from replay buffer (for training)
- `context` property — return collected context
- `z` attribute — latent task variable tensor
- `networks` property — list of nn.Module for GPU transfer

### Algorithm Interface (`CustomMetaRLAlgorithm`)
Your algorithm must implement:
- `collect_initial_data()` — gather initial exploration data for all training tasks
- `train_iteration(iteration_idx)` -> `dict` — one meta-training iteration (data collection + gradient updates)
- `networks` property — all networks for GPU transfer

### Available Utilities
The template provides these fixed utilities you can use:
- `build_mlp(input_dim, output_dim, hidden_dim, n_layers)` — simple MLP
- `build_policy(obs_dim, action_dim, latent_dim, net_size)` — TanhGaussianPolicy
- `build_qf(obs_dim, action_dim, latent_dim, net_size)` — Q-function
- `build_vf(obs_dim, latent_dim, net_size)` — V-function
- `create_replay_buffers(env, tasks)` — replay buffer pair
- `sample_context_from_buffer(enc_replay_buffer, indices, batch_size, ...)` — sample context
- `sample_sac_batch(replay_buffer, indices, batch_size)` — sample RL batch
- `collect_data(agent, env, sampler, replay_buffer, enc_replay_buffer, ...)` — collect trajectories
- `InPlacePathSampler` from rlkit — trajectory sampler

## Environments
Three MuJoCo environments with different challenges:

1. **Half-Cheetah Velocity** (`cheetah-vel`): 30 train / 10 test tasks. Target velocities in [0, 3] m/s. Obs dim 20, action dim 6. Dense reward (velocity matching). High-dimensional observations require strong encoding.

2. **Sparse Point Robot** (`sparse-point-robot`): 40 train / 10 test tasks. Goals on a half-circle, sparse reward (+1 near goal, 0 otherwise). Obs dim 2, action dim 2. Sparse reward makes task inference especially challenging.

3. **Point Robot** (`point-robot`): 40 train / 10 test tasks. Goals in [-1, 1]^2. Dense reward (neg L2 distance). Obs dim 2, action dim 2. Tests basic meta-learning quality.

## Evaluation
Performance is measured by `meta_test_return` on each environment: average return on held-out test tasks after meta-training. The evaluation protocol collects exploration trajectories, calls `agent.adapt()`, then evaluates with a deterministic policy.

## Key Design Dimensions
- **Context encoding**: Permutation-invariant (MLP + aggregation) vs. sequential (RNN/GRU) vs. attention
- **Task variable**: Probabilistic (information bottleneck) vs. deterministic
- **Encoder loss**: KL divergence, contrastive, reward prediction, or reconstruction
- **RL algorithm**: SAC variants, policy gradient, or other

## Note on Training Budget
This task intentionally uses 20 meta-training iterations to keep wall time per environment <= 1 hour. This is a short benchmark budget, not the paper-scale training budget.

Absolute returns are not directly comparable to PEARL, VariBAD, or FOCAL paper numbers, which use 500+ iterations or roughly 1.5-2.0e6 environment steps. Only the relative ordering across baselines and agents within this fixed budget is meaningful.

On `sparse-point-robot`, methods that return 0 indicate that no goal was reached within 20 iterations, not algorithmic failure; the environment reward is binary.

The companion [`meta-rl`](../meta-rl/task_description.md) task uses the same budget convention.

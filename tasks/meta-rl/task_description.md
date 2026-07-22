# Meta-RL: Context Encoder for PEARL Task Inference

## Objective
Design a context encoder for the PEARL meta-reinforcement learning algorithm that maps transition tuples (state, action, reward) to latent task representations. The encoder should enable effective task inference from limited interaction data, allowing fast adaptation to unseen tasks.

## Background
PEARL (Probabilistic Embeddings for Actor-critic RL) is a meta-RL algorithm that learns a probabilistic latent task variable z from context transitions. During meta-testing, the agent collects a few transitions from a new task, encodes them into a posterior distribution q(z|c), and conditions its policy on the sampled z.

The context encoder processes individual transition tuples and outputs Gaussian parameters (mean and log-variance). The PEARLAgent aggregates per-transition outputs via product of Gaussians to form the task posterior. Your goal is to design a better encoder architecture.

You can modify the `CustomContextEncoder` class (lines 27-53) and add custom imports (lines 21-23) in `custom_encoder.py`.

## Interface
Your `CustomContextEncoder` must:
- Extend `PyTorchModule` and call `self.save_init_params(locals())` in `__init__`
- Accept `hidden_sizes`, `input_size`, `output_size` as constructor arguments
- Set `self.output_size` attribute in `__init__`
- Implement `forward(self, input, return_preactivations=False)` returning tensors of shape `(*, output_size)`
- Implement `reset(self, num_tasks=1)` to reset any stateful components

## Environments
The encoder is evaluated across three environments with different reward structures and task complexities:

1. **Half-Cheetah Velocity** (`cheetah-vel`): 30 train / 10 test tasks. Target velocities in [0, 3] m/s. Obs dim 20, action dim 6. Dense reward based on velocity matching. Tests encoding quality on a continuous task distribution with high-dimensional observations.

2. **Sparse Point Robot** (`sparse-point-robot`): 40 train / 10 test tasks. Goals on a half-circle, sparse reward (+1 within goal radius, 0 otherwise). Obs dim 2, action dim 2. Tests the encoder's ability to extract task information from sparse reward signals.

3. **Point Robot** (`point-robot`): 40 train / 10 test tasks. Goals sampled uniformly from [-1, 1]^2. Dense reward (negative L2 distance to goal). Obs dim 2, action dim 2. Tests basic encoding quality on a simple but diverse continuous task distribution.

## Evaluation
Performance is measured by `meta_test_return` on each environment: average return on held-out test tasks after 20 meta-training iterations.

## Note on Training Budget
This task intentionally uses 20 meta-training iterations to keep wall time per environment <= 1 hour. This is a short benchmark budget, not the paper-scale training budget.

Absolute returns are not directly comparable to PEARL, VariBAD, or FOCAL paper numbers, which use 500+ iterations or roughly 1.5-2.0e6 environment steps. Only the relative ordering across baselines and agents within this fixed budget is meaningful.

On `sparse-point-robot`, methods that return 0 indicate that no goal was reached within 20 iterations, not algorithmic failure; the environment reward is binary.

The companion [`meta-rl-algorithm`](../meta-rl-algorithm/task_description.md) task uses the same budget convention.

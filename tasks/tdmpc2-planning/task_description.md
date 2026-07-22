# Planning Algorithm for Model-Based RL

## Objective
Design and implement a custom trajectory optimization algorithm for online planning in model-based reinforcement learning. Your code goes in the `custom_plan()` function in `custom_planner.py`. This function is called at every environment step to select actions using the learned world model.

## Background
TD-MPC2 uses **Model Predictive Path Integral (MPPI)** for planning. At each step, the agent:
1. Samples `num_pi_trajs=24` trajectories from the learned policy as warm-starts
2. Iterates `iterations=6` rounds of:
   - Samples `num_samples=512` action sequences from N(mean, std)
   - Rolls out each trajectory through the latent dynamics model for `horizon=3` steps
   - Estimates trajectory value using predicted rewards + terminal Q-value
   - Selects `num_elites=64` best trajectories
   - Updates mean/std using softmax-weighted (temperature=0.5) elite statistics
3. Selects final action via Gumbel-softmax sampling from elites

Alternative planning approaches could improve sample efficiency, convergence speed, or final performance:
- **Cross-Entropy Method (CEM)**: simpler elite selection without softmax weighting
- **iCEM**: improved CEM with temporally correlated noise and keep-elites
- **Gradient-based planning**: backpropagating through the world model
- **Hybrid approaches**: combining sampling with gradient refinement
- **Adaptive methods**: adjusting sampling parameters during optimization

## What You Can Modify
The `custom_plan()` function (lines 15-120) in `custom_planner.py`. You have access to:
- `agent.model`: WorldModel with `encode`, `next`, `pi`, `Q`, `reward` methods
- `agent._estimate_value(z, actions, task)`: evaluates trajectory returns
- `agent._prev_mean`: warm-start buffer from previous planning step
- `agent.cfg`: all configuration parameters (horizon, num_samples, etc.)
- `common.math`: utility functions (gumbel_softmax_sample, two_hot_inv, etc.)

## Evaluation
- **Metric**: Episode reward (higher is better)
- **Environments**: DMControl walker-walk and cheetah-run
- **Model**: TD-MPC2 with 1M parameters, 200K training steps
- **Note**: The planning algorithm affects both data collection quality during training and action selection during evaluation

## Key Constraints
- The function must return a single action tensor of shape `(action_dim,)` clamped to `[-1, 1]`
- The function runs under `@torch.no_grad()` — no gradient computation
- Must update `agent._prev_mean` for temporal consistency across steps
- Planning budget: keep total computation comparable to the default (6 iterations x 512 samples)

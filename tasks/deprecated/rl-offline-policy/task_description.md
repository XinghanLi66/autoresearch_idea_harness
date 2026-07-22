# Offline RL Policy Training

## Objective
Design a better policy training approach for offline reinforcement learning that effectively leverages offline data while maximizing Q-values. Your approach is evaluated on OGBench environments by success rate.

## Background
Offline RL trains policies from fixed datasets without environment interaction. The key challenge is balancing:
1. **Staying close to the data** (behavioral cloning) to avoid out-of-distribution actions
2. **Maximizing Q-values** to discover better actions than those in the dataset

Different approaches tackle this differently:
- **IQL**: Uses implicit Q-learning with advantage-weighted regression (AWR) for the actor
- **TD3+BC / ReBRAC**: Deterministic policy gradient (DDPG) regularized with a BC loss
- **FQL**: Trains a flow-matching BC policy, then distills it into a one-step policy guided by Q-values

## What to implement
Implement the `PolicyTrainer` class with three core methods:
1. `create_policy_networks()` -- Define what neural network(s) your policy uses
2. `compute_actor_loss()` -- Define the training loss for the policy
3. `sample_actions()` -- Define how to produce actions at test time

You may also override `get_target_update_modules()` and `needs_next_actions()` if needed.

## Fixed components
- **Critic**: Twin Q-networks (Value with num_ensembles=2) trained with standard TD loss
- **Target network**: Polyak-averaged target critic for stable Q-value targets
- **Training loop**: 1M offline gradient steps, periodic evaluation
- **Dataset**: OGBench offline datasets with observations, actions, rewards, masks

## Available primitives
- `GaussianActor`: Gaussian policy network (supports const_std, state_dependent_std, tanh_squash)
- `VectorFieldActor`: Vector field network for flow matching (takes observations, actions, optional times)
- `Value`: Value/critic network with optional ensemble
- `MLP`: Multi-layer perceptron with optional layer norm
- `distrax`: JAX probability distributions library
- `optax`: JAX optimizer library

## Evaluation
Success rate on OGBench goal-reaching tasks (antmaze navigation, cube manipulation). Higher is better. The metric ranges from 0.0 (never reaches goal) to 1.0 (always reaches goal).

## Hyperparameters available in config
- `alpha`: BC/regularization coefficient (environment-specific, passed via script)
- `discount`: Discount factor (default 0.99)
- `tau`: Target network EMA rate (default 0.005)
- `lr`: Learning rate (default 3e-4)
- `q_agg`: Q aggregation for target ('min' or 'mean', passed via script)
- `normalize_q_loss`: Whether to normalize Q loss by |Q| mean
- `actor_hidden_dims`, `value_hidden_dims`: Network sizes (default (512,512,512,512))

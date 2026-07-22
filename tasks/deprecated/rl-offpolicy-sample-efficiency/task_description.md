# Off-Policy RL Sample Efficiency: Algorithm Design for Humanoid Locomotion

## Objective
Design a more sample-efficient off-policy reinforcement learning algorithm that achieves higher performance than FastTD3, FastSAC, and PPO on humanoid locomotion tasks within the same training budget.

## Background
FastTD3 is a high-performance off-policy RL algorithm that combines TD3 with distributional value estimation (categorical DQN with 101 atoms), parallel environments (128 envs), observation normalization, mixed-precision training, and torch.compile for speed. It uses a deterministic actor with Gaussian exploration noise and twin distributional critics with clipped double Q-learning.

Your task is to design an improved off-policy algorithm by modifying the Actor, Critic, and/or update functions. The training infrastructure (environment, replay buffer, evaluation) is fixed.

## What You Can Modify
The editable section contains:
- **Actor**: Network architecture, forward pass, exploration strategy
- **Critic**: Q-network architecture, distributional parameters, ensemble design
- **build_algorithm()**: Component construction, optimizers, schedulers, auxiliary modules
- **update_critic()**: Critic loss computation, target calculation, auxiliary objectives
- **update_actor()**: Policy gradient objective, entropy regularization, etc.
- **soft_update()**: Target network update strategy

## Key Design Dimensions
- **Architecture**: LayerNorm, spectral norm, residual connections, different activations
- **Exploration**: Noise schedule, parameter-space noise, curiosity, optimistic exploration
- **Value estimation**: Distributional RL (atoms, quantiles), ensemble methods, uncertainty
- **Policy optimization**: Entropy regularization, policy constraints, advantage weighting
- **Sample reuse**: Update-to-data ratio, replay prioritization, n-step returns
- **Representation**: Feature normalization, auxiliary losses, self-predictive representations

## Constraints
- The algorithm must work with continuous action spaces (actions clipped to [-1, 1])
- Must use the provided replay buffer and environment interface
- Total training budget: 100,000 gradient steps with 128 parallel environments
- Must produce deterministic actions at evaluation time via `actor(obs)`

## Evaluation
The algorithm is evaluated on three HumanoidBench locomotion tasks:
- **h1hand-stand-v0**: Humanoid standing balance
- **h1hand-walk-v0**: Humanoid walking
- **h1hand-run-v0**: Humanoid running

Performance is measured as mean episode return over 3 evaluation rollouts at the end of training. Higher is better.

# Safe RL: Constraint-Handling Mechanism Design

## Objective
Design a constraint-handling mechanism for safe reinforcement learning. Your code goes in `custom_lag.py`, a subclass of PPO registered as `CustomLag`. Reference implementations (PPOLag using Lagrange multiplier, CPPOPID using PID controller) are provided as read-only.

## Background
Safe RL aims to maximize reward while satisfying safety constraints (keeping episode cost below a limit). The key challenge is how to adaptively balance reward and cost: the Lagrangian approach converts the constrained problem to an unconstrained dual problem via a multiplier lambda, while PID methods use control theory for more responsive constraint satisfaction. You must design: (1) a multiplier update rule in `_update()`, and (2) an advantage combination formula in `_compute_adv_surrogate()`.

## Evaluation
Evaluated on 3 Safety-Gymnasium environments to test generalization:
- **SafetyPointGoal1-v0**: point robot navigating to goals while avoiding hazards
- **SafetyCarGoal1-v0**: car robot (non-holonomic) navigating to goals while avoiding hazards
- **SafetyPointButton1-v0**: point robot pressing goal buttons while avoiding hazards

Metrics: episode reward (higher is better) and episode cost (lower is better, target <= 25.0). Each environment trains for 2M steps.

## Baselines
- **naive**: no constraint handling (pure PPO, ignores cost)
- **ppo_lag**: Lagrangian multiplier updated via Adam optimizer
- **pid_lag**: PID controller for multiplier update

# Safe Exploration Under Fixed Cost Budgets

## Research Question
Design a constraint-handling mechanism for safe reinforcement learning when the cost budget is fixed and must be respected across different environments.

The task isolates one modular question: given a PPO-style safe RL scaffold, what update rule best keeps episode cost under a budget while retaining reward?

## Background
Safe exploration is not just about reducing cost on average. In many settings the agent must satisfy a hard budget during training and evaluation, which makes the controller for the cost multiplier the key algorithmic choice.

This benchmark focuses on a single transferable component:
- how the Lagrange multiplier is updated,
- how reward and cost advantages are combined, and
- how aggressively the policy should react when the budget is overshot.

## Task
Modify the `CustomLag` class in `custom_lag.py`. The rest of the pipeline is fixed: PPO backbone, OmniSafe training loop, and evaluation script.

Your implementation receives episode cost statistics from the logger and must control the policy update so that cost stays within a fixed budget.

## Evaluation
The benchmark runs on three Safety-Gymnasium environments that stress different control geometries:

- `SafetyPointGoal1-v0`
- `SafetyCarGoal1-v0`
- `SafetyPointButton1-v0`

Each environment is trained for 1M steps with a fixed cost budget of 25.0.

## Metrics
Higher is better for:
- `ep_ret`
- `budget_success_rate`

Lower is better for:
- `ep_cost`

The budget success rate is 1 when the final episode cost is at or below the fixed budget, and 0 otherwise.

## Baselines
- `naive`: pure PPO, ignores cost
- `ppo_lag`: standard Lagrangian PPO with Adam multiplier updates
- `pid_lag`: PID-style multiplier control
- `budget_margin_lag`: budget-aware Lagrangian proxy that reacts more aggressively to budget overshoot

## Notes
This task intentionally reuses the existing OmniSafe scaffold so that all baselines are expressed as edits to the same code path.

The exact CUP / APPO / Simmer family was not ported here because those implementations are heavier than this benchmark needs. The `budget_margin_lag` baseline is a practical MLS-Bench substitute that captures the same budget-first behavior without introducing a new training stack.

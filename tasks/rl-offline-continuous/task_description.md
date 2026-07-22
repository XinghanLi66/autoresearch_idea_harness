# Offline RL: Q-Value Overestimation Suppression in Continuous Control


## Objective
Design and implement an offline RL algorithm that suppresses Q-value overestimation while learning from static datasets. Your code goes in `custom.py`. Four reference implementations (BC, TD3+BC, IQL, CQL) are provided as read-only.

## Background
In offline RL, standard Q-learning tends to overestimate Q-values for out-of-distribution actions since the agent cannot collect new data, leading to poor policy performance.

## Constraints
- **Network dimensions are fixed at 256.** All MLP hidden layers must use 256 units. A `_mlp()` factory function is provided in the FIXED section for convenience. You may define custom network classes but hidden widths must remain 256.
- **Total parameter count is enforced.** The training loop checks that total trainable parameters do not exceed 1.2x the largest baseline architecture. Focus on algorithmic innovations (loss functions, regularization, training procedures), not network capacity.
- Do NOT simply copy a reference implementation with minor changes

## Evaluation
Trained and evaluated on HalfCheetah, Hopper, Walker2d using D4RL MuJoCo `medium-v2` datasets. Additional held-out environments (not shown during intermediate testing) are used to assess generalization. Metric: D4RL normalized score (0 = random, 100 = expert).

# Offline RL: Dexterous Manipulation with Narrow Expert Data (Adroit)


## Objective
Design and implement an offline RL algorithm for high-dimensional dexterous manipulation from narrow human demonstration data (~25 demos). Your code goes in `custom_adroit.py`. Three reference implementations (BC-10%, AWAC, ReBRAC) are provided as read-only.

## Background
Adroit tasks involve a 24-DoF robotic hand with high-dimensional action spaces (24-30 dims) and narrow `human-v1` datasets containing only ~25 human teleoperation demonstrations, creating severe distribution shift compared to locomotion tasks.

## Constraints
- **Network dimensions are fixed at 256.** All MLP hidden layers must use 256 units. A `_mlp()` factory function is provided in the FIXED section for convenience. You may define custom network classes but hidden widths must remain 256.
- **Total parameter count is enforced.** The training loop checks that total trainable parameters do not exceed 1.2x the largest baseline architecture. Focus on algorithmic innovations (loss functions, regularization, training procedures), not network capacity.
- Do NOT simply copy a reference implementation with minor changes

## Evaluation
Trained and evaluated on Pen (rotation), Door (opening), Hammer (nailing) using Adroit `human-v1` datasets. Additional held-out environments (not shown during intermediate testing) are used to assess generalization. Metric: D4RL normalized score (0 = random, 100 = expert).

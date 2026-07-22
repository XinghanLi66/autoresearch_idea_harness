# Offline RL: Discrete Action Control on Atari


## Objective
Design and implement an offline RL algorithm for discrete action spaces with pixel observations. Your code goes in the `QNetwork` and `OfflineAlgorithm` classes in `custom_atari.py`. Three reference implementations (BC, CQL, BCQ) are provided as read-only.

## Background
The offline datasets are "mixed" quality replay buffer data from a partially trained DQN agent. The agent must learn entirely from this fixed dataset without environment interaction during training.

## Constraints
- The `NatureDQNEncoder` (CNN feature extractor) is FIXED and must not be replaced or modified. Your `QNetwork` must use it via `self.encoder = NatureDQNEncoder(...)`. The convolutional layers are verified at runtime.
- Total model parameter count must not exceed 5,000,000. This is enforced at runtime; exceeding it will crash training.
- Focus on algorithmic innovation (loss functions, training procedures, action selection) rather than scaling up network capacity.
- Do NOT simply copy a reference implementation with minor changes

## Evaluation
Trained and evaluated on Breakout (4 actions), Pong (6 actions), Qbert (6 actions) using d4rl-atari "mixed" datasets. Additional held-out environments (not shown during intermediate testing) are used to assess generalization. Metric: mean episode return over 10 evaluation episodes.

# Online RL: Value-Based Methods for Visual Control (Atari)


## Objective
Design and implement a value-based RL algorithm for visual/Atari environments using CNN feature extraction. Your code goes in `custom_value_atari.py`. Three reference implementations (DQN, DoubleDQN, C51) are provided as read-only.

## Background
Atari games require learning from raw pixel observations (84x84 grayscale, 4 stacked frames). Value-based methods must learn effective visual representations alongside Q-value estimation. Key challenges include high-dimensional observations, sparse rewards, and memory-efficient experience replay. Different approaches address these through distributional value functions, frame stacking, or architecture innovations.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified
- Total parameter count is enforced at runtime
- Focus on algorithmic innovation: new loss functions, update rules, exploration strategies, etc.
- Do NOT simply copy a reference implementation with minor changes

## Evaluation
Trained and evaluated on Breakout, Pong, BeamRider. Additional held-out environments (not shown during intermediate testing) are used to assess generalization. Metric: mean episodic return over 10 evaluation episodes (higher is better).

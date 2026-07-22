# Online RL: Value-Based Methods for Discrete Control


## Objective
Design and implement a value-based RL algorithm for discrete action spaces. Your code goes in `custom_value_discrete.py`. Three reference implementations (DQN, DoubleDQN, C51) are provided as read-only.

## Background
Value-based methods estimate Q-values Q(s,a) for each state-action pair and derive a policy by selecting actions with the highest Q-value. Key challenges include overestimation bias, sample efficiency, and representing uncertainty. Different approaches address these through double Q-learning, distributional value functions, or prioritized replay.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified
- Total parameter count is enforced at runtime
- Focus on algorithmic innovation: new loss functions, update rules, exploration strategies, etc.
- Do NOT simply copy a reference implementation with minor changes

## Evaluation
Trained and evaluated on CartPole-v1, LunarLander-v2, Acrobot-v1. Additional held-out environments (not shown during intermediate testing) are used to assess generalization. Metric: mean episodic return over 10 evaluation episodes (higher is better).

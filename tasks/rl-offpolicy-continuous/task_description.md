# Online RL: Off-Policy Actor-Critic for Continuous Control


## Objective
Design and implement an off-policy actor-critic RL algorithm for continuous control. Your code goes in `custom_offpolicy_continuous.py`. Three reference implementations (DDPG, TD3, SAC) are provided as read-only.

## Background
Off-policy methods maintain a replay buffer of past experience and update the policy using data collected under previous policies. Key challenges include overestimation bias in Q-value estimates, exploration-exploitation tradeoff, and sample efficiency. Different approaches address these through twin critics, entropy regularization, target smoothing, or delayed updates.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified
- Total parameter count is enforced at runtime
- Focus on algorithmic innovation: new loss functions, update rules, exploration strategies, etc.
- Do NOT simply copy a reference implementation with minor changes

## Evaluation
Trained and evaluated on HalfCheetah-v4, Hopper-v4, Walker2d-v4. Additional held-out environments (not shown during intermediate testing) are used to assess generalization. Metric: mean episodic return over 10 evaluation episodes (higher is better).

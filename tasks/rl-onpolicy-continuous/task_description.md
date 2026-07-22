# Online RL: On-Policy Actor-Critic for Continuous Control


## Objective
Design and implement an on-policy actor-critic RL algorithm for continuous control. Your code goes in `custom_onpolicy_continuous.py`. Three reference implementations (PPO, RPO, PPO-Penalty) are provided as read-only.

## Background
On-policy methods collect trajectories using the current policy, compute advantages via Generalized Advantage Estimation (GAE), and update the policy using mini-batch optimization. Key challenges include sample efficiency, stability of policy updates, and balancing exploration with exploitation. Different approaches address these through clipped surrogate objectives, stochasticity injection, or direct policy gradient estimation.

## Constraints
- Network architecture dimensions are FIXED and cannot be modified
- Total parameter count is enforced at runtime
- Focus on algorithmic innovation: new loss functions, update rules, exploration strategies, etc.
- Do NOT simply copy a reference implementation with minor changes

## Evaluation
Trained and evaluated on HalfCheetah-v4, Hopper-v4, Walker2d-v4. Additional held-out environments (not shown during intermediate testing) are used to assess generalization. Metric: mean episodic return over 10 evaluation episodes (higher is better).

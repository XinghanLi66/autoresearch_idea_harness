# Inverse RL: Reward Learning from Expert Demonstrations

## Objective
Design and implement an inverse reinforcement learning (IRL) algorithm that learns a reward function from expert demonstrations. Your code goes in `custom_irl.py`, specifically the `RewardNetwork` and `IRLAlgorithm` classes. Three reference implementations (GAIL, AIRL, BC) from the `imitation` library are provided as read-only context.

## Background
Inverse reinforcement learning recovers a reward function that explains observed expert behavior. The learned reward is then used to train a policy via standard RL (PPO in this benchmark). Key challenges include:
- Designing reward network architectures that capture the structure of expert behavior
- Balancing discriminator training with policy improvement
- Avoiding reward hacking where the policy exploits learned reward artifacts
- Ensuring the learned reward generalizes across different states visited during training

Different IRL approaches address these through adversarial training (GAIL), potential-based reward shaping (AIRL), or direct behavioral cloning. Your goal is to design a novel reward network architecture or IRL training algorithm that outperforms these baselines.

## Evaluation
Trained and evaluated on three MuJoCo locomotion environments using pre-generated expert demonstrations: HalfCheetah-v4, Hopper-v4, Walker2d-v4. Metric: mean episodic return over 10 evaluation episodes (higher is better). The policy is trained using PPO with the learned reward signal.

# Humanoid Control: PPO Feature Extractor Architecture

## Objective
Improve PPO performance on humanoid locomotion by designing a better feature extractor network architecture. You can modify the `CustomFeatureExtractor` class (lines 20-38) and add custom imports (lines 14-16) in `train_custom.py`.

## Background
The training uses Stable Baselines3 PPO with 4 parallel environments on three tasks from humanoid-bench: h1-stand-v0, h1-walk-v0, and h1-run-v0. The Unitree H1 humanoid robot must learn standing, walking, and running locomotion. The feature extractor processes raw proprioceptive observation vectors into feature representations used by the policy and value networks.

The default feature extractor is a 2-layer MLP with Tanh activations (64 hidden units, 64 output features). Training runs for 1M timesteps with a learning rate of 3e-4.

## Interface
Your `CustomFeatureExtractor` must:
- Inherit from `BaseFeaturesExtractor`
- Call `super().__init__(observation_space, features_dim)` in `__init__`
- Accept `observation_space` and `features_dim` as constructor arguments
- Implement `forward(self, observations) -> torch.Tensor` returning a tensor of shape `(batch, features_dim)`

## Evaluation
Final performance is measured by mean reward over 20 evaluation episodes with deterministic policy after 1M training timesteps, across all three environments.

# Behavioral Cloning: Observation Encoder Design for Robot State Fusion

## Research Question
Design an improved observation encoder that fuses multiple robot state modalities for behavioral cloning. In robot manipulation, observations consist of heterogeneous components (end-effector pose, gripper state, object state) that may benefit from non-trivial fusion strategies beyond simple concatenation.

## What You Can Modify
The `CustomObsEncoder` class (lines 19-47) in `custom_obs_encoder.py`. This module receives a dictionary of observation tensors and must return a fused feature vector.

Interface:
- **Input**: `obs_dict` -- dictionary with keys:
  - `robot0_eef_pos`: [B, 3] end-effector position
  - `robot0_eef_quat`: [B, 4] end-effector quaternion orientation
  - `robot0_gripper_qpos`: [B, 2] gripper joint positions
  - `object`: [B, D_obj] object state
- **Output**: [B, output_dim] fused feature vector
- **Required attribute**: `self.output_dim` (int) -- dimensionality of the output

You may add parameters to `__init__`, define helper methods, and add learnable layers.

## Evaluation
- **Metric**: `success_rate` -- rollout success rate in the environment (higher is better)
- **Tasks**: Lift, Can, Square (robot manipulation with proficient human demonstrations)
- **Dataset**: ~200 proficient human demonstrations per task, low-dimensional observations
- **Policy**: GMM (Gaussian Mixture Model) with 5 mixture components, trained with NLL loss. A 2-layer MLP backbone (1024, 1024) with ReLU feeds into GMM heads (means, log-stds, mixture logits) on top of encoder output
- **Training**: 2000 epochs, Adam optimizer (lr=1e-4), batch size 100
- **Rollout**: 50 episodes per task, horizon 400 steps, every 50 epochs

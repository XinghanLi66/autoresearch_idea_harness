# Generative Training Objective for Robot Imitation Learning

## Objective
Design a novel generative training objective that best learns multi-modal action distributions from robot demonstrations. You must implement both the training loss and the inference-time sampling procedure.

## Background
Imitation learning from demonstrations requires learning a policy pi(a|o) that maps observations to actions. When expert demonstrations are multi-modal (multiple valid actions for the same observation state), the choice of training objective significantly impacts performance:

- **MSE Regression**: Simple behavioral cloning with L2 loss averages over modes, producing suboptimal actions
- **DDPM (Denoising Diffusion)**: Iterative denoising from noise to actions, high quality but slow (many sampling steps)
- **Flow Matching**: ODE-based transport from noise to data along straight trajectories, faster inference
- **CVAE**: Encoder-decoder with latent variable, single-step generation but may suffer from posterior collapse

Each approach uses a different loss function, noise schedule, and sampling procedure. The question is: what training formulation works best for ACTION generation in robotics (different characteristics from image generation)?

## What You Must Implement
Modify the `ImitationObjective` class, which has two key methods:

1. **`compute_loss(network, actions, condition)`**: Given the backbone network, clean demonstration actions (B, 16, 10), and observation condition (B, obs_dim), return a scalar training loss.

2. **`sample(network_ema, condition, prior_shape)`**: Given the EMA network and observation condition, generate actions of the specified shape.

You may also modify the `CONFIG` dictionary and add helper functions/classes within the editable region.

## Fixed Components (DO NOT MODIFY)
- **Backbone network**: ChiUNet1d (1D convolutional U-Net) with signature `network(x, t, condition)` where x is (B, horizon, act_dim), t is (B,) timestep, condition is (B, obs_steps*obs_dim)
- **Dataset**: Robomimic demonstrations (state-based, low-dimensional)
- **Training loop**: AdamW optimizer, cosine LR schedule, EMA updates, 300k gradient steps
- **Evaluation**: 50 rollout episodes, measuring task success rate

## Environments
- **Lift**: Pick up a cube (obs_dim=19, act_dim=10, easiest)
- **Can**: Pick and place a can (obs_dim=23, act_dim=10, medium)
- **Square**: Insert a square nut onto a peg (obs_dim=23, act_dim=10, hardest)

## Metric
**Success rate** (higher is better): fraction of evaluation episodes where the task is completed.

## Tips
- The backbone network takes a timestep input `t` -- you control what values to pass and what the network learns to predict
- Actions are normalized to [-1, 1] range
- The horizon is 16 timesteps; at inference, steps [obs_steps-1 : obs_steps-1+action_steps] are executed
- Consider: noise schedules, what the network predicts (noise vs. velocity vs. clean data), number of sampling steps, and any regularization
- Fewer sampling steps = faster inference, but potentially lower quality

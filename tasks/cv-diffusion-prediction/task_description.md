# Diffusion Prediction Parameterization

## Background

In DDPM training, the model predicts a target from noisy input x_t. The noisy
sample is constructed as:

    x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon

There are three standard parameterizations for what the model predicts:

1. **Epsilon prediction** (Ho et al., 2020): predict the noise epsilon
2. **X0 prediction**: directly predict the clean image x_0
3. **V-prediction** (Salimans & Ho, 2022): predict the velocity
   v = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * x_0

These are mathematically equivalent (one can be converted to any other), but
they result in different loss landscapes and training dynamics, leading to
different FID scores under the same training budget.

## Research Question

Can we design a prediction parameterization that achieves better FID than
the standard epsilon, v-prediction, and x0-prediction baselines?

## Task

You are given `custom_train.py`, a self-contained training script that trains
an unconditional UNet2DModel (google/ddpm-cifar10-32 architecture) on CIFAR-10.

The editable region contains two functions:

1. `compute_training_target(x_0, noise, timesteps, schedule)` — defines what
   the model should predict during training.

2. `predict_x0(model_output, x_t, timesteps, schedule)` — recovers the
   predicted clean image from the model's output (used during DDIM sampling).

These two functions must be consistent: the sampling procedure must correctly
invert the training parameterization.

The `schedule` dict provides precomputed noise schedule tensors:
- `alphas_cumprod`: cumulative product of (1 - beta)
- `sqrt_alpha`: sqrt(alphas_cumprod)
- `sqrt_one_minus_alpha`: sqrt(1 - alphas_cumprod)

## Evaluation

- Dataset: CIFAR-10 (32x32)
- Model: UNet2DModel (diffusers backbone) at three scales:
  - Small: block_out_channels=(64,128,128,128), ~9M params, batch 128
  - Medium: block_out_channels=(128,256,256,256), ~36M params, batch 128
  - Large: block_out_channels=(256,512,512,512), ~140M params, batch 64
- Training: 35000 steps per scale, AdamW lr=2e-4, EMA rate 0.9995, 8-GPU DDP
- Inference: 50-step DDIM sampler
- Metric: FID (lower is better), computed with clean-fid against CIFAR-10 train set (50k samples)

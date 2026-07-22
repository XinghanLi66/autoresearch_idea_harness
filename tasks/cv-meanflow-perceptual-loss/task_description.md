# Flow Matching with Perceptual Loss

## Background

Flow matching trains a neural network to predict velocity fields that transport samples
from noise to data. Traditional training uses only MSE loss on the predicted velocity:

    loss = ||v_pred - v_target||^2

However, we can also compute the **denoised image** from the predicted velocity:

    x_denoised = x_t - t * v_pred

And apply perceptual losses (LPIPS, gradient loss, etc.) on x_denoised to encourage
the network to generate high-quality images, not just accurate velocities.

## Research Question

Can adding perceptual losses to flow matching training improve FID scores?

## Task

You are given `custom_train_perceptual.py`, a self-contained training script that trains a
small DiT on CIFAR-10 (32x32) using flow matching with mean velocity objectives.

The editable region contains the **loss computation** in the training loop:

```python
# Current: MSE loss only
loss_mse = ((pred_mean_vel - mean_vel_target) ** 2).mean()
loss = loss_mse
```

The fixed code already exposes:
- `lpips_fn(x_denoised, x_target)` - perceptual loss
- `compute_gradient_loss(x_denoised, x_target)` - gradient-domain loss
- `compute_multiscale_loss(x_denoised, x_target)` - multi-resolution loss

**Key constraint**: Only apply auxiliary losses when `t > 0.1` to avoid instability at small noise levels.

## Evaluation

- Dataset: CIFAR-10 (32x32)
- Model: SmallDiT (512 hidden, 8 layers, ~40M params)
- Training: 10000 steps, batch size 128
- Metric: FID (lower is better), computed with clean-fid against CIFAR-10 train set
- Inference: 10-step Euler sampler

## Baselines

1. **mse_base**: Pure MSE on velocity — clean linear formulation, the floor reference
2. **lpips_grad**: MSE + Charbonnier-smoothed L1 on velocity + LPIPS + Sobel gradient + multiscale L1 on the denoised image, with `(1-t)^2` perceptual schedule and `t<=0.1` mask (spatial-domain perceptual recipe)
3. **lpips_spectral**: lpips_grad's stack + FFT-magnitude L1 on the denoised image (spatial + frequency-domain perceptual recipe)

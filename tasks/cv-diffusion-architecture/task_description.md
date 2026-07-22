# Diffusion Model Architecture Design

## Background

The UNet backbone is the standard architecture for denoising diffusion models.
Key design choices include:

- **Block types**: Whether to use pure convolutional blocks (`DownBlock2D` /
  `UpBlock2D`) or blocks with self-attention (`AttnDownBlock2D` /
  `AttnUpBlock2D`), and at which resolution levels.
- **Attention placement**: Self-attention is expensive at high resolutions
  (32x32) but may improve global coherence. The original DDPM places attention
  only at 16x16.
- **Depth and normalization**: `layers_per_block`, `norm_num_groups`,
  `attention_head_dim`, and other structural hyperparameters.
- **Custom modules**: Entirely new backbone designs (e.g., hybrid
  convolution-transformer, gated blocks, multi-scale fusion) as long as they
  satisfy the input/output interface.

## Research Question

What UNet architecture achieves the best FID on unconditional CIFAR-10
diffusion, given a fixed training procedure (epsilon prediction, DDIM sampling,
same optimizer and schedule)?

## Task

You are given `custom_train.py`, a self-contained unconditional DDPM training
script on CIFAR-10. Everything is fixed except the `build_model(device)`
function.

Your goal is to design a model architecture that achieves **lower FID** than
the baselines. The model must satisfy:

- **Input**: `(x, timestep)` where `x` is `[B, 3, 32, 32]`, `timestep` is `[B]`
- **Output**: an object with `.sample` attribute of shape `[B, 3, 32, 32]`
- `UNet2DModel` from diffusers satisfies this interface, but you may also build
  a fully custom `nn.Module`.

Channel widths are provided via the `BLOCK_OUT_CHANNELS` environment variable
(e.g. `"128,256,256,256"`) so the same architecture scales across evaluation
tiers. `LAYERS_PER_BLOCK` (default 2) is also available.

## Evaluation

- Dataset: CIFAR-10 (32x32, unconditional)
- Training: fixed epsilon prediction, MSE loss, AdamW lr=2e-4, EMA
- Model scales:
  - Small: block_out_channels=(64,128,128,128), ~9M params, batch 128
  - Medium: block_out_channels=(128,256,256,256), ~36M params, batch 128
  - Large: block_out_channels=(256,512,512,512), ~140M params, batch 64
- Training: 35,000 steps per scale, EMA rate 0.9995
- Metric: FID (lower is better), computed with clean-fid against CIFAR-10
  train set (50,000 samples)
- Inference: 50-step DDIM sampling

## Baselines

1. **standard**: Original DDPM architecture (Ho et al., 2020). Self-attention
   placed only at the 16x16 resolution level. This is the default
   `google/ddpm-cifar10-32` configuration.
2. **full-attn**: Self-attention at every resolution level (32x32, 16x16, 8x8,
   4x4). More expressive but significantly more compute and memory per step.
3. **no-attn**: Pure convolutional UNet with no per-resolution self-attention.
   Only the mid-block retains its default self-attention layer. Fastest and
   fewest parameters.

# Class-Conditional Diffusion: Conditioning Injection Methods

## Background

Class-conditional diffusion models generate images conditioned on a class label.
The key design choice is **how** to inject the class information into the UNet:

- **Cross-Attention**: Class embedding serves as key/value in a cross-attention
  layer after each ResBlock. Used in Stable Diffusion for text conditioning.
- **Adaptive Normalization (AdaLN-Zero)**: Class embedding modulates LayerNorm
  with learned scale, shift, and gating parameters. Used in DiT.
- **FiLM Conditioning**: Class embedding is added to the timestep embedding
  and injected via adaptive GroupNorm (scale/shift) in ResBlocks.

## Research Question

Which conditioning injection method achieves the best class-conditional FID
on CIFAR-10?

## Task

You are given `custom_train.py`, a self-contained class-conditional DDPM
training script with a small UNet on CIFAR-10 (32x32, 10 classes).

The editable region contains:

1. `prepare_conditioning(time_emb, class_emb)` — controls how class embedding
   is combined with the timestep embedding before entering ResBlocks.

2. `ClassConditioner(nn.Module)` — an additional conditioning module applied
   after each ResBlock, enabling methods like cross-attention or adaptive norm.

Your goal is to design a conditioning injection method that achieves **lower
FID** than the baselines.

## Evaluation

- Dataset: CIFAR-10 (32x32, 10 classes)
- Model: UNet2DModel (diffusers backbone) at three scales:
  - Small: block_out_channels=(64,128,128,128), ~9M params, batch 128
  - Medium: block_out_channels=(128,256,256,256), ~36M params, batch 128
  - Large: block_out_channels=(256,512,512,512), ~140M params, batch 64
- Training: 35000 steps per scale, AdamW lr=2e-4, EMA rate 0.9995
- Metric: FID (lower is better), computed with clean-fid against CIFAR-10 train set (50k samples)
- Inference: 50-step DDIM sampling (class-conditional)

## Baselines

1. **concat-film**: Class embedding added to timestep embedding, injected via
   FiLM (adaptive GroupNorm) in ResBlocks. Simplest method.
2. **cross-attn**: Class embedding used as key/value in cross-attention layers
   after ResBlocks. Most expressive method.
3. **adanorm**: Class embedding generates scale/shift/gate parameters for
   adaptive LayerNorm after ResBlocks. DiT-style method.

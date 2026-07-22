# Observation Conditioning Mechanisms for Generative Robot Policies

## Research Question
Design an improved observation conditioning mechanism for a diffusion-based robot policy. In generative policies, observations must condition the action generation process -- but *how* this conditioning is applied (concatenation, FiLM, cross-attention, adaptive normalization, etc.) significantly impacts policy quality.

The core challenge: given an observation embedding and intermediate features from a denoising network, how should the observation information modulate the denoising process at each layer?

## What You Can Modify
The `ConditioningModule` class (editable region) in `custom_conditioning.py`. This module wraps the fixed base denoising network and controls how observations influence action generation.

Interface:
- `__init__(self, obs_dim, act_dim, hidden_dim=256)` -- initialize conditioning layers. `obs_dim` is the total observation dimension, `act_dim` is the action dimension, `hidden_dim` is the feature dimension used in the base network.
- `encode_condition(self, obs)` -> tensor -- process raw observation (B, obs_dim) into a conditioning representation. This could be a simple embedding, a set of per-layer parameters, or any other representation.
- `condition_features(self, h, t_emb, cond, layer_idx)` -> tensor -- apply conditioning to intermediate features. Takes feature tensor `h` (B, hidden_dim), timestep embedding `t_emb` (B, hidden_dim), conditioning representation `cond` (output of encode_condition), and `layer_idx` (int, 0-indexed layer number). Returns conditioned features (B, hidden_dim).

The base denoising network (FIXED) is a simple residual MLP with 4 blocks:
```
input: (B, act_dim) noisy action + (B, hidden_dim) timestep embedding
For each block i in [0, 1, 2, 3]:
    h = Linear(hidden_dim -> hidden_dim) + GELU + Linear(hidden_dim -> hidden_dim)
    h = h + residual  (skip connection)
    h = condition_features(h, t_emb, cond, layer_idx=i)  # YOUR conditioning
output: Linear(hidden_dim -> act_dim)
```

You may add parameters, helper layers, and state to `ConditioningModule`. The module is jointly trained with the base network end-to-end.

## Evaluation
- **Metric**: `success_rate` -- fraction of successful task completions (higher is better)
- **Tasks**: Lift (pick up a cube), Can (pick and place a can), Square (insert a square nut onto a peg)
- **Dataset**: ~200 proficient human demonstrations per task, low-dimensional state observations
- **Policy**: DDPM diffusion policy with DDIM sampling, 100 diffusion steps training, 10 steps DDIM inference
- **Training**: 300k gradient steps, AdamW (lr=1e-4), batch size 256, EMA rate 0.995
- **Evaluation**: 50 episodes per task, success determined by task-specific reward threshold

## Background
Current conditioning mechanisms in the literature:
- **Concatenation**: simplest approach -- concat obs with noisy action at input
- **FiLM** (Feature-wise Linear Modulation): obs produces per-layer scale+shift: y = gamma(obs)*x + beta(obs)
- **Cross-Attention**: features attend to observation tokens via multi-head attention
- **AdaLN** (Adaptive Layer Normalization): obs modulates LayerNorm parameters, as used in DiT

Each mechanism trades off expressiveness, compute cost, and training stability. Novel combinations or entirely new mechanisms may outperform these baselines.

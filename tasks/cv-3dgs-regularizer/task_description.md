# 3D Gaussian Splatting Regularizer Design

## Objective

Design a regularizer on the 3D Gaussian parameters that improves novel-view
reconstruction quality (higher PSNR / SSIM, lower LPIPS) on MipNeRF360
scenes, without using any depth, normal, or feature-level supervision.

## Background

3D Gaussian Splatting (3DGS) optimizes millions of anisotropic Gaussians
(means, scales, quats, opacities, spherical-harmonic colors) by gradient
descent on a per-scene photometric loss:

```
loss_photo = 0.8 * L1(rendered, gt) + 0.2 * (1 - SSIM(rendered, gt))
```

The photometric loss alone is under-constrained: optimization often
produces elongated "needle" Gaussians, semi-transparent floaters, and
other artifacts that look correct on training views but hurt novel-view
quality. Hand-designed regularizers address this:

- **Scale/opacity L1 penalty** (gsplat default): encourages compact,
  sparse Gaussians.
- **Anisotropy / aspect-ratio penalty** (PhysGaussian): caps
  `max(scale) / min(scale)` to keep Gaussians near-isotropic.
- **Neighbor consistency** (CANOR-style blob priors): penalizes
  parameter diversity among spatially-adjacent Gaussians.
- **Feature distillation** (LangSplat / FeatureGS): aligns 3D features
  with a frozen 2D model's features.

Each of these is a tiny, modular addition to the loss, yet they can
improve reconstruction quality by 0.3–1.0 PSNR on standard benchmarks.

## Task

Implement the function `compute_regularizer(splats, step, scene_scale)`
in `gsplat/custom_regularizer.py`. Its scalar return value is added to
the photometric loss at every training step, for the entire 30k-step
per-scene optimization.

### Editable Region

The editable region in `custom_regularizer.py` is delimited by comment
banners (`EDITABLE REGION` → `End editable region`). You may:

- Add helper functions or module-level constants inside the region.
- Change the body of `compute_regularizer` freely.
- Import additional modules if needed (add imports inside the editable
  region; `torch` and `torch.nn.functional as F` are pre-imported at the
  top of the file).

You **must** keep the public signature `compute_regularizer(splats,
step, scene_scale) -> scalar torch.Tensor`.

### Inputs

- `splats` — `torch.nn.ParameterDict` with keys
  (all first dim is N = current number of Gaussians):

  | key | shape | notes |
  |-----|-------|-------|
  | `means`     | `[N, 3]` | world-space positions |
  | `scales`    | `[N, 3]` | log-scales; `torch.exp(...)` for actual |
  | `quats`     | `[N, 4]` | rotation quaternion (unnormalized) |
  | `opacities` | `[N]`    | logit; `torch.sigmoid(...)` for [0,1] |
  | `sh0`       | `[N, 1, 3]` | DC SH coefficients |
  | `shN`       | `[N, K, 3]` | higher-order SH, K depends on degree |

- `step` — current training iteration (`0` to `max_steps - 1`).
- `scene_scale` — approximate scene radius for distance normalization.

### Output

A scalar `torch.Tensor` (a single number, any device). It is added
directly to the photometric loss with no extra scaling.

## Baselines

| baseline | what it does |
|----------|--------------|
| `none`      | returns 0 — photometric loss only |
| `scale_opa` | L1 on `exp(scales)` + `sigmoid(opacities)`, coeff 1e-2 each (3DGS-MCMC) |
| `erank_opa` | scale_opa (full strength) + erank log-barrier with warmup at step 7000 (Hyung et al. NeurIPS 2024, arXiv:2406.11672). Pushes effective rank ≥ 2 for planar Gaussians while keeping compactness pressure. |

## Reference Baseline PSNRs (MipNeRF360)
| baseline    | garden | bicycle | bonsai | stump |
|---|---|---|---|---|
| `none`      | 29.067 | 26.641 | 32.531 | 27.460 |
| `scale_opa` | 29.318 | **26.844** | 32.685 | **27.720** |
| `erank_opa` | **29.623** | 26.725 | **32.943** | 27.668 |

Your goal: beat the best baseline per-scene. `scale_opa` and `erank_opa`
are complementary — `scale_opa` wins outdoor scenes with high-frequency
detail (bicycle/stump), `erank_opa` wins where needle artifacts hurt
quality (garden/bonsai). Beating BOTH requires a regularizer that
adapts to scene type or combines their mechanisms more carefully than
naive stacking.

## Evaluation

4 MipNeRF360 scenes: **garden**, **bicycle**, **bonsai**, **stump**.
Each scene trains for 30k steps (identical optimizer schedule across
baselines and agent submissions), then evaluates on held-out views
using metrics:

- **PSNR** (higher is better) — primary metric
- **SSIM** (higher is better)
- **LPIPS** (lower is better)

The densification strategy is fixed to gsplat's `DefaultStrategy`
(original 3DGS clone/split/prune) — the regularizer is the only thing
you change.

## Hints

- Regularizer magnitudes should be small; they're added directly to the
  photometric loss. Photometric loss typical magnitude: 0.03–0.1 per
  step. Keep your regularizer in the 1e-4 to 1e-1 range.
- `step` lets you schedule the regularizer (warmup, cooldown, etc.).
- `scene_scale` normalizes distances: `means / scene_scale` gives
  unit-scale coordinates that transfer across scenes.
- Backward pass will flow through every operation — keep gradients
  finite, avoid `log(0)` / `exp(big_number)` / divide-by-zero.
- Each scene has 30k iterations, so the regularizer is evaluated
  ~30k times. Keep it O(N) at most (avoid all-pairs N×N computations
  on `means`).

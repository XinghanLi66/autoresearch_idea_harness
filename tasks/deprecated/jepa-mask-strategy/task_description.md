# I-JEPA Mask Sampling Strategy

## Design note

This task isolates a single ML-science research question from the I-JEPA
paper (Assran et al. 2023, "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture", https://arxiv.org/abs/2301.08243):

> **Given a fixed ViT encoder, EMA target encoder, predictor, and post-hoc
> frozen-encoder linear probe trained on Tiny-ImageNet-200, what mask-sampling strategy
> yields the most useful learned representation, measured by linear-probe
> top-1 accuracy?**

I-JEPA's central empirical claim (Sec 4.4 / Sec 9 "Ablations" -- Table 6,
"Ablating masking strategy"; the description appears in "Predicting
Multiple Targets Together" of the paper) is that the particular
*multi-block* recipe — one large *context* block (scale 0.85–1.0) with
FOUR small *target* blocks (scale 0.15–0.20, aspect 0.75–1.5) carved out
of it — beats rasterized, single-block, and random-patch masking on
ImageNet-1k linear probe.

Direct paper citation, Table 6 (https://arxiv.org/abs/2301.08243, page 8):

| Mask Type    | Targets               | Freq | Context                   | Avg.Ratio | Top-1 |
|--------------|-----------------------|------|---------------------------|-----------|-------|
| multi-block  | Block(0.15, 0.2)      | 4    | Block(0.85, 1.0) × Compl. | 0.25      | 54.2  |
| rasterized   | Quadrant              | 3    | Complement                | 0.25      | 15.5  |
| block        | Block(0.6)            | 1    | Complement                | 0.4       | 20.2  |
| random       | Random(0.6)           | 1    | Complement                | 0.4       | 17.6  |

The mask sampler is therefore an atomic, generalizable algorithmic
component: it can be swapped in and out without touching the rest of the
pipeline.

## Experimental design: fixed context, vary target

A naive port of Table 6 changes BOTH (a) the encoder context shape/size
and (b) the target sampling. That confounds two variables: a baseline
that gives the encoder more visible patches will look "better" simply
because the prediction task is easier, not because the masking shape is
better. To isolate the variable that the paper actually argues for —
**target sampling strategy** — this task **fixes the encoder context
block to scale (0.85, 1.0) across all four masking baselines** and only
varies the target distribution. The encoder block sampling routine is
the verbatim ``_sample_block_size`` + ``_sample_block_mask`` from
``facebookresearch/ijepa/src/masks/multiblock.py`` (commit ``main``),
with ``allow_overlap=False`` so the context never sees what it must
predict.

This makes the comparison apples-to-apples: across baselines, the
encoder always receives ~0.85–1.0 of the patches minus the target
patches; only the target distribution (multi-block / single-block /
random / horizontal-strip) varies.

Editable interface: a single class, `CustomMaskSampler`, with
`__init__(self, grid_size)` and `sample(self, generator) -> (context_idx,
[target_idx, ...])`. The collator calls `sample()` once per image and
truncates the resulting tensors to the per-batch minimum so they stack.
Every other component — ViT (tiny/small/base), EMA momentum schedule,
predictor MLP/transformer, linear probe head, AdamW optimizer, cosine LR
+ WD schedule, smooth-L1 prediction loss — is fixed infrastructure
copied from the official I-JEPA repo configs (`configs/in1k_vith14_ep300.yaml`).

A small `CONFIG_OVERRIDES` dict (whitelist: `pred_depth`, `pred_dim`)
lets each method tune predictor capacity, since some mask strategies
(e.g. multi-block with disjoint targets) benefit from a deeper
predictor.

## Why Tiny-ImageNet (not CIFAR-10)

The paper's `enc_mask_scale=(0.85, 1.0)` + `npred=4` recipe assumes a
~196-patch grid (ImageNet-1k at 224×224 / patch=16 → 14×14). On the
CIFAR-10 8×8 = 64-patch grid, four target blocks at scale 0.15–0.20
(~10 patches each) carve away most of the context and the SSL signal
degenerates — the multi-block context collapses to ~5 patches. Switching
to **Tiny-ImageNet-200 at 64×64 / patch=4 → 16×16 = 256 patches** puts
the recipe back into its native regime (slightly larger than the
paper's 196-patch grid) and restores the partial ordering the paper
reports.

## Research question

Propose and implement a mask-sampling strategy that maximizes
Tiny-ImageNet-200 linear-probe top-1 accuracy across three ViT scales
(tiny, small, base).

## Editable component

File: `eb_jepa/custom_mask.py`, lines 59–91 (between
`# EDITABLE REGION START` and `# EDITABLE REGION END`).

Interface:

```python
class CustomMaskSampler:
    def __init__(self, grid_size):  # grid_size = (H_p, W_p) = (16, 16) for Tiny-ImageNet
        ...
    def sample(self, generator) -> (LongTensor[N_ctx], [LongTensor[N_tgt_m], ...]):
        ...

CONFIG_OVERRIDES = {"pred_depth": <int>, "pred_dim": <int>}
```

`generator` is a freshly-seeded `torch.Generator` unique to each
(worker, batch, image) call. Indices must lie in `[0, H_p * W_p)`.

## Evaluation

- **Dataset**: Tiny-ImageNet-200 (Le & Yang 2015,
  https://www.kaggle.com/c/tiny-imagenet), 64×64 RGB, 200 classes,
  100k train + 10k val. Patch_size=4 → 16×16 = 256 patches per image.
  Random baseline accuracy = 0.5%; published SSL baselines reach 30–50%
  linear-probe top-1.
- **Metric**: linear-probe top-1 accuracy on the Tiny-ImageNet val set
  (10000 images, 50/class). After JEPA pretraining, a fresh linear probe is
  trained post-hoc on frozen EMA-target features (mean-pooled).
- **Test envs**: ViT-Tiny (12L/3H/192D), ViT-Small (12L/6H/384D),
  ViT-Base (12L/12H/768D). All trained 100 epochs with batch 256,
  walltime 12 h per env on a single 80 GB H100 (compute=0.33).

## Baselines and expected partial ordering

Per I-JEPA paper Table 6 (ImageNet-1k 1% linear-probe ablation), the
expected partial ordering on the ablation is:

> **multiblock > block > random > rasterized**

(54.2 / 20.2 / 17.6 / 15.5 in the paper's exact numbers). On
Tiny-ImageNet with our **fixed-context** experimental design, the expected
qualitative ordering remains `multiblock > block > random > rasterized`;
this benchmark measures whether proposed target samplers can improve within
that controlled setup.

1. `multiblock` — paper-faithful port of
   `facebookresearch/ijepa/src/masks/multiblock.py` with the
   `in1k_vith14_ep300.yaml` hyperparameters (4 target blocks scale
   0.15–0.2, 1 context block scale 0.85–1.0, aspect 0.75–1.5,
   `allow_overlap=False`, `min_keep=10`). Citation: Assran et al. 2023.
2. `block` — ONE big target block (scale 0.5–0.75, aspect 0.75–1.5);
   context = enc Block(0.85–1.0) ∩ target-complement. Adapted from
   I-JEPA Table 6 ``block`` row (paper used scale 0.6 single block).
3. `rasterized` — target = a contiguous horizontal strip (4 rows × 16
   cols = 64 patches) at random vertical offset; context = enc
   Block(0.85–1.0) ∩ strip-complement. Adapted from I-JEPA Table 6
   ``rasterized`` row (paper used 3 of 4 image quadrants).
4. `random` — target = ~30% of patches sampled uniformly at random
   (independent of spatial structure); context = enc Block(0.85–1.0) ∩
   target-complement. Adapted from I-JEPA Table 6 ``random`` row (paper
   used Random(0.6) targets with image-complement context).

## Hints

- The editable region must contain ONLY the `CustomMaskSampler` class
  and the `CONFIG_OVERRIDES` dict. The collator (above) and training
  loop (below) are fixed.
- Use the supplied `generator` for all random ops to keep runs
  reproducible across DataLoader workers.
- The grid is 16×16 = 256 patches. Multi-block target scale 0.15–0.20
  yields ~38–51 patches per target; scale 0.85–1.0 context yields
  ~218–256 patches before target carving. This matches the paper's
  ImageNet regime closely.

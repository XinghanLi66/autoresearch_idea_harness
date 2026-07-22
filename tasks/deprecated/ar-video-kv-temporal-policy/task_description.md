# ar-video-kv-temporal-policy

## Research Question

Design a better temporal frame-retention policy for autoregressive video generation.
Given a fixed host model (FAR — Flow Autoregressive Reconstruction, a transformer-based AR video model) and a hard frame-count budget, can you decide which historical frames to keep, demote, or drop at each generation step so that long-horizon video quality (PSNR/SSIM/LPIPS vs ground truth) stays high while the retained-history size stays low?

This is a **real-rollout benchmark**: the policy directly controls which latent frames the FAR transformer attends to during video prediction on UCF-101 and DMLab.
Metrics are measured on real decoded pixel frames against ground-truth video clips — not proxies.

This task isolates one scientific question: **temporal history management for AR video generation**.
You may not change the video tokenizer, sampler, architecture, dataset, or prompt set.

## Harness Design

The evaluator runs FAR (`far_model.py` + `autoencoder_dc_model.py`) directly in a **policy-managed generation loop**:

1. Encode the first `n_context` real video frames to latent space.
2. For each prediction step (1 to `n_predict`):
   a. Run FAR's ODE solver to generate the next latent frame, attending to all frames in `kept_latents`.
   b. Append the generated latent to `kept_latents`.
   c. Call `policy.build_temporal_cache_plan(chunk_meta_list, rollout_state, budget_state)`.
   d. Apply the plan: literally prune `kept_latents` to only the frames marked `keep` or `demote_to_long_term` (respecting `budget_state['max_frames']`).
3. Decode all generated frames and measure PSNR / SSIM / LPIPS vs. ground-truth frames.

The policy therefore controls **which real video latents the model attends to** at every generation step.
Evicted frames are truly removed from the attention context — they are not re-encoded in subsequent steps.

Key model facts:
- FAR-B (130M params, `far_model.py`): used for UCF-101 short prediction.
- FAR-B-Long (150M params, `far_long_model.py`): used for DMLab long prediction.
- DCAE (8× spatial compression, 32 latent channels, 64px): encodes/decodes pixel frames.
- Inference: 20 ODE steps (FlowMatchEulerDiscreteScheduler), unconditional generation.

## What You Can Modify

You may edit only the `VideoTemporalKVPolicy` class in `custom_video_eval.py`.

The single editable method is:

- `build_temporal_cache_plan(chunk_meta_list, rollout_state, budget_state) -> TemporalCachePlan`

`chunk_meta_list` contains one `FrameMeta` per frame currently in `kept_latents`:

| Field | Description |
|---|---|
| `chunk_id` | 0-based index in current history (0 = oldest) |
| `age` | frames since creation (0 = most recent) |
| `dynamic_age` | same as age |
| `recentness` | `1 / (1 + age)` |
| `size_mb` | latent memory footprint (float32) |
| `boundary_score` | normalised latent L2 change from prev frame [0,1] |
| `feature_drift` | cumulative L2 drift from first context frame [0,1] |
| `temporal_similarity` | cosine similarity to previous frame [0,1] |
| `motion_strength` | magnitude of frame-to-frame latent change [0,1] |
| `keyframe_score` | same as motion_strength (local peak proxy) |
| `long_term_resident` | always False in this harness |
| `compression_state` | always "none" |

`rollout_state`:

- `step`: current prediction step (0 = generating first new frame)
- `budget_capacity_mb`: latent buffer capacity for this budget regime
- `total_steps`: total number of prediction steps

`budget_state`:

- `max_frames`: hard upper bound on number of frames to retain
- `capacity_mb`: same as budget_capacity_mb

`TemporalCachePlan` fields the policy may set:

| Field | Type | Description |
|---|---|---|
| `chunk_decisions` | `List[ChunkDecision]` | per-frame `action` ("keep"/"drop"/"demote_to_long_term") + `priority` |
| `retention_family` | str | descriptive tag: "recency" / "anchor" / "queue" / "chunkwise" |
| `anchor_preservation_rule` | str | informational only in this harness |
| `boundary_transition_rule` | str | informational only in this harness |
| `queue_depth` | int | informational only in this harness |
| `long_term_budget_fraction` | float | informational only in this harness |
| `chunkwise_reuse` | bool | informational only |
| `compression_mode` | str | informational only |

The evaluator's `_apply_plan` enforces two rules on top of `chunk_decisions`:
1. The most recent frame is always kept (even if marked `drop`).
2. If the number of kept frames > `budget_state['max_frames']`, the oldest frames are dropped until the budget is met.

## What You Cannot Modify

- the host FAR model (transformer + DCAE)
- the evaluation datasets (UCF-101 test set, DMLab)
- the generation protocol (n_context, n_predict, n_inference_steps, seed)
- the metric definitions (PSNR / SSIM / LPIPS)

## Controlled Evaluation

Visible workloads:

| Workload | Dataset | Context | Predict | Model | Budget |
|---|---|---|---|---|---|
| `ucf101_short_prediction` | UCF-101 (50 clips) | 5 frames | 11 frames | FAR-B | `medium_history_budget` |
| `ucf101_short_prediction` | UCF-101 (50 clips) | 5 frames | 11 frames | FAR-B | `tight_history_budget` |
| `dmlab_long_prediction` | DMLab (20 clips) | 36 frames | 36 frames | FAR-B-Long | `medium_history_budget` |

Visible budget regimes:

| Budget | Max Frames | Description |
|---|---|---|
| `full_history_budget` | unlimited | keep all frames (upper-bound quality reference) |
| `medium_history_budget` | 8 frames | moderate memory pressure |
| `tight_history_budget` | 4 frames | aggressive eviction |

## Metrics

The parser expects `TEST_METRICS:` lines with:

| Metric | Description |
|---|---|
| `temporal_quality_main` | primary quality score (0–100; maps PSNR linearly, capped at 40 dB → 100) |
| `psnr` | Peak Signal-to-Noise Ratio (dB) vs ground truth predicted frames |
| `ssim` | Structural Similarity Index vs ground truth |
| `lpips` | LPIPS perceptual distance (AlexNet; lower is better) |
| `peak_kv_memory_mb` | retained latent buffer size (MB) at `max_frames` |
| `fps` | predicted frames per second |
| `n_frames_kept` | effective retained history length at max budget |
| `eval_mode` | `real_rollout` (always) |

## Canonical Baselines

| Baseline | Family | Description |
|---|---|---|
| `recent_window` | recency | naive most-recent: keep the highest-recentness frames |
| `packcache` | anchor | paper-backed (arXiv:2601.04359): temporal anchor + keyframe check precedes drift eviction |
| `causalcache_vdm` | queue | paper-backed (arXiv:2411.16375): shared pool + queue eviction + long-term resident promotion |
| `flowcache` | chunkwise | SOTA paper-backed (arXiv:2602.10825): chunkwise adaptive caching with boundary detection |

### Baseline arXiv Sources

| Baseline | Canonical Paper | arXiv |
|---|---|---|
| `packcache` | PackCache: Training-Free Acceleration for Unified Autoregressive Video Generation via Compact KV-Cache | 2601.04359 |
| `flowcache` | Flow Caching for Autoregressive Video Generation | 2602.10825 |
| `causalcache_vdm` | Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing | 2411.16375 |

## FAR Model Weights

Required weights (placed at `FAR_WEIGHTS_DIR=/data/far_weights/`):

| File | Model | Task |
|---|---|---|
| `short_video_prediction/FAR_B_UCF101_Uncond64-381d295f.pth` | FAR-B transformer | UCF-101 prediction |
| `long_video_prediction/FAR_B_Long_DMLab_Action64-c09441dc.pth` | FAR-B-Long transformer | DMLab prediction |
| `dcae/DCAE_UCF101_Res64-9da18dcf.pth` | DCAE VAE | UCF-101 encode/decode |
| `dcae/DCAE_DMLab_Res64-17035ae5.pth` | DCAE VAE | DMLab encode/decode |

HuggingFace repository: `guyuchao/FAR_Models`

## Notes

- FAR is the fixed host model / runtime substrate for this domain. It does not appear as a baseline.
- Features in `FrameMeta` (boundary_score, temporal_similarity, etc.) are computed from the latent tensors, not from decoded pixel space. They are lightweight and aligned with the model's internal representation.
- `demote_to_long_term` is treated identically to `keep` in `_apply_plan`; policies may use it as a semantic tag.
- The policy is evaluated under a strict frame-count budget. Memory efficiency matters — policies that waste budget on uninformative frames will produce lower-quality predictions.
- For long-horizon DMLab prediction (36 context + 36 predict), frame retention decisions compound: wrong choices early propagate to many subsequent steps.
- Reference quality at `full_history_budget` on UCF-101: PSNR ≈ 23.1 dB, SSIM ≈ 0.78, LPIPS ≈ 0.056 (50 clips, 20 ODE steps).

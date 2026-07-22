# LLM Online RL: Importance-Sampling Granularity for Policy Optimization

## Objective
Design and implement a custom **importance-sampling (IS) strategy** for LLM online RL. Your code goes in the `compute_custom_policy_loss()` function in `custom_policy_loss.py`. The read-only reference file `core_algos.py` contains built-in policy-loss implementations — `compute_policy_loss_vanilla` (token-level PPO), `compute_policy_loss_gspo` (sequence-level), `compute_policy_loss_dppo_kl`, `compute_policy_loss_clip_cov`, and more — that you may study.

## Background
In PPO-style LLM RL, the per-token policy objective uses the importance ratio
```
r_{i,t} = exp(log_prob_new(y_{i,t}) - log_prob_old(y_{i,t}))
```
applied to per-token advantages. The choice of **granularity** of this ratio — and of the clipping — is an open research axis:

- **Token-level IS** (vanilla PPO, GRPO): Each token has its own ratio and is clipped independently via `clip(r_{i,t}, 1-ε, 1+ε)`. This is the standard in verl/OpenRLHF. Variance can be very high for long LLM responses because each ratio is noisy and compounding errors across tokens inflate gradient variance.

- **Sequence-level IS** (GSPO, Zheng et al. 2025, https://arxiv.org/abs/2507.18071): A single scalar ratio per sequence, `s_i = exp( mean_t (log_prob_new − log_prob_old) )`, is broadcast to every token. This dramatically reduces variance for long responses and was shown to stabilize training of MoE models, at the cost of losing per-token correction.

- **Truncated / first-k-token IS**: Apply the ratio only on a prefix of the response, or detach the ratio beyond a cutoff. Motivated by the observation that late tokens often have very large log-ratio magnitudes (they are most affected by compounding drift). DAPO (Yu et al. 2025, https://arxiv.org/abs/2503.14476) and related work use token-length-decoupled clipping and dynamic sampling.

- **CISPO-style clipped IS with stop-grad** (MiniMax M1, https://arxiv.org/abs/2506.13585): Use the clipped ratio in a stop-gradient, recovering a REINFORCE-like gradient on `log π` scaled by a bounded IS weight, to avoid zeroing out the gradient on clipped tokens.

- **Other variants**: dual-clip PPO, geometric-mean aggregation over groups, per-prompt normalised ratios, etc.

The research question: **design a novel importance-sampling granularity / clipping strategy that improves math-reasoning accuracy over vanilla token-level PPO.**

## Evaluation
Your policy loss is used to train **Qwen2.5-0.5B** (full parameter training) using the verl framework with standard **GRPO advantage estimation** (so the only variable is the policy-loss / IS strategy). Training uses **simpleRL-Zoo MATH level-3-5 (Qwen split)** — a curated ~8K-problem training set derived from the MATH train set and used by the simpleRL-reason project for RL fine-tuning of Qwen models.

The model is evaluated on **three math reasoning benchmarks**:

1. **GSM8K** — Grade school math (1,319 test problems). Metric: `val-core/openai/gsm8k/acc/mean@1`.
2. **MATH-500** — Curated 500-problem subset of MATH competition problems. Metric: `val-core/HuggingFaceH4/MATH-500/acc/mean@1`.
3. **AMC 2022-2023** — 30 problems from the American Mathematics Competitions (AMC 12 2022-2023, 83 problems). Metric: `val-core/amc23/acc/mean@1`.

The primary score is the **mean accuracy across the three benchmarks**. Training runs for **100 PPO steps** with **16 rollout samples per prompt** (batch size 128), on **1 H200 GPU** per experiment.

## Interface Contract
The training loop calls your function via the verl policy-loss registry:

```python
@register_policy_loss("custom")
def compute_custom_policy_loss(
    old_log_prob: torch.Tensor,      # (bs, response_length)
    log_prob: torch.Tensor,          # (bs, response_length)
    advantages: torch.Tensor,        # (bs, response_length)
    response_mask: torch.Tensor,     # (bs, response_length)
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
```

**Parameters:**
- `old_log_prob`: per-token log-prob under the rollout (old) policy.
- `log_prob`: per-token log-prob under the current policy being updated.
- `advantages`: per-token advantage (already computed by the GRPO advantage estimator and broadcast across tokens).
- `response_mask`: 1 for valid response tokens, 0 for padding/prompt.
- `loss_agg_mode`: aggregation mode forwarded to `agg_loss`. Common values: `"token-mean"`, `"seq-mean-token-mean"`.
- `config`: `ActorConfig` dataclass. Relevant fields:
  - `config.clip_ratio` — primary clip parameter ε (e.g. 0.2).
  - `config.clip_ratio_low`, `config.clip_ratio_high` — optional asymmetric clips; fall back to `clip_ratio` if `None`.
  - `config.get("clip_ratio_c", default)` — used by dual-clip variants.
  - `config.global_batch_info` — dict of kwargs to pass through to `agg_loss`.
- `rollout_is_weights`: optional per-token rollout-correction weight (multiplicative on `pg_losses`); may be `None`.

**Return values:**
- `pg_loss`: scalar tensor (the aggregated policy loss).
- `metrics`: dict with at minimum `"actor/pg_clipfrac"` and `"actor/ppo_kl"` as Python floats.

**Canonical aggregation (copy from vanilla):**
```python
pg_loss = agg_loss(
    loss_mat=pg_losses,
    loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    **config.global_batch_info,
)
```

**Available utilities:**
- `verl_F.masked_mean(values, mask, dim=None)` — masked mean.
- `verl_F.masked_whiten(values, mask)` — masked whitening.
- `agg_loss` — final loss aggregation (imported at module top).
- `torch`, standard ops for indexing / `clamp` / `exp` / `detach`.

**Important notes:**
- `assert config is not None` and access `config.clip_ratio` — do NOT hardcode ε.
- Remember to clamp `log_prob - old_log_prob` to a safe range (e.g. `[-20, 20]`) before `exp` for numerical stability, as the reference implementations do.
- If your strategy aggregates across the sequence (e.g. GSPO), you may want to use `loss_agg_mode="seq-mean-token-mean"` inside your `agg_loss` call regardless of the input `loss_agg_mode`.
- Apply `rollout_is_weights` multiplicatively on `pg_losses` if it is not None (see vanilla for the pattern).

## Baselines
Three reference baselines are provided covering the primary granularities discussed in the literature:

| Baseline | Granularity | Reference |
| --- | --- | --- |
| `token_level` | Per-token ratio, per-token clip (vanilla PPO / GRPO) | Schulman et al. 2017 (PPO) |
| `sequence_level` | Single sequence ratio = `exp(mean_t(log_prob_new − log_prob_old))`, broadcast to all tokens, then clipped | GSPO, Zheng et al. 2025 (arXiv:2507.18071) |
| `first_k_tokens` | Per-token ratio for the first K=64 response tokens; ratio is detached (stop-grad) for later tokens to reduce variance at the cost of some bias | DAPO-style truncated IS (arXiv:2503.14476) |

# LLM Online RL: KL-Divergence Estimator for Actor KL Loss

## Objective
Design and implement a custom KL-divergence estimator for the actor-side
KL-loss term used during LLM reinforcement learning (RL) training.  Your
code goes in the `compute_custom_kl_penalty()` function in
`custom_kl_penalty.py`.  The read-only reference file `core_algos.py`
contains verl's built-in estimators (`k1`/`kl`, `k2`/`mse`, `k3`/`low_var_kl`,
`abs`, plus straight-through `k3+` gradient variants) that you can study.

## Background
When training LLM policies with GRPO / PPO-style objectives, a KL penalty
keeps the policy close to a frozen reference model (or the supervised fine-
tuned initialization).  In verl the KL penalty enters the objective in two
possible places:

1. **KL-in-reward** (`algorithm.use_kl_in_reward=True`): per-token KL is
   subtracted from the reward before advantage estimation.
2. **KL-loss** (`actor.use_kl_loss=True`): per-token KL is aggregated and
   added to the policy-gradient loss with coefficient `kl_loss_coef`.
   **This is the path this task studies.**

Because the policy and reference distributions are over large vocabularies,
the full KL is expensive to compute; practitioners use Monte-Carlo
estimators from per-token log-probabilities.  The main options
(Schulman, "Approximating KL divergence", 2020, http://joschu.net/blog/kl-approx.html):

| Name | Formula | Bias | Variance |
|---|---|---|---|
| `k1` (`kl`) | `log p - log q` | unbiased | high |
| `k2` (`mse`) | `0.5 * (log p - log q)^2` | biased (overestimates) | low |
| `k3` (`low_var_kl`) | `exp(log q - log p) - (log q - log p) - 1` | unbiased | low, always `>=0` |
| `abs` | `|log p - log q|` | biased | medium, robust |

Each has different gradient properties — `k2` has exact-KL gradients in
expectation; `k1`/`k3` have exact-KL values in expectation but biased
gradients.  verl also exposes straight-through variants (`k3+`) that
combine `k3` forward values with `k2` backward gradients.  Which
choice is best for actually converging an LLM policy under GRPO is
empirical.

## Evaluation
Your KL estimator is used to train **Qwen2.5-0.5B** (full-parameter)
using the verl framework with GRPO advantage estimation.  The only axis
changed across baselines/agents is `compute_custom_kl_penalty()`;
everything else (learning rate, batch size, rollout n, KL coefficient)
is held constant.

Training data is **simpleRL-Zoo MATH level-3-5** plus a 5K subset of
**DeepMath**.  The model is evaluated on three math reasoning benchmarks:

1. **GSM8K** — Grade-school math (1,319 test problems). Metric: `val-core/openai/gsm8k/acc/mean@1`.
2. **MATH-500** — 500 competition-level problems. Metric: `val-core/HuggingFaceH4/MATH-500/acc/mean@1`.
3. **AMC 2022-2023** — 83 competition problems. Metric: `val-core/amc23/acc/mean@1`.

Training runs for **100 steps** with **16 rollout samples per prompt**
(batch size 128), `actor.use_kl_loss=True`, `actor.kl_loss_coef=0.001`,
`actor.kl_loss_type=custom` (dispatches to your function).

## Reference baselines

| Baseline | Formula | Notes |
|---|---|---|
| `k1` | `logprob - ref_logprob` | naive unbiased |
| `k2` | `0.5 * (logprob - ref_logprob) ** 2` | Schulman mse |
| `k3` | `exp(r - l) - (r - l) - 1` (with clamps) | verl default `low_var_kl` |
| `abs` | `|logprob - ref_logprob|` | robust |

## Interface Contract

The training loop calls your function from `dp_actor.py`'s
`compute_kl_loss` path (gated by `actor.use_kl_loss=True`):

```python
def compute_custom_kl_penalty(
    logprob: torch.Tensor,      # (bs, response_length)
    ref_logprob: torch.Tensor,  # (bs, response_length)
) -> torch.Tensor:              # (bs, response_length), per-token KL
```

**Parameters:**
- `logprob`: Per-token log-probs under the *current* actor policy.
- `ref_logprob`: Per-token log-probs under the *frozen reference* policy.

**Return:**
- `kl_estimate`: per-token KL estimate, same shape as inputs.  The
  training loop multiplies this by `response_mask` and aggregates via
  `agg_loss(...)` before multiplying by `kl_loss_coef` and adding to the
  policy-gradient loss.

**How it is wired up:**
Your module is imported by `dp_actor.py` at startup.  At import time it
monkey-patches `verl.trainer.ppo.core_algos.kl_penalty_forward` so that
the string `"custom"` (from `actor.kl_loss_type=custom` on the command
line) dispatches into `compute_custom_kl_penalty()`.  All other
`kl_loss_type` strings continue to route through verl's original
implementation.

**Notes:**
- No need to wrap your computation in `torch.no_grad()` — the KL loss
  must carry gradients back to the actor.
- The returned tensor should have the same shape as `logprob`/`ref_logprob`.
- Numerical safety (clamping extreme log-ratios) is recommended for
  estimators that exponentiate, as `k3` demonstrates.
- Do not touch the wiring section at the bottom of the file or the
  import / monkey-patch will break.

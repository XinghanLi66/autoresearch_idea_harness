# LLM Online RL: Advantage Estimation for GRPO-Style Training

## Objective
Design and implement a custom advantage estimator for LLM reinforcement learning training. Your code goes in the `compute_custom_advantage()` function in `custom_advantage.py`. The read-only reference file `core_algos.py` contains 13 built-in estimators (GRPO, RLOO, REINFORCE++, REINFORCE++-baseline, OPO, REMAX, GPG, etc.) that you can study.

## Background
In LLM RL (e.g., RLHF, GRPO), advantage estimation determines how each response is weighted during policy optimization. Given a batch of prompt–response pairs with scalar rewards, the advantage estimator computes per-token advantage values that the PPO loss uses for gradient updates. Key design choices include:

- **Group normalization (GRPO)**: Multiple responses are sampled per prompt. Advantages are computed as `(reward - group_mean) / (group_std + eps)`, making optimization invariant to reward scale. Dr.GRPO omits the std normalization.
- **Leave-one-out baseline (RLOO)**: For each response, the baseline is the mean reward of all *other* responses in the group, reducing variance via `r_i - mean(r_{j≠i})`.
- **REINFORCE++ family**: Token-level discounted returns (REINFORCE++) or group-centered rewards with batch-level token whitening (REINFORCE++-baseline, https://arxiv.org/abs/2501.03262).
- **Outcome-level vs token-level**: Most estimators broadcast a single per-sequence advantage to all tokens. Token-level methods assign different advantages to different positions.

## Evaluation
Your advantage estimator is used to train **Qwen2.5-0.5B** (full parameter training) using the verl framework. Training uses **simpleRL-Zoo MATH level-3-5 (Qwen split)** — a curated ~8K-problem training set derived from the MATH train set and used by the simpleRL-reason project for RL fine-tuning of Qwen models.

The model is evaluated on **three math reasoning benchmarks**:

1. **GSM8K** — Grade school math (1,319 test problems). Metric: `val-core/openai/gsm8k/acc/mean@1`.
2. **MATH-500** — Curated 500-problem subset of MATH competition problems, a standard eval benchmark. Metric: `val-core/HuggingFaceH4/MATH-500/acc/mean@1`.
3. **AMC 2022-2023** — 30 problems from the American Mathematics Competitions (AMC 12 2022-2023, 83 problems) (very hard competition math). Metric: `val-core/amc23/acc/mean@1`.

Training runs for **100 steps** with **16 rollout samples per prompt** (batch size 128). Each experiment uses 1 H200 GPU. Higher accuracy indicates the model solves more math problems correctly.

## Reference baselines

| Baseline | Axis tested |
|---|---|
| `grpo` | group mean + group std (std-normalized) |
| `dr_grpo` | group mean, no std (https://arxiv.org/abs/2503.20783) |
| `reinforce_plus_plus_baseline` | group mean + token-level batch whitening (https://arxiv.org/abs/2501.03262) |

## Interface Contract
The training loop calls your function via the verl advantage estimator registry:

```python
@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: np.ndarray = None,           # (bs,) group ID per sample
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    old_log_probs: Optional[torch.Tensor] = None,  # (bs, response_length)
    ref_log_probs: Optional[torch.Tensor] = None,  # (bs, response_length)
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:  # (advantages, returns)
```

**Parameters:**
- `token_level_rewards`: Per-token rewards. For outcome-based rewards, the scalar is at the last valid token; use `.sum(dim=-1)` for per-sequence scores.
- `response_mask`: Binary mask (1 = valid token). Shape `(bs, response_length)`.
- `index`: Group/prompt identifier. Samples with the same index were generated from the same prompt (16 per prompt by default).
- `epsilon`: Small constant (1e-6) to avoid division by zero.
- `config`: `AlgoConfig` dataclass with fields `gamma` (discount, default 1.0), `lam` (GAE lambda), `norm_adv_by_std_in_grpo` (bool), etc.
- `old_log_probs`: Per-token log-probabilities under the current rollout policy. Useful for entropy bonuses, importance weighting, or probability-aware advantage shaping.
- `ref_log_probs`: Per-token log-probabilities under the frozen reference policy. Per-token KL divergence can be approximated as `old_log_probs - ref_log_probs`.

**Return values:**
- `advantages`: `(bs, response_length)` — per-token advantage estimates, masked by `response_mask`.
- `returns`: `(bs, response_length)` — per-token return estimates, masked by `response_mask`.

**Available utilities:**
- `verl_F.masked_whiten(values, mask)` — zero-mean unit-variance whitening over masked elements
- `verl_F.masked_mean(values, mask)` — mean over masked elements
- `defaultdict` from collections (already imported)
- `torch`, `numpy` (already imported)

**Important notes:**
- Computation should be wrapped in `torch.no_grad()`.
- For outcome-level estimators, broadcast the per-sequence advantage to all tokens: `scores.unsqueeze(-1) * response_mask`.
- The `index` array groups responses by prompt. Use `defaultdict(list)` to collect per-group scores.

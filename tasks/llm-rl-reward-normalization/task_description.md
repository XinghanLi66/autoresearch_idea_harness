# LLM Online RL: Reward Normalization Before Advantage Estimation

## Objective
Design and implement a custom **reward-normalization** strategy that runs BEFORE advantage estimation in LLM reinforcement learning. Your code goes in the `normalize_rewards()` function in `custom_reward_normalization.py`. The read-only reference file `core_algos.py` contains verl's built-in advantage estimators (GRPO, REINFORCE++, Dr.GRPO, RLOO, …) — your normalization runs upstream of all of them, so the advantage estimator you use (GRPO by default) sees your normalized rewards instead of the raw scalars.

## Background
In GRPO-style LLM RL, the reward manager produces a per-response scalar score (e.g. 1.0 for a correct answer, 0.0 otherwise, possibly with a format bonus). This scalar is placed at the last valid token of the response, becoming a `(batch_size, response_length)` tensor `token_level_scores`. Downstream, the advantage estimator (GRPO) subtracts a per-prompt baseline and optionally divides by group std. The quality of that baseline and the stability of the policy gradient critically depend on the **scale and distribution of the rewards going in**.

Key design choices for reward normalization include:

- **Raw / outcome-only**: No normalization — pass the reward through and let the advantage estimator do all the work. This is verl's current default.
- **Batch-std whitening**: Subtract the batch mean and divide by batch std + eps. A classic RLHF baseline (Ouyang et al., 2022).
- **Group-std (GRPO-style)**: Subtract the per-prompt group mean and divide by per-prompt group std. Same statistic GRPO uses downstream, but applied here at the reward stage.
- **Length-aware**: Divide the scalar reward by a function of response length (e.g. √T) before broadcasting. Motivated by the observation in DAPO (Liu et al., 2024, https://arxiv.org/abs/2503.14476) that longer responses accumulate more gradient signal per token, biasing the policy toward verbose outputs.
- **Percentile clipping**: Clip the reward distribution to robust quantiles (e.g. 5th–95th percentile) before normalization to limit the influence of outlier responses.

## Evaluation
Your reward normalization is used to train **Qwen2.5-0.5B** (full parameter training, GRPO advantage estimator, n=16 rollouts per prompt) using the verl framework. Training data is a mix of **simpleRL-Zoo MATH level 3–5 (Qwen split)** and **5K deepmath problems**.

The model is evaluated on **three math reasoning benchmarks**:

1. **GSM8K** — Grade school math (1,319 test problems). Metric: `val-core/openai/gsm8k/acc/mean@1`.
2. **MATH-500** — Curated 500-problem subset of MATH competition problems. Metric: `val-core/HuggingFaceH4/MATH-500/acc/mean@1`.
3. **AMC 2022-2023** — American Mathematics Competitions test set. Metric: `val-core/amc23/acc/mean@1`.

Training runs for **100 steps**, batch size 128 with 16 rollouts per prompt, max response length 16384 tokens, `test_freq=25`, `total_epochs=1`. Higher accuracy = better.

## Reference baselines

| Baseline | Strategy |
|---|---|
| `outcome_only` | No normalization — pass raw reward through (verl default). |
| `group_std` | Per-prompt group mean + group std normalization, applied at reward stage. |
| `batch_std` | Batch-mean + batch-std whitening over valid response tokens (RLHF-style). |
| `length_aware` | Divide the scalar by √(response_length) before broadcast (DAPO length-bias fix). |

## Interface Contract
The training loop calls your function immediately after the reward manager has produced `token_level_scores`, and BEFORE KL-in-reward penalties / advantage computation:

```python
def normalize_rewards(
    token_level_scores: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: np.ndarray = None,           # (bs,) group/prompt identifier
    epsilon: float = 1e-6,
    config: Optional[object] = None,    # algorithm hydra config
    **kwargs,
) -> torch.Tensor:                      # (bs, response_length) normalized rewards
```

**Parameters:**
- `token_level_scores`: Per-token rewards. For outcome-based rewards the scalar lives at the last valid token; use `.sum(dim=-1)` to recover per-sequence scores.
- `response_mask`: Binary mask (1 = valid token).
- `index`: Per-sample group/prompt identifier (GRPO groups of 16). Samples with the same index come from the same prompt.
- `epsilon`: Small constant to avoid division by zero.
- `config`: The full `algorithm` hydra config (DictConfig). Useful for reading new hyperparameters you may add.

**Return value:**
- `token_level_scores`: `(bs, response_length)` — the transformed reward tensor. It is written back into `data.batch["token_level_scores"]` and then consumed by the advantage estimator (GRPO by default).

**Available utilities:**
- `verl_F.masked_whiten(values, mask)` — zero-mean unit-variance whitening over masked elements.
- `verl_F.masked_mean(values, mask)` — masked mean.
- `defaultdict` from collections (already imported).
- `torch`, `numpy` (already imported).

**Important notes:**
- The computation should be wrapped in `torch.no_grad()`.
- You must keep the returned tensor's **shape** equal to the input (`(bs, response_length)`). To preserve the "outcome reward at last token" semantics, multiply by `response_mask` where appropriate.
- Per-prompt grouping: samples sharing the same `index[i]` were generated from the same prompt (16 rollouts per prompt here). Use `defaultdict(list)` to collect per-group statistics.
- This runs **before** the advantage estimator. GRPO will still subtract the group mean and divide by group std on top of whatever you return; account for that when designing your strategy (e.g. if you already whiten batch-wise, GRPO will re-normalize per group).

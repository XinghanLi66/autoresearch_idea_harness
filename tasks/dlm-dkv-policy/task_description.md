# Diffusion LM KV Cache Policy

## Research Question

Design a cache policy for diffusion language-model inference. Given a fixed LLaDA host model and public final-task benchmarks, can a method preserve benchmark accuracy while reusing KV state during denoising?

## Evaluation Setup

The harness runs real `LLaDA-8B-Instruct` inference end to end. For each workload it:

1. Loads the public benchmark dataset.
2. Runs one fixed denoising rollout with a shared cache-plan interface.
3. Generates deterministic outputs using the submitted cache policy.
4. Scores the generated outputs with benchmark-native final-task metrics.
5. Emits the benchmark-native final score.

The old token-trajectory agreement score is not part of the canonical ranking.
The task also does not ask participants to choose a named cache backend. The
paper implementations define and validate the shared cache-control surface; they
are not exposed as a menu of runtime backends.

Some cache mechanisms require additional LLaDA forward arguments such as active
query rows or tracked-token positions. The harness may load task-local
compatibility model classes to expose those forward hooks, but the outer rollout
remains policy-driven and does not call paper repository generation functions.

## Editable Surface

You may edit only the policy class in `dLLM-cache/custom_dlm_eval.py`. The
compatibility class name is `DLMRefreshPolicy`, but semantically it is a DLM
cache-plan policy.

The required hook families are:

| Method | Purpose |
|---|---|
| `block_schedule(request_meta)` | Controls generation length, block length, steps per block, and whether a block starts with a full warm forward. |
| `query_plan(step_meta, mask_state, cache_state)` | Selects token positions to forward or recompute: full sequence, current block, active query rows, tracked tokens, or a masked query window. |
| `cache_refresh_plan(layer_meta, step_meta, token_stats, cache_state)` | Decides per-layer recompute/reuse, prompt-vs-generation refresh, selected row refresh, KV overwrite, and layer reset. |
| `attention_probe_plan(layer_meta, step_meta)` | Requests attention weights or attention-similarity probes and supplies parameters such as rollout fraction, `current_k`, `gamma`, and `track_num`. |
| `token_transfer_plan(logits, mask_state, step_meta)` | Chooses which masked tokens are committed back to the global denoising state. |
| `after_step(step_meta, logits, attention_stats, transfer_state, cache_state)` | Updates state such as active query masks, attention rollout, tracked tokens, density scores, and layer reset boundaries. |

The full hook contract and baseline mapping are recorded in
`CACHE_HOOK_CONTRACT.md`.

## Fixed Components

Participants may not modify:

- the model weights or tokenizer
- benchmark loaders and scorers
- task scripts, parser, score spec, or leaderboard schema
- source-reference snapshots under `third_party/official_dlm_cache_baselines`
- any harness code outside the editable policy region

## Workloads

Visible workloads:

| Label | Workload | Public Source | Final metric |
|---|---|---|---|
| `math` | `math` | MATH-500 test split | exact final-answer accuracy |
| `humaneval` | `humaneval` | OpenAI HumanEval | pass@1 execution accuracy |

Held-out workload:

| Label | Workload | Public Source | Final metric |
|---|---|---|---|
| `lm-eval` | `lm_eval` | ARC-Challenge test split | exact answer-letter accuracy |

All examples in the selected public splits are evaluated by default. Feedback
from held-out workloads is withheld during iteration but remains part of the
canonical final score.

## Metrics

Each script prints one `TEST_METRICS:` line. The parser records the benchmark
score and numeric runtime diagnostics in the leaderboard artifact:

| Metric | Direction | Meaning |
|---|---|---|
| `final_score` | higher | benchmark-native final task score on a 0-100 scale |
| `reuse_ratio` | higher | diagnostic fraction of generated-token cache work reused by the hook plan |
| `refresh_ratio` | lower | diagnostic `1 - reuse_ratio` |
| `tokens_per_s` | higher | diagnostic decode throughput on the current hardware |
| `peak_memory_mb` | lower | diagnostic peak GPU memory allocated during the example loop |
| `n_examples` | fixed | number of examples evaluated |
| `elapsed` | lower | diagnostic wall-clock time recorded by the harness for the script |

`final_score` is the canonical quality metric. `reuse_ratio` and `tokens_per_s`
are included in the scalar ranking because the task is a cache-policy
benchmark: methods should preserve final-task quality while reducing redundant
denoising work and improving decode throughput. `refresh_ratio`,
`peak_memory_mb`, `n_examples`, and `elapsed` remain review diagnostics. Text
tags such as `eval_mode=real_rollout` are emitted in raw logs for traceability
rather than scalar scoring.

## Canonical Ranking

The canonical score in `score_spec.py` follows the MLS-Bench pattern used by
efficiency-oriented mature tasks:

- each workload applies `final_score_*` as a near-lossless soft quality gate
- once the quality gate is satisfied, small benchmark-native score differences
  are not rewarded further
- each workload ranks cache reuse and decode throughput as efficiency terms
- throughput is normalized against the visible baseline envelope rather than a
  hard hardware-specific pass/fail range
- the task score is the geometric mean across `math`, `humaneval`, and `lm-eval`

The quality gates are calibrated from the current upstream-aligned rerun:
`math >= 35.0`, `humaneval >= 40.0`, and `lm-eval >= 84.0`. Below a gate, the
workload score receives an exponential penalty; above it, ranking is driven by
`reuse_ratio` and `tokens_per_s`. These thresholds are not lowered to rescue a
paper baseline that underperforms in the fixed-setting benchmark. In particular,
the current `d2cache` MATH row fails the MATH quality gate and is penalized
rather than treated as quality-preserving.

## Baselines

| Baseline | Paper-backed source | Implementation |
|---|---|---|
| `vanilla_uncached` | no-cache LLaDA control | full denoising forward every step |
| `dllm_cache` | dLLM-Cache, `maomaocun/dLLM-cache` | prompt/generation feature refresh and low-similarity generated-row update |
| `d2cache` | d2Cache, `Kamichanw/d2Cache` | active query mask, eager attention rollout, and certainty-density top-up |
| `elastic_cache` | Elastic-Cache, `VILA-Lab/Elastic-Cache` | tracked-token query window and attention-similarity layer reset |

The task-local official source snapshots and commit hashes are documented in `third_party/official_dlm_cache_baselines/NOTICE.md`.

## Reproduction Boundary

The selected paper baselines are included because their official repositories
identify concrete cache mechanisms that can be represented on the shared
rollout hooks. Baseline edits must implement those mechanisms on the common
policy surface rather than directly selecting or calling a paper repository as a
black-box backend.

The leaderboard is a fixed-setting benchmark, not a per-paper retuning suite.
To avoid rewarding task-specific hyperparameter search, each baseline uses one
predeclared cache policy across `math`, `humaneval`, and `lm-eval`. This means a
paper baseline can be source-backed and mechanism-faithful while still differing
from a paper table that tuned decoding details, prompt protocol, or metric
postprocessing for a single benchmark. For example, the d2Cache row uses the
published active-query / attention-rollout / certainty-prior cache mechanism
with one shared parameterization across workloads, rather than selecting a
Math-500-specific variant to match the paper's reported Math-500 number.

Small compatibility shims may be used for model loading and source-oracle
checks when a hook requires extra forward arguments. They are capability
adapters, not participant-facing backend choices; canonical task behavior must
still be explained in terms of the shared hook contract.

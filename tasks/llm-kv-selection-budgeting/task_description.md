# llm-kv-selection-budgeting

Design a KV token-retention controller inside a shared full-attention Hugging
Face decoding harness.

## Background

Long-context LLM inference stores key-value (KV) tensors for every attention
layer. KV selection methods reduce this cache by retaining a subset of
historical tokens while preserving enough context for generation quality.

This task evaluates prefill-time KV retention choices through one full-attention
Hugging Face runtime surface. The scaffold runs a standard full-attention
prefill, then applies the policy's token-scoring rule to the generated KV cache
before greedy decoding.

## Task

Modify only the `SelectionPolicy` class in
`transformers-kv-lab/custom_selection_eval.py` (lines 40-101). Implement:

- `retention_plan(layer_id, request_meta, cache_meta)`
- `score_tokens(module, hidden_states, keys, values, kwargs, plan)`
- `select_cache(module, keys, values, scores, n_kept)`

The harness owns the model, datasets, prompt templates, cache budget, decode
loop, and scoring. The editable policy owns the retention metadata and the
per-token scoring rule used to rank prefill KV entries. The shared hook exposes
the union needed by the included source-backed baselines:

- no compression
- attention-sink/recent-window retention
- expected future-attention scoring
- LagKV lag-relative key/value scoring
- optional RoPE-aware key rerotation after pruning

The canonical compression setting is `SELECTION_KV_COMPRESSION_RATIO=0.8`,
meaning methods retain roughly 20% of prefill KV tokens unless the
`full_attention` anchor explicitly disables compression. Final scoring applies
a soft over-budget penalty once `mean_retained_fraction > 0.25`.

## Evaluation

The canonical model is aligned with `llm-kv-adaptive-quantization`:

- `Qwen/Qwen2.5-3B-Instruct`

The canonical workloads follow the same public text-benchmark protocol used by
`llm-kv-adaptive-quantization`. Dataset sources, split names, prompt templates,
generation limits, and scoring semantics must stay aligned across those tasks
whenever the same workload label is used.

The canonical workloads are:

- `longbench_hotpotqa`
  - LongBench-E `hotpotqa_e`
  - final score: upstream LongBench QA F1 on a 0-100 scale
- `longbench_passage_retrieval`
  - LongBench-E `passage_retrieval_en_e`
  - final score: upstream LongBench retrieval score on a 0-100 scale
- `longbench_repobench`
  - LongBench-E `repobench-p_e`
  - final score: upstream LongBench code-similarity score on a 0-100 scale
- `longbench_v2`
  - LongBench v2 `train` split
  - final score: official multiple-choice exact accuracy on a 0-100 scale
  - long inputs are truncated with the official head-tail policy when they
    exceed the model context window
- `gsm8k`
  - `openai/gsm8k` main test split
  - final score: exact final-answer accuracy after numeric normalization on a 0-100 scale
  - hidden workload: results contribute to final scoring, but intermediate feedback is withheld

Canonical data sources are the public upstream datasets listed above:
`THUDM/LongBench`, `THUDM/LongBench-v2` / `zai-org/LongBench-v2`, and
`openai/gsm8k`. Dataset sources, split names, prompt templates, generation
limits, and scoring semantics are intentionally matched to
`llm-kv-adaptive-quantization` whenever a workload label is shared.
The runtime resolves the model and datasets from the local build-time or mounted
Hugging Face cache only; missing assets are hard failures rather than implicit
network downloads.

`SELECTION_KV_MAX_EXAMPLES=0` means full available workload. Setting it to a
positive integer is allowed for local smoke validation but should not be used
for final leaderboard evidence.

## Baselines

The visible baselines are source-backed shared-hook implementations, not claims
of using each paper's original runtime or custom kernel path. Each baseline
cites the original paper as the algorithm's source of truth and identifies the
canonical reference implementation it mirrors:

- `full_attention`
  - uncompressed full-cache reference anchor (HuggingFace `DynamicCache`).
- `streamingllm`
  - Xiao et al., ICLR 2024 (arXiv:2309.17453), "Efficient Streaming Language
    Models with Attention Sinks". Section 3.2 + Table 2 defaults:
    `sink_tokens = 4`, recent-window inferred from the budget. RoPE-aware key
    rerotation following the paper's "Rolling KV Cache with Attention Sinks"
    formulation; reference re-rotation routine mirrors NVIDIA/kvpress
    `KeyRerotationPress` (audit commit 0.3.0). Original code:
    github.com/mit-han-lab/streaming-llm.
- `expected_attention`
  - Devoto, Jeblick, Jegou (NVIDIA), 2025 (arXiv:2510.00636), "Expected
    Attention: KV Cache Compression by Estimating Attention from Future Queries
    Distribution". Equations 4-7 for the score; defaults from the paper's
    experimental section (matched in the canonical NVIDIA/kvpress
    `ExpectedAttentionPress`, audit 0.3.0): `n_future_positions=512`,
    `n_sink=4`, covariance enabled, value-norm rescaling enabled. The paper
    authors are NVIDIA-affiliated; kvpress is the official reference impl.
- `lagkv`
  - Liang et al., 2025 (arXiv:2504.04704), "LagKV: Lag-Relative Information
    for KV Cache Compression". Algorithm 1 (Section 3.2) defines the
    lag-relative score; defaults from the paper Section 4.1 / Table 1 (mirrored
    in NVIDIA/kvpress `LagKVPress`, audit 0.3.0): `n_sink=4`, `lag_size=128`,
    `cross_scoring=False`.

### Expected baseline ranking (Qwen2.5-3B-Instruct, retained ≈ 0.20)

The current leaderboard rows establish the calibration envelope used by
`score_spec.py`. Cross-workload rankings inform what a healthy submission
should look like:

| Workload | Quality (0-100) leader | Trailing baseline | Notes |
|---|---|---|---|
| LongBench HotpotQA | full_attention 37.1 | streamingllm 25.6 | expected_attention 33.4 ≈ lagkv 31.6 |
| LongBench Passage Retrieval | full_attention 62.4 | streamingllm 53.1 | lagkv 60.4 > expected_attention 50.3 |
| LongBench RepoBench | full_attention 47.6 | lagkv 40.9 | expected_attention 47.3 ≈ streamingllm 43.2 |
| LongBench v2 | full_attention ≈ all 29.0–29.6 | (saturated; weak signal at this scale) | LongBench-v2 is hard for 3B; expect tight cluster |
| GSM8K (hidden) | full_attention 31.8 | streamingllm 1.7, lagkv 2.0 | huge gap — chain-of-thought tokens are critical, retention destroys reasoning |

Headlines:
- **Aggressive selection (20% retention) costs ~10-30% accuracy** on most
  long-context QA workloads at this model scale.
- **GSM8K accuracy collapses** (~30→2) when reasoning tokens are evicted; this
  is the most discriminative signal for "selection-aware reasoning preservation".
- **expected_attention and lagkv** generally outperform `streamingllm` on QA
  workloads because they score by content rather than recency, but the gap is
  workload-dependent.
- **`full_attention` is best on quality** by construction but fails the
  `mean_retained_fraction ≤ 0.25` budget constraint; its reduction term and
  the soft penalty exclude it from being a valid submission.

A submission should: (a) keep `mean_retained_fraction ≤ 0.20` (matching the
canonical `SELECTION_KV_COMPRESSION_RATIO=0.8`), (b) **beat or match
expected_attention/lagkv on quality** on at least three of the five workloads,
and (c) ideally narrow the GSM8K gap, since GSM8K is the most algorithmically
informative workload.

### Harness enforcement vs. advisory metadata

The `retention_plan(...)` dict serves as the policy's internal communication
channel between `retention_plan` and `score_tokens`. The harness enforces only
what is observable from the post-`select_cache` cache state:

| Field | Status | Notes |
|---|---|---|
| `compression_ratio` | **enforced** | Harness force-overrides to its own value at the call site (`PrefillSelectionCompressor.forward_hook`). Policies cannot lie about the budget. |
| `mean_retained_fraction` | **measured, enforced** | Computed from `select_cache`'s actual output `n_kept / keys.shape[2]` per layer, then averaged. Drives the soft budget penalty in `score_spec.py`. |
| `disable_compression` | **enforced** | If `True`, harness skips `score_tokens`/`select_cache` entirely and reports `retained = 1.0`. Used by the `full_attention` anchor. |
| `method` | **logged only** | Recorded for provenance; not used in scoring. |
| `sink_tokens`, `lag_size`, `n_future_positions`, `subspace_dim`, etc. | **advisory** | Used internally by the policy's own `score_tokens`. The harness does not verify that declared "sinks" are actually preserved by `select_cache`'s top-K output. Honesty here only matters for provenance and ablation reproducibility, not for scoring. |

This separation is intentional: the task's research question is "given a fixed
retained budget, how good a quality/speed tradeoff can the selection policy
achieve". Final scoring depends only on the end-to-end measured signals
(`final_score`, `mean_retained_fraction`, `runtime_seconds`) — not on the
policy's self-reported methodology.

## Metrics

The parser expects one `TEST_METRICS:` line per workload with:

- `final_score`
  - benchmark-native final task score on a 0-100 scale
- `mean_retained_fraction`
  - average retained prefill KV fraction after the policy runs
- `runtime_seconds`
  - workload wall-clock runtime in seconds

## Canonical Ranking

The leaderboard uses a single scalar computed from accuracy, runtime, and cache
reduction under the fixed retained-fraction constraint:

- LongBench HotpotQA
- LongBench Passage Retrieval
- LongBench RepoBench
- LongBench v2
- GSM8K

Each workload combines three normalized terms with weights
`accuracy:time:reduction = 6:2:2`:

- `accuracy_score` is a bounded 0-100 quality normalization calibrated against
  the visible baseline envelope
- `time_score` is a soft lower-is-better sigmoid normalization of
  `runtime_seconds`, calibrated from the visible baseline runtime envelope
- `reduction_score` is a bounded lower-is-better normalization of
  `mean_retained_fraction`

The per-workload score is the weighted mean of those three terms, and the task
score is the geometric mean across workloads. Rows whose
`mean_retained_fraction_*` exceeds the fixed budget tolerance receive a soft
upper-bound penalty, so the `full_attention` row remains a visible reference
anchor rather than a valid compressed-cache submission.

This follows the common MLS-Bench pattern used by mature optimization tasks:
benchmark parsers emit final task metrics, and `score_spec.py` combines quality
and efficiency terms with explicit weights.

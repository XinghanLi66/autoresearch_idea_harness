# LLM KV Cache: Adaptive Quantization Policy

## Research Question

Design an adaptive 4-bit KV-cache quantizer for decoder-only LLM inference on top of a tensor-level `Transformers` replay harness. The task asks whether an algorithm can preserve benchmark output quality while reducing the effective KV footprint through bit allocation, axis selection, residual windows, and optional prefill-time observation.

## What You Can Modify

The editable region is the `AdaptiveKVQuantizer` class in `custom_quant_eval.py`.
The fixed harness supplies real KV-cache tensors and calls the editable class:

- `reset_request(request_meta, budget_state)` before each example
- `needs_prefill_qkv_observer() -> bool`
- `query_observation_position() -> str`
- `observe_prefill_qkv(layer_id, query_states, key_states, value_states, attention_meta)`
- `quantize_key(layer_id, key_states, cache_meta) -> tensor | (tensor, avg_bits)`
- `quantize_value(layer_id, value_states, cache_meta) -> tensor | (tensor, avg_bits)`
- `estimate_bits(layer_id, kv_kind, seq_len, head_dim, cache_meta) -> float`

`key_states` and `value_states` have shape `[batch, heads, seq_len, head_dim]`.
The editable class must implement the actual tensor algorithm: grouping,
asymmetric ranges, zero-points, per-layer bit presets, residual retention,
query-subspace transforms, and memory accounting belong inside this class. The
task no longer exposes a fixed algorithm enum or a backend selector.

## What You Cannot Modify

- The model family and deterministic decode replay loop
- The benchmark workload definitions
- The parser or final-score definitions
- The underlying `Transformers` model implementation

## Evaluation

The canonical visible model is:

- `Qwen/Qwen2.5-3B-Instruct`

The visible workloads use the same public text-benchmark protocol as
`llm-kv-selection-budgeting`. Dataset sources, split names, prompt templates,
generation limits, and scoring semantics must stay aligned across those tasks
whenever the same workload label is used:

- `longbench_hotpotqa`
  - LongBench-E `hotpotqa_e`
  - final score: LongBench QA F1 on a 0-100 scale
- `longbench_passage_retrieval`
  - LongBench-E `passage_retrieval_en_e`
  - final score: LongBench retrieval score on a 0-100 scale
- `longbench_repobench`
  - LongBench-E `repobench-p_e`
  - final score: LongBench code-similarity score on a 0-100 scale
- `needlebench_niah`
  - RULER/NeedleBench-style needle-in-a-haystack retrieval over public essay text
  - needle: `The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.`
  - final score: exact phrase retrieval accuracy on a 0-100 scale

The held-out generalization workload follows the same final-answer scoring
contract but is not part of the visible iteration set:

- `gsm8k`
  - full `openai/gsm8k` main test split
  - final score: exact final-answer accuracy after numeric normalization on a 0-100 scale

The parser expects one `TEST_METRICS:` line per workload with:

- `final_score`: benchmark-native quality on a 0-100 scale
- `effective_kv_bits`: quantizer-level effective KV bits per cached element
- `kv_compression_ratio`: `16 / effective_kv_bits`, using FP16 KV as the reference footprint
- `runtime_seconds`: task-level wall-clock runtime for the workload command

`effective_kv_bits` is computed from the submitted quantizer at a 4096-token reference KV span so the efficiency term is hardware-independent and does not depend on the evaluator GPU model.

## Baselines

Canonical baselines are restricted to paper-linked open-source configurations or adaptations that fit the current tensor-replay surface:

- `kivi_overlap_4bit`
  - KIVI-style K4/V4 adaptation implemented directly in `AdaptiveKVQuantizer`, with key per-channel, value per-token, group size `32`, key `block_modulo` residual blocks, and value tail residual length `128`
- `kvtuner4_pertoken_qwen25_3b`
  - KVTuner's official `Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml` preset implemented with its signed-asymmetric vanilla cache formula, axis `0/0`, `q_group_size=-1`, and `residual_length=0`
- `kvtuner4_kivi_qwen25_3b`
  - KVTuner's official `Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml` preset implemented with the same signed-asymmetric formula, axis `1/0`, `block_modulo` residual length `32`, and `q_group_size=32`
- `squat_subspace_4bit`
  - SQuat's official LongBench-style 4-bit configuration implemented with
    query SVD during prefill, future-dimension key correction, subspace
    dimension `60`, `squat_lambda=0.001`, `quant_group_size=64`,
    `shared_svd=True`, K/V group size `32`, and residual block length `32`

## Canonical Ranking

The leaderboard uses the same pattern as mature MLS-Bench text tasks:

- each workload emits a benchmark-native `final_score_*` quality column
- each workload also emits a hardware-independent `kv_compression_ratio_*` efficiency column
- quality uses the repository standard bounded-power normalization with the
  worst current baseline as the floor, `100` as the bound, and the best current
  baseline as the reference point
- efficiency uses bounded-power normalization on `kv_compression_ratio` with
  the worst current baseline as the floor, `4x` compression as the reference,
  and `8x` compression as the bound
- `runtime_seconds_*` remains an emitted diagnostic column but is not part of
  the final score, because this tensor-replay harness is not a runtime-native
  packed-cache speed benchmark
- each workload score is a weighted mean with quality weight `6` and KV efficiency weight `4`
- the task score is the geometric mean across the three LongBench-E columns,
  NeedleBench/RULER NIAH, and GSM8K workload scores

## Notes

- The harness runs deterministic greedy generation over `Transformers` decode steps and scores the generated answers.
- At each decode step after prefill, it snapshots real KV tensors, quantizes them with the current quantizer, restores the quantized cache, and advances generation.

# Codex Audit Fixes

- `tasks/mlsys-sparse-attention-inference/edits/dense.edit.py`: dense oracle now reports true full-attention density `1.0`.
- `tasks/mlsys-kv-cache-eviction/edits/dense.edit.py`: dense no-eviction oracle now reports true full-cache density `1.0`.
- `tasks/llm-qat-algorithm/task_description.md`: documentation now matches the runtime defaults: 500 steps, warmup 50, seqlen 1024, and about 900s per bit-width.
- `tasks/llm-pretrain-sparse-attention/edits/nsa_full.edit.py`: NSA full selected branch `TOP_K` is set to 4 for the 16-block, 1024-token context.
- `tasks/llm-pretrain-sparse-attention/edits/moba.edit.py`: MoBA forward pads non-block-multiple sequences and trims outputs back to the original length.
- `tasks/llm-pretrain-sparse-attention/edits/reformer.edit.py`: Reformer forward pads non-chunk-multiple sequences and trims outputs back to the original length.
- `.saves/llm-pretrain-sparse-attention/moba/seed_42/model_source_gpt-345m.py`: existing MoBA checkpoint source has the same pad-then-trim short-sequence fix.
- `.saves/llm-pretrain-sparse-attention/reformer/seed_42/model_source_gpt-345m.py`: existing Reformer checkpoint source has the same pad-then-trim short-sequence fix.
- `tasks/mlsys-kv-cache-eviction/edits/mid_edit.py`: comments now clarify that HF `DynamicCache` is full-size bookkeeping only and computation uses the physically compressed module-local `_kv_past_k/_kv_past_v` cache.

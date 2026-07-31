## Overview

This report analyzes the character-length distribution of abstracts in the proposal_rl training dataset (runs/dataset/train.jsonl), covering both the **target paper abstracts** (the ground-truth used as reward signal) and the **reference paper abstracts** (the conditioning input fed to the model via prompt_builder.py).

The current truncation threshold in _ref_entry() is **400 characters** (default abstract_chars).

---

## Target Paper Abstracts

These are the abstracts of the papers being proposed — used as ground truth for PRS/PPL rewards.

| Metric | Value |
| --- | --- |
| Records | 7,357 |
| Missing / empty | 0 |
| Min | 259 chars |
| Max | 1,920 chars |
| Mean | 1,319 chars |
| Median | 1,318 chars |
| Std dev | 302 chars |
| p5 | 827 chars |
| p25 | 1,104 chars |
| p75 | 1,536 chars |
| p95 | 1,831 chars |
| **≤ 400 chars (not truncated)** | **7 records (0.1%)** |
| > **400 chars (truncated)** | **7,350 records (99.9%)** |

---

## Reference Paper Abstracts

These are abstracts of the cited papers passed to the model as conditioning context. They are truncated by _ref_entry() before being included in the prompt.

| Metric | Value |
| --- | --- |
| Total refs | 185,754 |
| Missing / empty | 0 (0.0%) |
| Min (non-zero) | 1 char |
| Max | 10,000 chars |
| Mean | 1,248 chars |
| Median | 1,238 chars |
| p5 | 729 chars |
| p95 | 1,818 chars |
| **≤ 400 chars (not truncated)** | **641 refs (0.3%)** |
| > **400 chars (truncated)** | **185,113 refs (99.7%)** |

---

## Key Finding

<redoc-highlight emoji="gantanhao" fillColor="orange">
**99.7–99.9% of all abstracts exceed the 400-char truncation threshold.** The median abstract is ~1,300 chars, meaning the current default discards roughly **70% of each abstract's content** before it reaches the model.
</redoc-highlight>

The 400-char cutoff was likely chosen to keep prompt length manageable. Given that max_prompt_length is 3,072–4,096 tokens and each record has up to 40 refs, raising abstract_chars requires careful budgeting. At ~4 chars/token, 400 chars ≈ 100 tokens per ref; raising to 800 chars ≈ 200 tokens per ref × 40 refs = 8,000 tokens — already over budget for most strategies.

A targeted increase (e.g. 600 chars for top_k_refs with k=5, where total ref tokens stay manageable) may improve signal quality without hitting prompt length limits.
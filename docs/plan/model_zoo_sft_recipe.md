# Model-zoo SFT recipe (V3 Phase 1) — ready to run

**Status (2026-07-17):** dataset ✓, trainer scripts ✓, recipe ✓. **Only blocker = base-model weights**
(Qwen3-* / DeepSeek not local; HF is out → acquire via **ModelScope** `Qwen/Qwen3-*` or the **QS model
registry** used for Qwen2.5). Fail-fast order: **S1 (8B) first** to validate the Qwen3 recipe, then scale.

## Fixed inputs (all arms)
- **Data:** `autoresearch_idea_harness/runs/training_data/v3_researcher_cot_anchored/{train,val}.jsonl`
  (1,626 / 85; system = researcher-conditioned "You are <name>…", assistant carries **Mocking** +
  **Core idea/Non-trivial crux** anchors). Do NOT swap in the prejudge set.
- **Trainers:** full-FT = `scripts/train_v3_researcher_cot_full_fsdp.py`; LoRA =
  `scripts/train_v3_researcher_cot_lora.py` (r64/α128 all-linear). QS prep =
  `scripts/prepare_v3_qs_researcher_full_sft_run.py` / `prepare_v3_qs_researcher_lora_train_run.py`.
- **‼ Qwen3 chat template:** train/serve **non-thinking** — pass `enable_thinking=False` to
  `apply_chat_template` (Qwen3 dense 8/14/32B default to hybrid-thinking and will otherwise inject
  `<think>` that collides with our anchors). Verify the trainer's template call before S1.
- **DeepSeek:** reasoning-native `<think>` → reconcile format (strip think, keep our anchors).

## Arms (size × series × structure)
| ID | Model | Method | Workers | Epochs | lr | Notes |
|----|-------|--------|---------|--------|----|-------|
| S1 | Qwen3-8B | full-FT | 1 | 1–2 | 5e-6 | **run first** (recipe check: non-thinking + anchors) |
| S2 | Qwen3-14B | full-FT | 1 | 1 | 3e-6 | |
| S3 | Qwen3-32B | full-FT | 1 (tight) | 1 | 2e-6 | clean A/B vs Qwen2.5-32B; 2ep OOMs (fp32 AdamW) → 1ep or 8-bit optim; post-eval `empty_cache` already patched |
| M1 | Qwen3-30B-A3B | LoRA | 1 | 2–3 | 1e-4 | MoE structure axis |
| M2 | Qwen3-235B-A22B | LoRA | 1–2 | 2–3 | 5e-5 | ~470 GB weights |
| D1 | DeepSeek (latest trainable distill, or V4-Flash LoRA) | FT/LoRA | 1 / 2–4 | 1–3 | — | series axis; confirm tag |

Common: seq_len 1664, eff batch 16 (per-dev 1 × accum 4 × 4 GPU), bf16 + activation checkpointing,
warmup 0.03, wd 0.01, cosine. Anchor point: Qwen2.5-32B full-FT = 101 steps / ~17 min on 4×GB200.

## Per-arm checkpoints feed Phase 2 (native mlsbench-lite eval, base vs SFT, multi-seed Δ).

## Validation each arm
Run `scripts/verify_v3_qs_checkpoint.py` (or the local demo) on held-out researchers: expect **format
≈1.0, anchors 3/3, correct names, in-voice**, no `<think>` leakage. Watch val loss (85) + creativity
eval for overfitting — if the larger arms degrade vs 8B/14B, that's the signal to expand/upgrade data
(otherwise the current 1,626 is sufficient — decided 2026-07-17 to run the ladder as-is).

# transformers 5.x port notes — V3 researcher-CoT SFT trainers

Prep work for making the custom FSDP full-FT and LoRA SFT trainers run under
**transformers 5.x** so the 2026-era model zoo (Qwen3.6, DeepSeek-V4-Flash,
GLM-5.2) becomes launchable, **without breaking the 4.x path**.

- **New files (additive, originals untouched):**
  - `scripts/train_v3_researcher_cot_full_fsdp_tf5.py`
  - `scripts/train_v3_researcher_cot_lora_tf5.py`
- **Originals kept as-is:** `train_v3_researcher_cot_full_fsdp.py`,
  `train_v3_researcher_cot_lora.py`.
- **Target image:** `tutu_cybertron:ngc2510` (python 3.12, torch ~2.9, aarch64/GB200).
- **Verified against:** the transformers build actually installed on D0 —
  **`transformers 5.7.0`, `torch 2.11.0+cu130`** (ground truth for every "verified"
  row below; secondary web sources are marked where they diverge from the install).

## 0. TL;DR — what actually breaks

Both original scripts were *already* partly modernised: they use `eval_strategy`
(not `evaluation_strategy`) and `processing_class=` (not `tokenizer=`). Under the
installed **5.7.0** the **only hard failure** is:

```
TrainingArguments.__init__() got an unexpected keyword argument 'save_safetensors'
```

Everything else is either unchanged or a *deprecation warning* (still functional).
The ports therefore make a small, surgical set of changes plus forward-proofing,
all gated by runtime version detection so the same file runs on 4.51+ and 5.x.

Version detection used in both ports:

```python
_TF_VERSION = tuple(int("".join(c for c in p if c.isdigit()) or 0)
                    for p in transformers.__version__.split(".")[:3])
_IS_TF5 = _TF_VERSION[0] >= 5
_TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)  # capability probe
```

Version-gated kwargs are added by **capability probing the installed signature**
(`_TA_PARAMS`) rather than by version number alone, which is the most robust way
to stay dual-compatible.

## 1. TrainingArguments API changes handled

| Old (4.x) | New (5.x) | Changed in | How the port handles it | Status |
|---|---|---|---|---|
| `save_safetensors=True` | **removed** — safetensors is always-on; `safe_serialization=False` dropped from all `save_pretrained`/`push_to_hub` | 5.0 | Only pass `save_safetensors=True` when `"save_safetensors" in _TA_PARAMS` (i.e. on 4.x). On 5.x it is omitted; checkpoints/final save are safetensors regardless. | **verified**: kwarg absent from 5.7.0 signature |
| `evaluation_strategy=` | `eval_strategy=` | deprecated 4.41 (PR #30190), **removed 5.0** | Originals already use `eval_strategy`; ports keep it. | **verified**: `evaluation_strategy` absent, `eval_strategy` present in 5.7.0 |
| `warmup_ratio=X` | `warmup_steps=X` accepts a **float in [0,1)** as a ratio | `warmup_ratio` deprecated (warning says "removed in v5.2", but it is **still present & functional in 5.7.0**) | On 5.x pass `warmup_steps=<ratio>` (float); on 4.x pass `warmup_ratio=<ratio>`. CLI keeps `--warmup-ratio` for back-compat. Semantically identical. | **verified**: `warmup_steps=0.03` stored as `0.03`; `warmup_ratio=0.03` still works but logs a deprecation line |
| `tf32=True` | `tf32=True` (unchanged arg) | — | Kept. Additionally `_configure_tf32()` sets the torch≥2.9 API `torch.backends.cuda.matmul.fp32_precision="tf32"` (+cudnn) in a try/except; falls back to legacy `allow_tf32`. Silences the repeated PyTorch DeprecationWarning that transformers triggers by still toggling `allow_tf32`. | **verified**: `tf32` still in signature; legacy `allow_tf32` deprecated on torch 2.11 |
| `report_to` (auto-detect default) | `report_to` default is now `"none"` | 5.0 | Both scripts already pass `report_to="none"` explicitly, so no behaviour change. | **verified**: default is `'none'` in 5.7.0 |
| `torch_dtype=` (from_pretrained) | `dtype=` preferred; `torch_dtype` still accepted (deprecated) | 5.x | `_DTYPE_KW = "dtype" if "dtype" in from_pretrained-signature else "torch_dtype"`, passed via `**{_DTYPE_KW: torch.bfloat16}`. On 5.7.0 the explicit param is still `torch_dtype`, so that is used; forward-compatible if a later build swaps to `dtype`. | **verified**: 5.7.0 explicit param is `torch_dtype` |

**Unchanged / still valid in 5.7.0** (no action needed — all confirmed present in
the installed signature): `per_device_train_batch_size`, `per_device_eval_batch_size`,
`gradient_accumulation_steps`, `learning_rate`, `lr_scheduler_type`, `max_grad_norm`,
`weight_decay`, `logging_steps`, `save_strategy`, `save_steps`, `save_total_limit`,
`dataloader_num_workers`, `remove_unused_columns`, `gradient_checkpointing`,
`gradient_checkpointing_kwargs`, `save_only_model`, `bf16`, `seed`, `num_train_epochs`,
`max_steps`, `num_train_epochs`, `push_to_hub` (default still `False`).

Other 5.0 removals that don't affect these scripts (but worth knowing if the
recipe grows): `per_gpu_*_batch_size`, `overwrite_output_dir`, `logging_dir`,
`no_cuda`→`use_cpu`, `fp16_backend`/`half_precision_backend`,
`include_inputs_for_metrics`→`include_for_metrics`, `_n_gpu`, `jit_mode_eval`,
`tpu_num_cores`, `past_index`, `ray_scope`. (Source: v5 migration guide.)

## 2. Trainer / callback / train() API

| Item | Status in 5.x | Port handling |
|---|---|---|
| `Trainer(tokenizer=...)` | **removed** (deprecated 4.46) → `processing_class=` | Originals already use `processing_class=`; kept. **verified**: `Trainer.tokenizer` absent, `Trainer.processing_class` present in 5.7.0 |
| `trainer.train(resume_from_checkpoint=...)` | unchanged (old `model_path=` was the removed one) | Kept as-is |
| `trainer.save_model()`, `trainer.evaluate()`, `TrainerCallback.on_evaluate` | no breaking signature change | Kept; the `_EmptyCacheAfterEval` callback is unchanged |
| `transformers.trainer_utils.get_last_checkpoint` | **still present** | Kept import. **verified**: imports fine in 5.7.0 |
| `is_world_process_zero()` | unchanged | Kept |

## 3. FSDP auto-wrap & config plumbing

The 4.x recipe passed a top-level `fsdp="full_shard auto_wrap"` string plus an
`fsdp_config` dict with `fsdp_version: 1`, `transformer_layer_cls_to_wrap:
["Qwen2DecoderLayer"]`, `state_dict_type`, and FSDP1 knobs (`use_orig_params`,
`sync_module_states`, `limit_all_gathers`, `*_prefetch`).

**State in 5.7.0 (verified by building `TrainingArguments`):**
- The `fsdp="full_shard auto_wrap"` **string still works** and produced **no
  deprecation warning** in 5.7.0. (The v5 migration guide says string/list `fsdp=`
  is deprecated for removal in ~v5.20 in favour of `fsdp=True` + `fsdp_config`;
  it is *not yet* removed, so the port keeps the string for minimal churn.)
- `fsdp_config` with `fsdp_version: 1` **builds cleanly** — FSDP1 still supported
  (migration guide: deprecated, removal ~v5.20; **FSDP2 is the new default**).
- `transformer_layer_cls_to_wrap` inside `fsdp_config` still works.
- `auto_wrap_policy: "TRANSFORMER_BASED_WRAP"` is accepted **with or without** an
  explicit `transformer_layer_cls_to_wrap` list; without a list it wraps by the
  model's registered `_no_split_modules`. **verified** both.
- Note: transformers normalises `fsdp_version` → `version` and injects
  `min_num_params`, `xla*` defaults into the stored `fsdp_config`.

**What the ports do** (`build_fsdp_config`):
- New `--fsdp-version {1,2}` flag, **default 1** (matches today's recipe exactly).
- `--fsdp-transformer-layer` still defaults to `Qwen2DecoderLayer`, but now
  accepts **`auto`** (or empty/`none`): that switches to
  `auto_wrap_policy: TRANSFORMER_BASED_WRAP` and drops the explicit class list, so
  novel/hybrid 2026 architectures wrap correctly **without** hand-coding a class
  name. This is the recommended setting for everything past Qwen3.
- FSDP1-only knobs are emitted **only** when `--fsdp-version 1`; under
  `--fsdp-version 2` they are omitted (FSDP2 ignores/rejects several of them).
- `activation_checkpointing` stays inside `fsdp_config` (Trainer
  `gradient_checkpointing=False`), avoiding the "both set → error" trap and the
  redundant all-gather.

### Decoder-layer class names for `transformer_layer_cls_to_wrap`

Confirmed by grepping the **installed transformers 5.7.0** source
(`_no_split_modules` and `class *DecoderLayer` definitions) — these are the exact
strings the FSDP wrapper matches:

| Model | model_type | Decoder-layer class | In core 5.7.0? | Notes |
|---|---|---|---|---|
| Qwen2.5 (today's 32B) | `qwen2` | `Qwen2DecoderLayer` | yes | current default |
| Qwen3 dense | `qwen3` | `Qwen3DecoderLayer` | yes | |
| Qwen3-MoE | `qwen3_moe` | `Qwen3MoeDecoderLayer` | yes | |
| Qwen3-Next | `qwen3_next` | `Qwen3NextDecoderLayer` | yes | hybrid Gated-DeltaNet + attn |
| **Qwen3.6-27B dense** | `qwen3_5` | **`Qwen3_5DecoderLayer`** | **yes** | `_no_split_modules=["Qwen3_5DecoderLayer","Qwen3_5VisionBlock"]`; text uses `Qwen3_5DecoderLayer`. Built on the Qwen3-Next Gated-DeltaNet backbone (`Qwen3_5GatedDeltaNet`). **Verify the model repo's `config.json` `architectures`/`model_type` == `qwen3_5` at launch.** |
| **Qwen3.6-35B-A3B MoE** | `qwen3_5_moe` | **`Qwen3_5MoeDecoderLayer`** | **yes** | `_no_split_modules=["Qwen3_5MoeDecoderLayer","Qwen3_5MoeVisionBlock"]` |
| **GLM-5.2** | `glm_moe_dsa` | **`GlmMoeDsaDecoderLayer`** | **yes** | DeepSeek-Sparse-Attention family; **not** the older `Glm4MoeDecoderLayer` (that's GLM-4.5/4.6). Confirm repo `model_type == glm_moe_dsa`. |
| GLM-4.5/4.6 MoE (fallback) | `glm4_moe` | `Glm4MoeDecoderLayer` | yes | |
| **DeepSeek-V4-Flash** (284B MoE, FP4/FP8, CSA/HCA hybrid) | (n/a) | **not in core 5.7.0** | **NO** | Only `deepseek_v2`/`deepseek_v3` (`DeepseekV3DecoderLayer`) ship in 5.7.0. V4 needs either `--trust-remote-code` custom modeling from the model repo, or a transformers build that adds `deepseek_v4`. **Use `--fsdp-transformer-layer auto`** so wrapping follows the custom module's `_no_split_modules`. |

> **Recommendation:** for every model past Qwen3, launch with
> `--fsdp-transformer-layer auto` rather than hardcoding a class. Hybrid-attention
> models (Qwen3-Next / Qwen3.6 / GLM-5.2 DSA / DeepSeek-V4) have heterogeneous
> layers, and wrapping by `_no_split_modules` is both correct and future-proof.
> The class strings above are given for the record and for FSDP1 explicit-list
> use if desired.

## 4. New / relevant dependencies

- **transformers ≥ 5.0** is what unlocks the new architectures; **5.7.0** is the
  installed/verified version and already contains `qwen3_5`, `qwen3_5_moe`,
  `qwen3_next`, `glm_moe_dsa`. Pin suggestion for the tf5 path:
  `transformers>=5.3,<6` (GLM-5.2 `glm_moe_dsa` reportedly needs ≥5.3; verify).
- **accelerate**: FSDP2 default in v5 leans on a recent accelerate; ensure the
  image's accelerate matches the transformers 5.x it ships. FSDP config can be
  supplied via `fsdp_config` dict (as here) **or** an accelerate YAML — the dict
  path used here needs no accelerate config file.
- **torch ≥ 2.9** (image has ~2.9; D0 has 2.11): required for the newest kernels
  and for the `fp32_precision` TF32 API used in `_configure_tf32()`.
- **Optional kernels** for linear-attention/hybrid models (Qwen3-Next/3.6 Gated-
  DeltaNet): `causal_conv1d` and `fla` (flash-linear-attention) accelerate the
  DeltaNet path; without them it falls back to slower reference code. Not required
  for correctness. Confirm availability on aarch64/GB200.
- **peft**: LoRA path unchanged API-wise; `target_modules="all-linear"`,
  `get_peft_model`, `merge_and_unload` all still valid.
- **kernels / FlashAttention**: `--attn-implementation` is now a CLI flag
  (default `sdpa`). For GB200 you may want `flash_attention_2` if a compatible
  wheel is present; `sdpa` is the safe default.

## 5. Validation plan (8-row smoke, mirrors existing runs)

Run these on a QS trial (queue 532, GB200, 4 GPU/worker) **after** the current
recipe is idle. Do **not** submit as part of this prep. Each step writes a
`train_summary.json` you can diff against a known-good 4.x run.

**Preconditions:** code cloned from GitHub V3 (self-contained; no `/newcpfs` at
runtime); base model + data staged under `/mnt/3fs`; `transformers>=5.3` in the
image.

1. **Import / build smoke (CPU, seconds):** confirms no API drift on the exact
   image before burning GPU:
   ```bash
   python -c "import scripts.train_v3_researcher_cot_full_fsdp_tf5 as m; print(m._TF_VERSION, m._IS_TF5)"
   python -c "import scripts.train_v3_researcher_cot_lora_tf5 as m; print(m._DTYPE_KW)"
   ```
2. **LoRA 8-row smoke (single GPU, Qwen2.5-7B baseline to isolate the port from
   the new architectures):**
   ```bash
   python scripts/train_v3_researcher_cot_lora_tf5.py \
     --train-jsonl <train.jsonl> --limit 8 --max-steps 2 \
     --max-seq-length 1024 --output-dir /mnt/3fs/.../smoke_lora --no-merge --skip-gen-check
   ```
   Expect: `train_summary.json` with `transformers_version` 5.x, non-zero
   `approx_tokens_per_sec`, `Mask check: N/… active` > 0.
3. **Full-FT FSDP 8-row smoke on Qwen2.5-32B (regression parity vs the historical
   4.x run):**
   ```bash
   torchrun --nproc_per_node=4 scripts/train_v3_researcher_cot_full_fsdp_tf5.py \
     --train-jsonl <train> --val-jsonl <val> \
     --base-model /mnt/3fs/.../Qwen2.5-32B-Instruct \
     --output-dir /mnt/3fs/.../smoke_full_qwen25 \
     --fsdp-transformer-layer Qwen2DecoderLayer --fsdp-version 1 \
     --limit 8 --max-steps 1 --max-seq-length 1024
   ```
   Expect: FULL parity behaviour — 1 step, checkpoint written, final save,
   `active_labels_first > 0`, `status: ok`. This validates the port itself.
4. **New-architecture wrap smoke (Qwen3.6-27B dense), FSDP2 + auto-wrap:**
   ```bash
   torchrun --nproc_per_node=4 scripts/train_v3_researcher_cot_full_fsdp_tf5.py \
     --base-model /mnt/3fs/.../Qwen3.6-27B --train-jsonl <train> \
     --fsdp-transformer-layer auto --fsdp-version 2 \
     --fsdp-state-dict-type FULL_STATE_DICT \
     --limit 8 --max-steps 1 --max-seq-length 1024 \
     --output-dir /mnt/3fs/.../smoke_full_qwen36
   ```
   Watch the log line `fsdp_layer=auto`; confirm the model loads (right
   `model_type`), one step runs, and peak memory is sane. If it needs custom
   modeling, add `--trust-remote-code`.
5. **MoE smoke (Qwen3.6-35B-A3B):** as step 4 with `--base-model .../Qwen3.6-35B-A3B`.
   MoE all-gather is heavier — start with `--max-seq-length 1024`, `--per-device-batch-size 1`.
6. **GLM-5.2 smoke:** as step 4 with the GLM-5.2 path; confirm `model_type ==
   glm_moe_dsa` loads on the installed transformers (needs ≥5.3).
7. **DeepSeek-V4-Flash smoke (expected to need extra work):** step 4 +
   `--trust-remote-code`, `--fsdp-transformer-layer auto`. This is the highest-risk
   arm (see §6) — treat a clean single step as a stretch goal, not a gate.
8. **Final-save + reload check:** after step 3, load the FULL_STATE_DICT final
   save with `AutoModelForCausalLM.from_pretrained(<output-dir>)` and run one
   forward pass to confirm the checkpoint is well-formed safetensors.

Gate for "port is good": steps 1–3 green (parity with the 4.x 32B run). Steps 4–8
are architecture-enablement and may surface model-specific issues unrelated to the
Trainer port.

## 6. Open risks

### FP4/FP8-base LoRA (QLoRA-on-FP4) for DeepSeek-V4-Flash — separate open risk
- **4-bit QLoRA (NF4/FP4 via bitsandbytes + peft)** is standard and supported, but
  it *quantizes a bf16/fp16 checkpoint down* to 4-bit at load time; it does **not**
  consume a checkpoint whose weights are **already natively FP4** (NVFP4/QAT-FP4),
  which is how DeepSeek-V4-Flash ships. No documented out-of-the-box peft/
  bitsandbytes path was found for attaching LoRA adapters on top of native-FP4 MoE
  expert weights. **Treat native-FP4 + LoRA as unsupported today without custom
  work.** FP8 LoRA exists only via NVIDIA Transformer Engine (`TeLinear`, no
  `merge()`/DoRA) — hardware/stack specific, not general.
- Practical fallback for V4-Flash LoRA: obtain/dequantize to a bf16 checkpoint and
  run standard QLoRA (NF4), accepting the memory cost — or wait for upstream
  native-FP4 adapter support. Full-FT of a 284B MoE is out of scope for the
  current QS footprint regardless.

### Biggest remaining risk per target model
- **Qwen3.6-27B (dense):** class/`model_type` mapping assumed `qwen3_5` /
  `Qwen3_5DecoderLayer` from the installed source — **verify the released repo's
  `config.json` `architectures`/`model_type`** before trusting it; and the
  Gated-DeltaNet path may need `causal_conv1d`/`fla` kernels on aarch64/GB200
  (falls back to slow reference otherwise). Mitigated by `--fsdp-transformer-layer auto`.
- **Qwen3.6-35B-A3B (MoE):** FSDP sharding of MoE experts — memory spikes on
  all-gather and expert imbalance; correct auto-wrap of the MoE block
  (`Qwen3_5MoeDecoderLayer`) is essential. Start batch=1, short seq; watch peak
  memory in `train_summary.json`.
- **DeepSeek-V4-Flash (284B MoE, FP4/FP8, CSA/HCA hybrid):** **not in core
  transformers 5.7.0** — needs `--trust-remote-code` custom modeling (attention
  kernels dispatched per-layer by `config.layer_types`), and native-FP4 weights
  are incompatible with the current QLoRA/LoRA tooling (above). Highest chance of
  not running out-of-the-box; the single-decoder-class FSDP assumption also breaks
  on its heterogeneous blocks → must use auto-wrap.
- **GLM-5.2:** new `glm_moe_dsa` architecture (DeepSeek-Sparse-Attention +
  IndexShare) reportedly requires **transformers ≥ 5.3** — the biggest risk is an
  image whose transformers is 5.0–5.2 and cannot load the `model_type`. Pin
  `transformers>=5.3` and confirm `GlmMoeDsaForCausalLM` imports before the run.

## 7. Faithfulness checklist (unchanged behaviour vs originals)

- Same CLI surface; only **additive** new flags with back-compatible defaults:
  `--fsdp-version` (default 1), `--fsdp-transformer-layer auto` (default still
  `Qwen2DecoderLayer`), `--trust-remote-code` (default off), `--attn-implementation`
  (default `sdpa`).
- Same output artifacts: `train_summary.json` (now also records
  `transformers_version`/`fsdp_version`), `checkpoint-*` dirs, FULL/SHARDED final
  save via `trainer.save_model`, LoRA `merged/` + `gen_check.txt`.
- Same assistant-turn label masking (`<|im_start|>assistant\n`, `-100` prefix).
- Same sitecustomize-style shims: wandb no-op, `transformers.modeling_layers`
  `GradientCheckpointingLayer`, apex.amp fallback.
- Same FSDP save-mode guard (`save_only_model` incompatible with
  `SHARDED_STATE_DICT`) and the post-eval `empty_cache` callback (full-FT).

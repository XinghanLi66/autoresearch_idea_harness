#!/usr/bin/env python3
"""
Single-GPU LoRA SFT for V3 researcher-CoT distillation.

Uses peft + transformers Trainer (no torchrun/FSDP needed).
Loss is masked manually to the assistant turn via a custom data collator
that identifies the <|im_start|>assistant block in Qwen2.5's chat template.

Usage (smoke test - 20 steps):
    python scripts/train_v3_researcher_cot_lora.py \
        --train-jsonl runs/researcher_cot/cots/pilot_cots.jsonl \
        --output-dir runs/training/v3_researcher_cot_7b/smoke \
        --max-steps 20

Usage (full training - 1 epoch):
    python scripts/train_v3_researcher_cot_lora.py \
        --train-jsonl /path/to/train.jsonl \
        [--val-jsonl /path/to/val.jsonl] \
        --output-dir runs/training/v3_researcher_cot_7b/full \
        --num-epochs 1

The script:
  1. Loads JSONL with a `messages` field (3-turn: system/user/assistant).
  2. Applies the Qwen2.5 chat template; masks loss to the assistant turn only.
  3. Trains with LoRA r=64, alpha=128, target_modules=all-linear, bf16, grad-ckpt.
  4. After training, merges LoRA into a standalone HF model at <output-dir>/merged/.
  5. Runs a quick generation sanity check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Patch broken wandb *before* trl/transformers imports it.
# Installed wandb has a protobuf incompatibility; replace with no-op module.
# ---------------------------------------------------------------------------
def _patch_wandb() -> None:
    if "wandb" in sys.modules:
        return
    _spec = type(sys)("wandb")
    _spec.__file__ = ""
    _root = types.ModuleType("wandb")
    _root.__spec__ = _spec
    _root.__version__ = "0.0.0"
    _root.init = lambda *a, **kw: None
    _root.log = lambda *a, **kw: None
    _root.finish = lambda *a, **kw: None
    for _sub in [
        "sdk", "sdk.lib", "sdk.lib.telemetry", "sdk.lib.config_util",
        "sdk.wandb_helper", "sdk.wandb_settings", "sdk.lib.filesystem",
        "sdk.lib.runid", "sdk.lib.deprecation", "sdk.lib.urls",
        "proto", "proto.wandb_telemetry_pb2", "util",
    ]:
        _m = types.ModuleType(f"wandb.{_sub}")
        _m.__spec__ = _spec
        sys.modules[f"wandb.{_sub}"] = _m
    sys.modules["wandb"] = _root


_patch_wandb()

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


# ---------------------------------------------------------------------------
# Defaults / hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "/newcpfs/user/yuanqianhao/hf_models/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_ROOT = str(
    Path(__file__).resolve().parents[1] / "runs/training/v3_researcher_cot_7b"
)

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"   # matches V2.5 setup

MAX_SEQ_LENGTH = 2048       # covers max ~1648 tokens in pilot set; ~1.7k in full dataset
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
# 80 GB L20Z: bsz=4, accum=4 → effective global batch = 16; safe for 2048-tok seqs
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LOGGING_STEPS = 5
SAVE_STEPS = 50


# ---------------------------------------------------------------------------
# Data loading & tokenisation with assistant-turn masking
# ---------------------------------------------------------------------------

# For Qwen2.5: the assistant turn starts after <|im_start|>assistant\n
# We tokenize with the full chat template, then mask labels everywhere
# EXCEPT the assistant content + <|im_end|> token.
ASSISTANT_START_STR = "<|im_start|>assistant\n"


def tokenize_with_assistant_mask(
    examples: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict[str, list]:
    """
    Tokenize a batch of {messages: [...]} examples.
    Labels are -100 for all tokens except the assistant turn.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for msgs in examples["messages"]:
        # Apply chat template (no generation prompt — we include the assistant turn)
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize full text
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,  # returns plain lists
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Find where the assistant turn starts by tokenising the prefix up to the
        # assistant block, then using that length as the mask boundary.
        # Strategy: find the last occurrence of ASSISTANT_START_STR in the text,
        # then measure how many tokens precede it.
        assistant_start_pos = text.rfind(ASSISTANT_START_STR)
        if assistant_start_pos == -1:
            # Fallback: mask everything (no assistant turn found; shouldn't happen)
            labels = [-100] * len(input_ids)
        else:
            prefix_text = text[: assistant_start_pos + len(ASSISTANT_START_STR)]
            prefix_ids = tokenizer(
                prefix_text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            # Clamp in case truncation cut the sequence shorter than the prefix
            n_prefix = min(len(prefix_ids), len(input_ids))
            labels = [-100] * n_prefix + list(input_ids[n_prefix:])
            labels = labels[: len(input_ids)]  # safety trim

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


@dataclass
class PaddingCollator:
    """
    Collate tokenised examples into padded batches.
    Pads input_ids and attention_mask with tokenizer.pad_token_id / 0;
    pads labels with -100 (ignored by cross-entropy).
    """
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)

        batch_ids, batch_mask, batch_labels = [], [], []
        for f in features:
            ids = f["input_ids"][:max_len]
            mask = f["attention_mask"][:max_len]
            lbls = f["labels"][:max_len]

            pad_len = max_len - len(ids)
            ids   = ids   + [self.tokenizer.pad_token_id] * pad_len
            mask  = mask  + [0] * pad_len
            lbls  = lbls  + [-100] * pad_len

            batch_ids.append(ids)
            batch_mask.append(mask)
            batch_labels.append(lbls)

        return {
            "input_ids":      torch.tensor(batch_ids,   dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask,  dtype=torch.long),
            "labels":         torch.tensor(batch_labels, dtype=torch.long),
        }


def load_and_tokenize(
    path: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    limit: int | None = None,
) -> Dataset:
    """Load JSONL and return a tokenised HF Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({"messages": obj["messages"]})
            if limit and len(records) >= limit:
                break

    raw_ds = Dataset.from_list(records)
    tok_ds = raw_ds.map(
        tokenize_with_assistant_mask,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        batched=True,
        remove_columns=["messages"],
        desc="Tokenising",
    )
    return tok_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V3 researcher-CoT LoRA SFT")
    parser.add_argument(
        "--train-jsonl", required=True,
        help="Path to training JSONL (each line: {messages: [sys,user,asst]})",
    )
    parser.add_argument(
        "--val-jsonl", default=None,
        help="Optional validation JSONL (same format).",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(DEFAULT_OUTPUT_ROOT, "run"),
        help="Directory for checkpoints and merged model.",
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL,
        help="Path to base HF model (Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1,
        help="Override num-epochs (e.g. 20 for a smoke test).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit dataset to first N records.",
    )
    parser.add_argument(
        "--per-device-batch-size", type=int, default=PER_DEVICE_BATCH_SIZE,
    )
    parser.add_argument(
        "--grad-accum", type=int, default=GRAD_ACCUM_STEPS,
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=MAX_SEQ_LENGTH,
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip LoRA merge at end.",
    )
    parser.add_argument(
        "--skip-gen-check", action="store_true",
        help="Skip generation sanity check after merge.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("V3 Researcher-CoT LoRA SFT")
    print(f"  base model : {args.base_model}")
    print(f"  train data : {args.train_jsonl}")
    print(f"  output     : {output_dir}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  GPU        : {gpu_name}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Datasets
    # ------------------------------------------------------------------
    print("Loading & tokenising dataset...")
    train_ds = load_and_tokenize(
        args.train_jsonl, tokenizer, args.max_seq_length, limit=args.limit
    )
    eval_ds = None
    if args.val_jsonl and Path(args.val_jsonl).exists():
        eval_ds = load_and_tokenize(args.val_jsonl, tokenizer, args.max_seq_length)

    print(f"  Train: {len(train_ds)} examples")
    if eval_ds:
        print(f"  Val:   {len(eval_ds)} examples")

    # Quick sanity: verify at least some labels are non-(-100)
    sample = train_ds[0]
    n_active = sum(1 for l in sample["labels"] if l != -100)
    n_total = len(sample["labels"])
    print(f"  Mask check: {n_active}/{n_total} tokens active (should be > 0)")
    if n_active == 0:
        print("  WARNING: all labels are -100 for first example — check masking!")

    # ------------------------------------------------------------------
    # 3. LoRA config
    # ------------------------------------------------------------------
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )

    # ------------------------------------------------------------------
    # 4. Training arguments
    # ------------------------------------------------------------------
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,           # -1 → use num_train_epochs
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="no" if eval_ds is None else "steps",
        eval_steps=SAVE_STEPS if eval_ds is not None else None,
        dataloader_num_workers=2,
        report_to="none",
        seed=42,
        remove_unused_columns=False,  # our dataset already only has the right columns
    )

    # ------------------------------------------------------------------
    # 5. Model + LoRA
    # ------------------------------------------------------------------
    print("Loading base model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # required for gradient checkpointing

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    collator = PaddingCollator(tokenizer=tokenizer, max_length=args.max_seq_length)

    print("\nStarting training...")
    train_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

    train_elapsed = time.time() - train_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Rough tokens/sec: total tokens across all examples × epochs / wall-clock
    # (based on actual dataset, not steps, to give a meaningful throughput estimate)
    total_tokens_est = sum(len(ex["input_ids"]) for ex in train_ds) * args.num_epochs
    steps_actual = args.max_steps if args.max_steps > 0 else (
        len(train_ds) * args.num_epochs // (args.per_device_batch_size * args.grad_accum)
    )
    tokens_per_sec = total_tokens_est / train_elapsed if train_elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"Training complete: {train_elapsed:.0f}s elapsed")
    print(f"Peak GPU memory:   {peak_mem_gb:.1f} GB / {total_memory_gb:.0f} GB")
    print(f"Approx tokens/sec: {tokens_per_sec:.0f}")
    print(f"{'='*60}\n")

    # Save adapter + tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"LoRA adapter saved to: {output_dir}")

    # ------------------------------------------------------------------
    # 7. Merge LoRA → standalone HF model
    # ------------------------------------------------------------------
    if not args.no_merge:
        merged_dir = output_dir / "merged"
        print(f"\nMerging LoRA into standalone model → {merged_dir}")
        merge_start = time.time()

        from peft import PeftModel
        del model
        torch.cuda.empty_cache()

        base_model_cpu = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        peft_model = PeftModel.from_pretrained(base_model_cpu, str(output_dir))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Merged model saved in {time.time()-merge_start:.0f}s → {merged_dir}")

        # ------------------------------------------------------------------
        # 8. Generation sanity check
        # ------------------------------------------------------------------
        if not args.skip_gen_check:
            print("\nRunning generation sanity check...")
            del merged, peft_model, base_model_cpu
            torch.cuda.empty_cache()

            check_model = AutoModelForCausalLM.from_pretrained(
                str(merged_dir),
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
            )
            check_tok = AutoTokenizer.from_pretrained(str(merged_dir))

            with open(args.train_jsonl) as f:
                first = json.loads(f.readline())
            probe_messages = first["messages"][:2]   # system + user only
            prompt = check_tok.apply_chat_template(
                probe_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = check_tok(prompt, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = check_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=check_tok.eos_token_id,
                )
            gen_text = check_tok.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            print("\n--- Generation check (first 400 chars) ---")
            print(gen_text[:400])
            print("--- end ---\n")

            check_file = output_dir / "gen_check.txt"
            check_file.write_text(
                f"PROMPT (first 500 chars):\n{prompt[:500]}\n\nGENERATION:\n{gen_text}\n"
            )
            print(f"Generation check saved to: {check_file}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    summary = {
        "train_jsonl": args.train_jsonl,
        "base_model": args.base_model,
        "num_examples": len(train_ds),
        "num_epochs": args.num_epochs,
        "max_steps_override": args.max_steps,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": LORA_TARGET_MODULES,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "per_device_batch_size": args.per_device_batch_size,
        "grad_accum": args.grad_accum,
        "effective_global_batch": args.per_device_batch_size * args.grad_accum,
        "train_elapsed_s": round(train_elapsed, 1),
        "peak_gpu_memory_gb": round(peak_mem_gb, 2),
        "approx_tokens_per_sec": round(tokens_per_sec, 0),
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary written to: {output_dir / 'train_summary.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

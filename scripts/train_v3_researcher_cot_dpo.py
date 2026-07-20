#!/usr/bin/env python3
"""V3 RL-v2: DPO/IPO on the researcher-CoT SFT policy (LoRA on frozen SFT, ref = adapter-off).

Rubric reward is non-verifiable → pairwise preference is the trustworthy primitive → DPO/IPO.
Trains a LoRA policy on top of the SFT checkpoint; the reference is the same model with the adapter
disabled (peft, zero extra weights). Consumes the format-matched preference pairs from
build_v3_preference_pairs.py (trivialize margin-5 backbone + route auxiliary).

Data: runs/researcher_cot/preference/pairs_all.jsonl  — {prompt:[system,user], chosen:str, rejected:str, margin}
Run (single GB200, LoRA fits a 32B):
  python scripts/train_v3_researcher_cot_dpo.py \
    --base-model /mnt/3fs/lxh/agentic-training/runs/qs_researcher_full_sft/v3_qs_full_sft_32b_anchored_1ep_v4/output/checkpoint-101 \
    --pairs-jsonl runs/researcher_cot/preference/pairs_all.jsonl \
    --loss-type sigmoid --beta 0.1 --output-dir runs/researcher_cot/rl/dpo_32b_v1

loss-type: sigmoid = DPO (default) · ipo = IPO (robust to over-optimization) · kto_pair = KTO-pair.
Requires: trl>=0.12, peft, transformers>=4.51 (Qwen3-ready; cc000's env). For Qwen3 bases we set
enable_thinking=False so <think> is never injected.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

# Version-compat shim: transformers>=4.47 calls Trainer.log(logs, start_time) but trl 0.12's
# DPOTrainer.log(self, logs) takes only one positional -> TypeError. Qwen3-arch bases (D1/M2) need
# tfm>=4.51, so accept and drop the extra args. No-op on tfm 4.46 (called with a single arg).
_ORIG_DPO_LOG = DPOTrainer.log
def _dpo_log_compat(self, logs, *args, **kwargs):  # noqa: ANN001
    return _ORIG_DPO_LOG(self, logs)
DPOTrainer.log = _dpo_log_compat


def build_dataset(pairs_jsonl: str, tok) -> Dataset:
    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:  # tokenizers without the enable_thinking kwarg (e.g. Qwen2.5)
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    rows = []
    for line in Path(pairs_jsonl).open():
        r = json.loads(line)
        rows.append({"prompt": render(r["prompt"]), "chosen": r["chosen"], "rejected": r["rejected"]})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="SFT checkpoint = policy init + reference")
    ap.add_argument("--pairs-jsonl", default="runs/researcher_cot/preference/pairs_all.jsonl")
    ap.add_argument("--output-dir", default="runs/researcher_cot/rl/dpo_v1")
    ap.add_argument("--loss-type", default="sigmoid", choices=["sigmoid", "ipo", "kto_pair"])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = build_dataset(args.pairs_jsonl, tok).shuffle(seed=0)
    n_val = max(1, int(len(ds) * args.val_frac))
    train_ds, eval_ds = ds.select(range(n_val, len(ds))), ds.select(range(n_val))
    print(f"[dpo] {len(train_ds)} train / {len(eval_ds)} val pairs from {args.pairs_jsonl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM", target_modules="all-linear")

    cfg = DPOConfig(
        output_dir=args.output_dir, loss_type=args.loss_type, beta=args.beta,
        learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch, gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length, max_prompt_length=args.max_prompt_length,
        bf16=True, gradient_checkpointing=True, logging_steps=5,
        eval_strategy="steps", eval_steps=25, save_strategy="epoch",
        warmup_ratio=0.03, lr_scheduler_type="cosine", report_to=[],
    )
    # ref_model=None + peft_config -> reference is the adapter-disabled base (zero extra weights)
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg,
                         train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[dpo] saved adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Distributed full-parameter SFT for V3 researcher-CoT data.

This entrypoint is the QS/GB200 path for 32B full fine-tuning. It is meant to
be launched with torchrun, for example:

    torchrun --nproc_per_node=4 scripts/train_v3_researcher_cot_full_fsdp.py \
      --train-jsonl /mnt/3fs/lxh/agentic-training/data/researcher_cot/prejudge_v1/train.jsonl \
      --val-jsonl /mnt/3fs/lxh/agentic-training/data/researcher_cot/prejudge_v1/val.jsonl \
      --base-model /mnt/3fs/lxh/agentic-training/models/Qwen2.5-32B-Instruct \
      --output-dir /mnt/3fs/lxh/agentic-training/qs_researcher_full_sft/run/output \
      --limit 8 --max-steps 1 --max-seq-length 1024

It intentionally does not use PEFT/LoRA. Trainer is configured for FSDP
full_shard with transformer-layer auto wrapping and sharded state dict saves.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _patch_wandb() -> None:
    if "wandb" in sys.modules:
        return
    spec = type(sys)("wandb")
    spec.__file__ = ""
    root = types.ModuleType("wandb")
    root.__spec__ = spec
    root.__version__ = "0.0.0"
    root.init = lambda *a, **kw: None
    root.log = lambda *a, **kw: None
    root.finish = lambda *a, **kw: None
    for sub in [
        "sdk",
        "sdk.lib",
        "sdk.lib.telemetry",
        "sdk.lib.config_util",
        "sdk.wandb_helper",
        "sdk.wandb_settings",
        "sdk.lib.filesystem",
        "sdk.lib.runid",
        "sdk.lib.deprecation",
        "sdk.lib.urls",
        "proto",
        "proto.wandb_telemetry_pb2",
        "util",
    ]:
        module = types.ModuleType(f"wandb.{sub}")
        module.__spec__ = spec
        sys.modules[f"wandb.{sub}"] = module
    sys.modules["wandb"] = root


_patch_wandb()

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch


def _patch_transformers_modeling_layers() -> None:
    """Patch older/internal transformer builds for newer integration imports."""
    if importlib.util.find_spec("transformers.modeling_layers") is not None:
        return
    module = types.ModuleType("transformers.modeling_layers")
    module.__spec__ = importlib.machinery.ModuleSpec("transformers.modeling_layers", loader=None)

    class GradientCheckpointingLayer(torch.nn.Module):
        pass

    module.GradientCheckpointingLayer = GradientCheckpointingLayer
    sys.modules["transformers.modeling_layers"] = module


def _patch_broken_apex_amp() -> None:
    """Patch QS images where a non-NVIDIA apex package shadows apex.amp."""
    try:
        import apex  # type: ignore
    except Exception:
        apex = types.ModuleType("apex")
        apex.__spec__ = importlib.machinery.ModuleSpec("apex", loader=None)
        sys.modules["apex"] = apex
    if hasattr(apex, "amp"):
        return

    amp = types.ModuleType("apex.amp")
    amp.__spec__ = importlib.machinery.ModuleSpec("apex.amp", loader=None)

    def initialize(model, optimizer=None, **_: Any):
        return (model, optimizer) if optimizer is not None else model

    class _ScaleLoss:
        def __init__(self, loss, optimizer=None):
            self.loss = loss
            self.optimizer = optimizer

        def __enter__(self):
            return self.loss

        def __exit__(self, exc_type, exc, tb):
            return False

    def scale_loss(loss, optimizer=None):
        return _ScaleLoss(loss, optimizer)

    amp.initialize = initialize
    amp.scale_loss = scale_loss
    apex.amp = amp
    sys.modules["apex.amp"] = amp


_patch_transformers_modeling_layers()
_patch_broken_apex_amp()

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


DEFAULT_BASE_MODEL = "/mnt/3fs/lxh/agentic-training/models/Qwen2.5-32B-Instruct"
DEFAULT_OUTPUT_ROOT = str(
    Path(__file__).resolve().parents[1] / "runs/training/v3_researcher_cot_32b_full_fsdp"
)

ASSISTANT_START_STR = "<|im_start|>assistant\n"


def derive_assistant_start_str(tokenizer) -> str:
    """Model-agnostic assistant-turn marker = the exact suffix add_generation_prompt appends.
    Qwen -> '<|im_start|>assistant\\n'; DeepSeek-R1 -> '<｜Assistant｜>'; Llama -> its header block.
    Falls back to the Qwen default if the diff can't be computed."""
    probe = [{"role": "user", "content": "x"}]
    try:
        base = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
        gen = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
    except Exception:
        return ASSISTANT_START_STR
    if gen.startswith(base) and len(gen) > len(base):
        return gen[len(base):]
    return ASSISTANT_START_STR


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank0() -> bool:
    return rank() == 0


def log(message: str) -> None:
    print(f"[rank={rank()} local_rank={local_rank()}] {message}", flush=True)


def tokenize_with_assistant_mask(
    examples: dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict[str, list]:
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        assistant_start_pos = text.rfind(ASSISTANT_START_STR)
        if assistant_start_pos == -1:
            labels = [-100] * len(input_ids)
        else:
            prefix_text = text[: assistant_start_pos + len(ASSISTANT_START_STR)]
            prefix_ids = tokenizer(
                prefix_text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            n_prefix = min(len(prefix_ids), len(input_ids))
            labels = [-100] * n_prefix + list(input_ids[n_prefix:])
            labels = labels[: len(input_ids)]

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
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)

        batch_ids, batch_mask, batch_labels = [], [], []
        for feature in features:
            ids = feature["input_ids"][:max_len]
            mask = feature["attention_mask"][:max_len]
            labels = feature["labels"][:max_len]

            pad_len = max_len - len(ids)
            ids = ids + [self.tokenizer.pad_token_id] * pad_len
            mask = mask + [0] * pad_len
            labels = labels + [-100] * pad_len

            batch_ids.append(ids)
            batch_mask.append(mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def load_and_tokenize(
    path: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    limit: int | None = None,
) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"JSONL row missing messages in {path}")
            records.append({"messages": messages})
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"no records loaded from {path}")

    raw_ds = Dataset.from_list(records)
    return raw_ds.map(
        tokenize_with_assistant_mask,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        batched=True,
        remove_columns=["messages"],
        desc=f"Tokenising rank {rank()}",
    )


def checkpoint_dirs(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(p.name for p in output_dir.glob("checkpoint-*") if p.is_dir())


def resolve_checkpoint(output_dir: Path, checkpoint: str | None) -> str | None:
    if checkpoint in (None, "", "none"):
        return None
    if checkpoint == "latest":
        latest = get_last_checkpoint(str(output_dir))
        if latest is None:
            raise SystemExit(f"--resume-from-checkpoint latest requested, but no checkpoint found in {output_dir}")
        return latest
    return checkpoint


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    fsdp_config = {
        "fsdp_version": 1,
        "transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer],
        "activation_checkpointing": bool(args.fsdp_activation_checkpointing),
        "sync_module_states": True,
        "cpu_ram_efficient_loading": bool(args.fsdp_cpu_ram_efficient_loading),
        "use_orig_params": True,
        "limit_all_gathers": True,
        "forward_prefetch": False,
        "backward_prefetch": "backward_pre",
        "state_dict_type": args.fsdp_state_dict_type,
    }
    if args.save_only_model and args.fsdp_state_dict_type == "SHARDED_STATE_DICT":
        raise SystemExit("save_only_model is incompatible with FSDP SHARDED_STATE_DICT in this transformers build")
    return TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        tf32=True,
        # For FSDP full_shard, use activation_checkpointing in fsdp_config
        # instead of Trainer gradient_checkpointing to avoid redundant all-gather.
        gradient_checkpointing=False,
        fsdp="full_shard auto_wrap",
        fsdp_config=fsdp_config,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no" if args.val_jsonl is None else "steps",
        eval_steps=args.eval_steps if args.val_jsonl is not None else None,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        save_safetensors=True,
        save_only_model=args.save_only_model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3 researcher-CoT full SFT with FSDP")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_ROOT) / "run")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--assistant-start-str", default="auto",
                        help="assistant-turn marker for label masking; 'auto' derives it from the "
                             "chat template (works for Qwen/DeepSeek/Llama), else pass a literal.")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch", "best"])
    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fsdp-transformer-layer", default="Qwen2DecoderLayer")
    parser.add_argument(
        "--fsdp-state-dict-type",
        default="SHARDED_STATE_DICT",
        choices=["FULL_STATE_DICT", "SHARDED_STATE_DICT", "LOCAL_STATE_DICT"],
    )
    parser.add_argument("--fsdp-activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fsdp-cpu-ram-efficient-loading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-only-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--final-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--summary-name", default="train_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(args)
    resume_checkpoint = resolve_checkpoint(args.output_dir, args.resume_from_checkpoint)

    log("V3 researcher-CoT full SFT")
    log(f"base_model={args.base_model}")
    log(f"train_jsonl={args.train_jsonl}")
    log(f"val_jsonl={args.val_jsonl}")
    log(f"output_dir={args.output_dir}")
    log(f"resume_from_checkpoint={resume_checkpoint}")
    log(f"world_size={os.environ.get('WORLD_SIZE', '1')}")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        log(f"gpu={torch.cuda.get_device_name(local_rank())}")

    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="right")
    global ASSISTANT_START_STR
    ASSISTANT_START_STR = (derive_assistant_start_str(tokenizer)
                           if args.assistant_start_str == "auto" else args.assistant_start_str)
    log(f"assistant_start_str={ASSISTANT_START_STR!r} (mode={args.assistant_start_str})")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("loading datasets")
    train_ds = load_and_tokenize(args.train_jsonl, tokenizer, args.max_seq_length, limit=args.limit)
    eval_ds = None
    if args.val_jsonl and Path(args.val_jsonl).exists():
        eval_ds = load_and_tokenize(args.val_jsonl, tokenizer, args.max_seq_length)

    sample = train_ds[0]
    active_labels = sum(1 for label in sample["labels"] if label != -100)
    if active_labels == 0:
        raise SystemExit("first training row has zero active assistant labels")
    log(f"train_examples={len(train_ds)} active_labels_first={active_labels}/{len(sample['labels'])}")
    if eval_ds is not None:
        log(f"eval_examples={len(eval_ds)}")

    log("loading full model on CPU for FSDP wrapping")
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    log(f"model_loaded_s={time.time() - load_start:.1f}")

    collator = PaddingCollator(tokenizer=tokenizer, max_length=args.max_seq_length)

    from transformers import TrainerCallback

    class _EmptyCacheAfterEval(TrainerCallback):
        """Free eval-loop allocator residue; at 32B full-param the margin is ~1GB and the
        post-eval training step OOMs otherwise (observed twice at step 101)."""

        def on_evaluate(self, *a, **k):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[_EmptyCacheAfterEval()],
    )

    train_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_elapsed = time.time() - train_start

    eval_metrics = None
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()

    if args.final_save:
        trainer.save_model(str(args.output_dir))
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(str(args.output_dir))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    peak_mem_gb = None
    total_memory_gb = None
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated(local_rank()) / 1e9
        total_memory_gb = torch.cuda.get_device_properties(local_rank()).total_memory / 1e9

    if trainer.is_world_process_zero():
        total_tokens_est = sum(len(ex["input_ids"]) for ex in train_ds)
        summary = {
            "status": "ok",
            "mode": "full_fsdp_sft",
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
            "base_model": args.base_model,
            "output_dir": str(args.output_dir),
            "num_train_examples": len(train_ds),
            "num_eval_examples": len(eval_ds) if eval_ds is not None else 0,
            "num_epochs": args.num_epochs,
            "max_steps_override": args.max_steps,
            "resume_from_checkpoint": resume_checkpoint,
            "learning_rate": args.lr,
            "max_seq_length": args.max_seq_length,
            "per_device_batch_size": args.per_device_batch_size,
            "grad_accum": args.grad_accum,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_global_batch": args.per_device_batch_size * args.grad_accum * int(os.environ.get("WORLD_SIZE", "1")),
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": training_args.fsdp_config,
            "save_strategy": str(training_args.save_strategy),
            "save_only_model": args.save_only_model,
            "final_save": args.final_save,
            "train_elapsed_s": round(train_elapsed, 1),
            "peak_gpu_memory_gb_rank0": round(peak_mem_gb, 2) if peak_mem_gb is not None else None,
            "total_gpu_memory_gb_rank0": round(total_memory_gb, 2) if total_memory_gb is not None else None,
            "tokens_per_sec_rank0_est": round(total_tokens_est / train_elapsed, 1) if train_elapsed > 0 else 0,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "checkpoints": checkpoint_dirs(args.output_dir),
            "summary_name": args.summary_name,
        }
        summary_path = args.output_dir / args.summary_name
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()

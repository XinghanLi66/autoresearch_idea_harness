#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.benchmarking import get_task
from autoresearch_idea_harness.io import load_config, write_json
from autoresearch_idea_harness.training_manifest import PROPOSAL_FORMAT, SYSTEM_PROMPT


DEFAULT_MODEL_DIR = (
    ROOT
    / "runs"
    / "training"
    / "v3_sft_qwen25_32b"
    / "v3_sft_200_smoke"
    / "checkpoints"
    / "phase_000_2025-04"
    / "final"
)


def build_research_question(task_packet: dict[str, Any]) -> str:
    return (
        f"What concrete method change for `{task_packet['task']}` / `{task_packet.get('subtask')}` "
        f"is most likely to exceed pass metric {task_packet.get('pass_metric')} while staying implementable "
        "inside the benchmark worker constraints?"
    )


def build_proposal_task_packet(cfg: dict[str, Any], task_name: str, subtask: str | None) -> dict[str, Any]:
    task = get_task(cfg, task_name, subtask=subtask)
    task_packet = task.task_packet(cfg)
    task_packet["subtask"] = subtask or getattr(task, "active_subtasks", [None])[0]
    task_packet["condition_strategy"] = "with_research_question"
    task_packet["research_question"] = build_research_question(task_packet)
    task_packet["proposal_granularity"] = "one worker-implementable idea"
    task_packet["proposal_contract"] = {
        "single_proposal": True,
        "worker_implementable": True,
        "avoid_lists_of_alternatives": True,
        "must_respect_worker_constraints": True,
    }
    return task_packet


def build_messages(task_packet: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        f"{PROPOSAL_FORMAT}\n"
        "Use only the task packet below. The output should be one proposal, not a "
        "list of alternatives. Keep the proposal at worker-implementable granularity. "
        "Answer the `research_question` field directly.\n\n"
        "Task packet JSON:\n"
        f"{json.dumps(task_packet, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def generate_with_checkpoint(
    model_dir: Path,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    trust_remote_code: bool,
    attn_implementation: str | None,
) -> tuple[str, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    model.eval()
    load_elapsed = time.time() - t0

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_elapsed = time.time() - t1
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    meta = {
        "model_dir": str(model_dir),
        "prompt_tokens": int(enc["input_ids"].shape[1]),
        "new_tokens": int(new_tokens.shape[0]),
        "load_elapsed_s": round(load_elapsed, 3),
        "generation_elapsed_s": round(gen_elapsed, 3),
        "torch_dtype": "bfloat16",
        "device": str(model.device),
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    return text, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one V3 proposal from a trained checkpoint for an MLS/MLE task packet."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--task", default="dl_activation_function")
    parser.add_argument("--subtask", default="resnet20-cifar10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "v3_checkpoint_proposal_smoke",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write task packet and prompt without loading the model.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_packet = build_proposal_task_packet(cfg, args.task, args.subtask)

    messages = build_messages(task_packet)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "task_packet.json", task_packet)
    write_json(out_dir / "messages.json", messages)
    (out_dir / "prompt.txt").write_text(
        "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
    )

    if args.dry_run:
        write_json(out_dir / "meta.json", {"dry_run": True, "model_dir": str(args.model_dir)})
        print(json.dumps({"status": "dry_run", "output_dir": str(out_dir)}, ensure_ascii=False))
        return

    proposal, meta = generate_with_checkpoint(
        args.model_dir,
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    (out_dir / "proposal.txt").write_text(proposal)
    write_json(out_dir / "meta.json", {"dry_run": False, **meta})
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), **meta}, ensure_ascii=False))


if __name__ == "__main__":
    main()

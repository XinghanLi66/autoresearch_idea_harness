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
sys.path.insert(0, str(ROOT / "scripts"))

from autoresearch_idea_harness.benchmarking import MLS_SUBTASKS
from autoresearch_idea_harness.io import load_config, write_json, write_jsonl
from generate_v3_checkpoint_proposal import build_messages, build_proposal_task_packet
from score_v3_proposal_quality import score_proposal


DEFAULT_MODEL_DIR = (
    ROOT
    / "runs"
    / "training"
    / "v3_sft_qwen25_32b"
    / "v3_sft_strict_batch1000_auto"
    / "checkpoints"
    / "phase_000_2025-04"
    / "final"
)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_") or "task"


def _prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)


def _load_model(
    model_dir: Path,
    *,
    trust_remote_code: bool,
    attn_implementation: str | None,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    model.eval()
    return tokenizer, model, {
        "model_dir": str(model_dir),
        "load_elapsed_s": round(time.time() - started, 3),
        "torch_dtype": "bfloat16",
        "device": str(getattr(model, "device", "auto")),
        "attn_implementation": attn_implementation,
    }


def _generate_one(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, dict[str, Any]]:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    started = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, {
        "prompt_tokens": int(enc["input_ids"].shape[1]),
        "new_tokens": int(new_tokens.shape[0]),
        "generation_elapsed_s": round(time.time() - started, 3),
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }


def _task_list(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    tasks = list(args.tasks or [])
    if not tasks:
        tasks = list((cfg.get("v2_3") or {}).get("tasks") or [])
    if not tasks:
        raise ValueError("No tasks supplied and config.v2_3.tasks is empty.")
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]
    return tasks


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], load_meta: dict[str, Any]) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    scored = [row for row in ok if isinstance(row.get("quality"), dict)]
    summary = {
        "output_dir": str(output_dir),
        "task_count": len(rows),
        "ok_count": len(ok),
        "error_count": sum(1 for row in rows if row.get("status") == "error"),
        "quality_pass_count": sum(1 for row in scored if row["quality"].get("verdict") == "pass"),
        "quality_mean_score": round(
            sum(float(row["quality"].get("score", 0.0)) for row in scored) / len(scored),
            3,
        )
        if scored
        else None,
        "load": load_meta,
        "rows": rows,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "summary_rows.jsonl", rows)
    lines = [
        "# V3 Checkpoint Proposal Batch",
        "",
        f"- Task count: {summary['task_count']}",
        f"- OK: {summary['ok_count']}",
        f"- Errors: {summary['error_count']}",
        f"- Quality pass: {summary['quality_pass_count']}",
        f"- Mean score: {summary['quality_mean_score']}",
        "",
        "| Task | Subtask | Tokens | Score | Verdict | Failed criteria |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        failed = [
            name
            for name, item in (quality.get("criteria") or {}).items()
            if isinstance(item, dict) and not item.get("passed")
        ]
        lines.append(
            f"| `{row.get('task')}` | `{row.get('subtask')}` | "
            f"{row.get('prompt_tokens', 0)}+{row.get('new_tokens', 0)} | "
            f"{quality.get('score', '-')} | {quality.get('verdict', row.get('status'))} | "
            f"{', '.join(failed) or '-'} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _task_list(cfg, args)
    rows: list[dict[str, Any]] = []

    if args.dry_run:
        load_meta = {"dry_run": True, "model_dir": str(args.model_dir.resolve())}
        tokenizer = model = None
    else:
        tokenizer, model, load_meta = _load_model(
            args.model_dir.resolve(),
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
        write_json(output_dir / "load_meta.json", load_meta)

    for index, task_name in enumerate(tasks):
        subtask = args.subtask or (MLS_SUBTASKS.get(task_name) or [None])[0]
        task_dir = output_dir / f"{index:02d}_{_safe_name(task_name)}"
        task_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        row: dict[str, Any] = {
            "index": index,
            "task": task_name,
            "subtask": subtask,
            "task_dir": str(task_dir),
            "status": "error",
        }
        try:
            task_packet = build_proposal_task_packet(cfg, task_name, subtask)
            messages = build_messages(task_packet)
            write_json(task_dir / "task_packet.json", task_packet)
            write_json(task_dir / "messages.json", messages)
            (task_dir / "prompt.txt").write_text(_prompt_text(messages))

            if args.dry_run:
                write_json(task_dir / "meta.json", {"dry_run": True})
                row.update({
                    "status": "dry_run",
                    "pass_metric": task_packet.get("pass_metric"),
                    "baseline_metric": task_packet.get("baseline_metric"),
                })
            else:
                assert tokenizer is not None and model is not None
                proposal, meta = _generate_one(
                    tokenizer,
                    model,
                    messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                (task_dir / "proposal.txt").write_text(proposal)
                score = score_proposal(proposal, task_packet)
                write_json(task_dir / "meta.json", {"dry_run": False, **load_meta, **meta})
                write_json(task_dir / "proposal_quality.json", score)
                row.update({
                    "status": "ok",
                    "pass_metric": task_packet.get("pass_metric"),
                    "baseline_metric": task_packet.get("baseline_metric"),
                    "prompt_tokens": meta["prompt_tokens"],
                    "new_tokens": meta["new_tokens"],
                    "generation_elapsed_s": meta["generation_elapsed_s"],
                    "quality": score,
                })
        except Exception as exc:  # noqa: BLE001 - batch run must preserve all failures.
            error = {"error": str(exc), "type": type(exc).__name__}
            write_json(task_dir / "error.json", error)
            row["error"] = error
        row["elapsed_s"] = round(time.time() - started, 3)
        rows.append(row)
        _write_summary(output_dir, rows, load_meta)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    return json.loads((output_dir / "summary.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3 checkpoint proposals for a batch of MLS tasks.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "v3_checkpoint_proposal_batch")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run_batch(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("error_count") == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

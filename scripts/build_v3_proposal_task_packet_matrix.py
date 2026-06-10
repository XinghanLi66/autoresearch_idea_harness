#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from autoresearch_idea_harness.benchmarking import MLS_SUBTASKS
from autoresearch_idea_harness.io import load_config, write_json
from generate_v3_checkpoint_proposal import build_messages, build_proposal_task_packet


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_matrix(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    tasks = args.tasks or list((cfg.get("v2_3") or {}).get("tasks") or [])
    if not tasks:
        raise ValueError("No tasks supplied and config.v2_3.tasks is empty.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    warnings: list[str] = []
    errors: list[str] = []

    for task_name in tasks:
        subtask = args.subtask or (MLS_SUBTASKS.get(task_name) or [None])[0]
        task_dir = output_dir / _safe_name(task_name)
        task_dir.mkdir(parents=True, exist_ok=True)
        try:
            packet = build_proposal_task_packet(cfg, task_name, subtask)
            messages = build_messages(packet)
        except Exception as exc:  # noqa: BLE001 - record all matrix failures
            errors.append(f"{task_name}: failed to build packet: {exc}")
            rows.append({"task": task_name, "subtask": subtask, "status": "error", "error": str(exc)})
            continue

        prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
        write_json(task_dir / "task_packet.json", packet)
        write_json(task_dir / "messages.json", messages)
        (task_dir / "prompt.txt").write_text(prompt)

        refs = packet.get("frontline_papers") or []
        row_warnings = []
        if packet.get("condition_strategy") != "with_research_question":
            row_warnings.append("condition_strategy is not with_research_question")
        if not packet.get("research_question"):
            row_warnings.append("missing research_question")
        if packet.get("proposal_granularity") != "one worker-implementable idea":
            row_warnings.append("unexpected proposal_granularity")
        if len(refs) < args.min_frontline_refs:
            row_warnings.append(f"frontline_ref_count={len(refs)} < {args.min_frontline_refs}")
        prompt_tokens = _approx_tokens(prompt)
        if prompt_tokens > args.max_prompt_tokens:
            row_warnings.append(f"approx_prompt_tokens={prompt_tokens} > {args.max_prompt_tokens}")

        for warning in row_warnings:
            warnings.append(f"{task_name}: {warning}")

        rows.append({
            "task": task_name,
            "subtask": subtask,
            "status": "ok",
            "frontline_ref_count": len(refs),
            "frontline_refs": [row.get("arxiv_id") for row in refs],
            "baseline_metric": packet.get("baseline_metric"),
            "pass_metric": packet.get("pass_metric"),
            "condition_strategy": packet.get("condition_strategy"),
            "research_question": packet.get("research_question"),
            "proposal_granularity": packet.get("proposal_granularity"),
            "prompt_chars": len(prompt),
            "approx_prompt_tokens": prompt_tokens,
            "warnings": row_warnings,
            "files": {
                "task_packet": str(task_dir / "task_packet.json"),
                "messages": str(task_dir / "messages.json"),
                "prompt": str(task_dir / "prompt.txt"),
            },
        })

    summary = {
        "output_dir": str(output_dir),
        "task_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "warning_count": len(warnings),
        "error_count": len(errors),
        "min_frontline_refs": args.min_frontline_refs,
        "max_prompt_tokens": args.max_prompt_tokens,
        "rows": rows,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(output_dir / "summary.json", summary)
    lines = [
        "# V3 Proposal Task Packet Matrix",
        "",
        f"- Task count: {summary['task_count']}",
        f"- OK: {summary['ok_count']}",
        f"- Warnings: {summary['warning_count']}",
        f"- Errors: {summary['error_count']}",
        "",
        "| Task | Subtask | Refs | Prompt tokens | Status | Warnings |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('task')}` | `{row.get('subtask')}` | {row.get('frontline_ref_count', 0)} | "
            f"{row.get('approx_prompt_tokens', 0)} | {row.get('status')} | "
            f"{'; '.join(row.get('warnings') or []) or '-'} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build offline with_research_question task packets/prompts for V3 checkpoint proposal smoke."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "v3_checkpoint_proposal_smoke" / "task_packet_matrix")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--min-frontline-refs", type=int, default=3)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    summary = build_matrix(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["error_count"] or (args.fail_on_warning and summary["warning_count"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

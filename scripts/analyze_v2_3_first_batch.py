#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.1f}%"


def sample_sort_key(path: Path) -> tuple[str, str, int]:
    parts = path.name.split("__")
    sample = parts[-1] if parts else "s00"
    try:
        idx = int(sample.lstrip("s"))
    except Exception:
        idx = 0
    task = parts[0] if parts else ""
    module = parts[2] if len(parts) > 2 else ""
    return task, module, idx


def infer_latest_events(root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    events_path = root / "events.jsonl"
    if not events_path.exists():
        return latest
    for row in read_jsonl(events_path):
        sid = row.get("sample_id")
        if sid:
            latest[sid] = row
    return latest


def eval_progress(sample_dir: Path) -> tuple[int | None, float | None]:
    eval_log = sample_dir / "eval.log"
    if not eval_log.exists():
        eval_log = sample_dir / "workspace" / "eval.log"
    if not eval_log.exists():
        return None, None
    text = eval_log.read_text(errors="replace")[-40000:]
    epochs = [int(m.group(1)) for m in re.finditer(r"epoch=(\d+)", text)]
    accs = [float(m.group(1)) for m in re.finditer(r"test_acc=([0-9.]+)", text)]
    return (max(epochs) if epochs else None), (accs[-1] if accs else None)


def collect_samples(root: Path) -> list[dict[str, Any]]:
    latest_events = infer_latest_events(root)
    rows = []
    for sample_dir in sorted(root.glob("*__s*"), key=sample_sort_key):
        if not sample_dir.is_dir():
            continue
        summary = read_json(sample_dir / "summary.json")
        task_packet = read_json(sample_dir / "task_packet.json")
        worker = summary.get("worker_result") or {}
        sample_id = sample_dir.name
        task = summary.get("task") or task_packet.get("task") or sample_id.split("__")[0]
        subtask = summary.get("subtask") or task_packet.get("subtask") or ""
        module = summary.get("module_id") or task_packet.get("module_id") or ""
        status = worker.get("status")
        if not status:
            if (sample_dir / "error.json").exists():
                status = "error"
            elif (sample_dir / "result.json").exists():
                status = "done"
            else:
                status = "running"
        epoch, acc = eval_progress(sample_dir)
        rows.append({
            "sample_id": sample_id,
            "path": sample_dir,
            "task": task,
            "subtask": subtask,
            "module": module,
            "status": status,
            "passed": worker.get("passed"),
            "metric": worker.get("val_metric"),
            "baseline_metric": summary.get("baseline_metric") or task_packet.get("baseline_metric"),
            "pass_metric": summary.get("pass_metric") or task_packet.get("pass_metric"),
            "improvement": worker.get("improvement"),
            "elapsed_s": worker.get("elapsed_s"),
            "mean_success_probability": summary.get("mean_success_probability"),
            "forecast_count": summary.get("forecast_count") or len(read_jsonl(sample_dir / "expert_forecasts.jsonl")),
            "latest_event": (latest_events.get(sample_id) or {}).get("event"),
            "epoch": epoch,
            "latest_acc": acc,
            "error": (read_json(sample_dir / "error.json").get("error") or worker.get("error")),
        })
    return rows


def aggregate_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "")].append(row)
    out = []
    for name, items in sorted(groups.items()):
        done = [x for x in items if x["status"] == "done" and x.get("metric") is not None]
        errors = [x for x in items if x["status"] == "error"]
        metrics = [float(x["metric"]) for x in done]
        passed = [x for x in done if x.get("passed")]
        out.append({
            key: name,
            "samples": len(items),
            "done": len(done),
            "running": len([x for x in items if x["status"] == "running"]),
            "errors": len(errors),
            "passed": len(passed),
            "pass_rate": (len(passed) / len(done)) if done else None,
            "mean_metric": mean(metrics) if metrics else None,
            "median_metric": median(metrics) if metrics else None,
            "best_metric": max(metrics) if metrics else None,
            "worst_metric": min(metrics) if metrics else None,
        })
    return out


def expert_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for settlement in root.glob("*__s*/settlement.json"):
        data = read_json(settlement)
        for row in data.get("rows") or []:
            rows.append(row)
    return rows


def aggregate_experts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("expert_id") or "")].append(row)
    out = []
    for expert, items in sorted(groups.items()):
        probs = [float(x.get("success_probability", 0.0)) for x in items]
        outcomes = [float(x.get("outcome", 0.0)) for x in items]
        briers = [float(x.get("brier_score", math.nan)) for x in items if x.get("brier_score") is not None]
        logs = [float(x.get("log_score", math.nan)) for x in items if x.get("log_score") is not None]
        out.append({
            "expert": expert,
            "n": len(items),
            "mean_probability": mean(probs) if probs else None,
            "empirical_pass_rate": mean(outcomes) if outcomes else None,
            "mean_brier": mean(briers) if briers else None,
            "mean_log": mean(logs) if logs else None,
        })
    return out


def render_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


def render_report(root: Path, out_path: Path) -> dict[str, Any]:
    samples = collect_samples(root)
    done = [x for x in samples if x["status"] == "done"]
    running = [x for x in samples if x["status"] == "running"]
    errors = [x for x in samples if x["status"] == "error"]
    total_planned = 10 * 9 * 20
    expert_settled = expert_rows(root)
    by_task = aggregate_by(samples, "task")
    by_module = aggregate_by(samples, "module")
    by_expert = aggregate_experts(expert_settled)

    lines = [
        "# V2.3 First-Batch Analysis",
        "",
        f"- Generated at: `{utc_now()}`",
        f"- Root: `{root}`",
        f"- Planned total: `{total_planned}` samples",
        f"- Materialized sample dirs: `{len(samples)}`",
        f"- Completed: `{len(done)}`",
        f"- Running: `{len(running)}`",
        f"- Errors: `{len(errors)}`",
        "",
        "## Read First",
        "",
    ]
    if not done:
        lines.append("No completed samples yet; this report is a progress snapshot rather than an outcome analysis.")
    else:
        pass_rate = sum(1 for x in done if x.get("passed")) / len(done)
        best = max(done, key=lambda x: float(x.get("metric") or -1e9))
        lines.extend([
            f"- Completed sample pass rate so far: `{fmt_pct(pass_rate)}`.",
            f"- Best completed metric so far: `{fmt_float(best.get('metric'))}` from `{best['sample_id']}`.",
            "- Treat these numbers as early evidence only; first-batch coverage is not yet balanced across all 9 modules.",
        ])

    lines.extend(["", "## Progress By Module", ""])
    lines.extend(render_table(
        ["Module", "Samples", "Done", "Run", "Err", "Pass", "Pass Rate", "Mean", "Best"],
        [[
            row["module"], row["samples"], row["done"], row["running"], row["errors"], row["passed"],
            fmt_pct(row["pass_rate"]), fmt_float(row["mean_metric"]), fmt_float(row["best_metric"]),
        ] for row in by_module],
    ))

    lines.extend(["", "## Progress By Task", ""])
    lines.extend(render_table(
        ["Task", "Samples", "Done", "Run", "Err", "Pass", "Pass Rate", "Mean", "Best"],
        [[
            row["task"], row["samples"], row["done"], row["running"], row["errors"], row["passed"],
            fmt_pct(row["pass_rate"]), fmt_float(row["mean_metric"]), fmt_float(row["best_metric"]),
        ] for row in by_task],
    ))

    if done:
        top_done = sorted(done, key=lambda x: float(x.get("metric") or -1e9), reverse=True)[:20]
        lines.extend(["", "## Top Completed Samples", ""])
        lines.extend(render_table(
            ["Sample", "Metric", "Pass Metric", "Passed", "Mean Forecast", "Elapsed Min"],
            [[
                x["sample_id"],
                fmt_float(x.get("metric")),
                fmt_float(x.get("pass_metric")),
                x.get("passed"),
                fmt_float(x.get("mean_success_probability"), 3),
                fmt_float((float(x["elapsed_s"]) / 60.0) if x.get("elapsed_s") else None, 1),
            ] for x in top_done],
        ))

    if by_expert:
        lines.extend(["", "## Expert Reliability Snapshot", ""])
        lines.extend(render_table(
            ["Expert", "N", "Mean P", "Empirical Pass", "Brier", "Log Score"],
            [[
                row["expert"], row["n"], fmt_float(row["mean_probability"], 3),
                fmt_pct(row["empirical_pass_rate"]), fmt_float(row["mean_brier"], 4), fmt_float(row["mean_log"], 4),
            ] for row in by_expert],
        ))
        lines.append("")
        lines.append("Note: early reliability is noisy until there are enough passed and failed samples per module/task.")

    if running:
        stage_counts = Counter(x.get("latest_event") or "unknown" for x in running)
        epoch_rows = sorted([x for x in running if x.get("epoch") is not None], key=lambda x: x["sample_id"])[:40]
        lines.extend(["", "## Running Snapshot", ""])
        lines.append("- Latest event counts: " + ", ".join(f"`{k}`={v}" for k, v in sorted(stage_counts.items())))
        if epoch_rows:
            lines.extend(["", *render_table(
                ["Sample", "Epoch", "Latest Acc", "Latest Event"],
                [[x["sample_id"], x.get("epoch"), fmt_float(x.get("latest_acc"), 2), x.get("latest_event")] for x in epoch_rows],
            )])

    if errors:
        lines.extend(["", "## Errors To Inspect", ""])
        lines.extend(render_table(
            ["Sample", "Error"],
            [[x["sample_id"], " ".join(str(x.get("error") or "")[:240].split())] for x in errors[:30]],
        ))

    lines.extend(["", "## Suggested Next Checks", ""])
    lines.extend([
        "- Confirm whether first-batch results are mostly `empty_worker_only`; do not compare modules until at least one non-control module has enough completed samples.",
        "- Watch for repeated API retry failures or worker timeout/background failures.",
        "- Once two or more modules have completed cells on the same task, compare expert probabilities against realized pass/fail by module.",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    latest = out_path.parent / "first_batch_latest.md"
    latest.write_text(out_path.read_text())
    summary = {
        "generated_at": utc_now(),
        "root": str(root),
        "out": str(out_path),
        "latest": str(latest),
        "samples": len(samples),
        "done": len(done),
        "running": len(running),
        "errors": len(errors),
    }
    (out_path.parent / "first_batch_latest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Render V2.3 first-batch analysis report.")
    p.add_argument("--root", default="autoresearch_idea_harness/runs/formal_sweeps/v2_3_mls10_modules9")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = Path(args.root)
    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = root / "analysis_reports" / f"first_batch_{stamp}.md"
    print(json.dumps(render_report(root, out_path), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

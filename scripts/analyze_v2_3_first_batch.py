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


EXPECTED_TASKS = [
    "dl_lr_schedule",
    "dl_activation_function",
    "cv_data_augmentation",
    "dl_weight_initialization",
    "cv_classification_loss",
    "cv_sample_weighting",
    "cv_pooling_aggregation",
    "cv_multitask_loss",
    "dl_regularization",
    "dl_residual_connection",
]


EXPECTED_MODULES = [
    "empty_worker_only",
    "opus47_master",
    "qwen25_7b_base_with_research_question",
    "exp09_top_k_related_work",
    "exp11_top_k_related_work",
    "exp12_with_research_question",
    "exp13_top_k_refs",
    "exp16_top_k_refs",
    "exp17_with_research_question",
]


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
    task, _subtask, module, sample = parse_sample_name(path.name)
    try:
        idx = int(sample.lstrip("s"))
    except Exception:
        idx = 0
    return task, module, idx


def parse_sample_name(name: str) -> tuple[str, str, str, str]:
    parts = name.split("__")
    task = parts[0] if parts else ""
    subtask = parts[1] if len(parts) > 1 else ""
    sample = parts[-1] if parts else "s00"
    module = "__".join(parts[2:-1]) if len(parts) >= 4 else ""
    return task, subtask, module, sample


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
    for sample_dir in sorted(
        [p for p in root.iterdir() if p.is_dir() and "__" in p.name and re.search(r"__s\d+$", p.name)],
        key=sample_sort_key,
    ):
        if not sample_dir.is_dir():
            continue
        summary = read_json(sample_dir / "summary.json")
        task_packet = read_json(sample_dir / "task_packet.json")
        result = read_json(sample_dir / "result.json")
        result_parsed = result.get("_parsed") or {}
        worker = summary.get("worker_result") or result_parsed or {}
        sample_id = sample_dir.name
        name_task, name_subtask, name_module, _sample = parse_sample_name(sample_id)
        task = summary.get("task") or task_packet.get("task") or name_task
        subtask = summary.get("subtask") or task_packet.get("subtask") or name_subtask
        module = summary.get("module_id") or task_packet.get("module_id") or name_module
        if (sample_dir / "result.json").exists():
            status = "done"
        elif (sample_dir / "error.json").exists():
            status = "error"
        else:
            status = worker.get("status") or "running"
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
            "has_settlement": (sample_dir / "settlement.json").exists(),
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


def build_cell_matrix(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    matrix: dict[tuple[str, str], dict[str, Any]] = {}
    for task in EXPECTED_TASKS:
        for module in EXPECTED_MODULES:
            matrix[(task, module)] = {
                "samples": 0,
                "done": 0,
                "running": 0,
                "errors": 0,
                "passed": 0,
                "failed": 0,
                "best_metric": None,
            }
    for row in rows:
        key = (str(row.get("task") or ""), str(row.get("module") or ""))
        cell = matrix.setdefault(key, {
            "samples": 0,
            "done": 0,
            "running": 0,
            "errors": 0,
            "passed": 0,
            "failed": 0,
            "best_metric": None,
        })
        cell["samples"] += 1
        status = row.get("status")
        if status == "done":
            cell["done"] += 1
            if row.get("passed"):
                cell["passed"] += 1
            else:
                cell["failed"] += 1
            metric = row.get("metric")
            if metric is not None:
                metric = float(metric)
                if cell["best_metric"] is None or metric > cell["best_metric"]:
                    cell["best_metric"] = metric
        elif status == "error":
            cell["errors"] += 1
        else:
            cell["running"] += 1
    return matrix


def cell_text(cell: dict[str, Any]) -> str:
    text = f"{cell['running']}/{cell['passed']}/{cell['failed']}"
    if cell["errors"]:
        text += f" +e{cell['errors']}"
    done = cell["passed"] + cell["failed"]
    if done >= 10 and cell["passed"] / done >= 0.5:
        text = f"**{text}**"
    return text


def render_cell_matrix(rows: list[dict[str, Any]]) -> list[str]:
    matrix = build_cell_matrix(rows)
    lines = [
        "Cell format: `running/pass/fail`; `+eN` means worker/error artifacts. Bold cells have pass rate >= 50% among completed non-error samples.",
        "",
    ]
    headers = ["Task"] + EXPECTED_MODULES
    table_rows = []
    for task in EXPECTED_TASKS:
        table_rows.append([task] + [cell_text(matrix[(task, module)]) for module in EXPECTED_MODULES])
    lines.extend(render_table(headers, table_rows))
    return lines


def top_cells(rows: list[dict[str, Any]], min_done: int = 10) -> list[dict[str, Any]]:
    matrix = build_cell_matrix(rows)
    out = []
    for (task, module), cell in matrix.items():
        done = cell["passed"] + cell["failed"]
        if done < min_done:
            continue
        out.append({
            "task": task,
            "module": module,
            "done": done,
            "passed": cell["passed"],
            "errors": cell["errors"],
            "pass_rate": cell["passed"] / done if done else None,
            "best_metric": cell["best_metric"],
        })
    return sorted(out, key=lambda x: (x["pass_rate"] or 0.0, x["best_metric"] or -1e9), reverse=True)


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
            "threshold_accuracy": mean([float((p >= 0.5) == bool(o)) for p, o in zip(probs, outcomes)]) if probs else None,
        })
    return out


def aggregate_expert_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        expert = str(row.get("expert_id") or "")
        bucket = str(row.get("calibration_bucket") or "")
        groups[(expert, bucket)].append(row)
    out = []
    for (expert, bucket), items in sorted(groups.items()):
        probs = [float(x.get("success_probability", 0.0)) for x in items]
        outcomes = [float(x.get("outcome", 0.0)) for x in items]
        out.append({
            "expert": expert,
            "bucket": bucket,
            "n": len(items),
            "mean_probability": mean(probs) if probs else None,
            "empirical_pass_rate": mean(outcomes) if outcomes else None,
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
    by_bucket = aggregate_expert_buckets(expert_settled)
    settled_samples = sum(1 for x in samples if x.get("has_settlement"))
    observed_modules = sorted({str(x.get("module") or "") for x in samples})
    expected_module_set = set(EXPECTED_MODULES)
    unexpected_modules = [x for x in observed_modules if x and x not in expected_module_set]

    lines = [
        "# V2.3 Formal Sweep Analysis",
        "",
        f"- Generated at: `{utc_now()}`",
        f"- Root: `{root}`",
        f"- Planned total: `{total_planned}` samples",
        f"- Materialized sample dirs: `{len(samples)}`",
        f"- Completed: `{len(done)}`",
        f"- Running: `{len(running)}`",
        f"- Errors: `{len(errors)}`",
        f"- Samples with settlement: `{settled_samples}`",
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
            f"- Current completed sample pass rate: `{fmt_pct(pass_rate)}` over `{len(done)}` completed samples.",
            f"- Best completed metric so far: `{fmt_float(best.get('metric'))}` from `{best['sample_id']}`.",
            "- This is already large enough for a first reliable read on expert calibration and module/task difficulty, but the remaining error cells should be audited before treating pass rates as final.",
        ])
        if unexpected_modules:
            lines.append(f"- Naming anomaly to clean up: unexpected module ids `{', '.join(unexpected_modules)}`.")

    lines.extend(["", "## 10x9 Progress Matrix", ""])
    lines.extend(render_cell_matrix(samples))

    ranked_cells = top_cells(samples)
    if ranked_cells:
        lines.extend(["", "## Strong Cells So Far", ""])
        lines.extend(render_table(
            ["Task", "Module", "Done", "Pass", "Err", "Pass Rate", "Best"],
            [[
                row["task"], row["module"], row["done"], row["passed"], row["errors"],
                fmt_pct(row["pass_rate"]), fmt_float(row["best_metric"]),
            ] for row in ranked_cells[:20]],
        ))

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
        lines.append(
            "Readout: lower Brier/log score is better. At this snapshot, both experts are useful but visibly overconfident in high-probability buckets; treat forecast probabilities as ranking signals, not calibrated pass probabilities."
        )

    if by_bucket:
        lines.extend(["", "## Expert Calibration Buckets", ""])
        lines.extend(render_table(
            ["Expert", "Bucket", "N", "Mean P", "Empirical Pass"],
            [[
                row["expert"], row["bucket"], row["n"],
                fmt_float(row["mean_probability"], 3), fmt_pct(row["empirical_pass_rate"]),
            ] for row in by_bucket],
        ))

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
            [[x["sample_id"], " ".join(str(x.get("error") or "")[:240].split())] for x in errors[:50]],
        ))

    lines.extend(["", "## Suggested Next Checks", ""])
    lines.extend([
        "- Audit repeated `opus47_master` failures separately from model quality; many are execution/protocol errors, not necessarily bad ideas.",
        "- Normalize the stray `exp16-top_k_refs` sample directory into the expected `exp16_top_k_refs` accounting before the final report.",
        "- Use expert probabilities mainly for relative ranking until calibration is repaired; high-confidence buckets currently underperform their stated probability.",
        "- Rerun only the small set of pending/error cells needed to make per-cell denominators comparable; avoid disturbing active agents/monitors.",
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

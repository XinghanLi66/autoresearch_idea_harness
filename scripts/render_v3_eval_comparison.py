#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "runs" / "v3_precomputed_worker_eval" / "dlc_mls10_qwen25_32b_base_v1sem"
DEFAULT_BASE_EXTRAS = [
    ROOT / "runs" / "v3_precomputed_worker_eval" / "dlc_mls10_qwen25_32b_base_v1sem_supp_pending2",
]
DEFAULT_SFT = ROOT / "runs" / "v3_precomputed_worker_eval" / "dlc_mls10_v1sem_cot_sft_16k_guarded"
DEFAULT_SFT_EXTRAS = [
    ROOT / "runs" / "v3_precomputed_worker_eval" / "dlc_mls10_v1sem_cot_sft_16k_guarded_reg_retry",
]
DEFAULT_OUTPUT = ROOT / "runs" / "reports" / "v3_base_vs_v1sem_cot_sft_eval.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"_load_error": str(exc)}


def load_plan(run_root: Path) -> dict[str, Any]:
    return load_json(run_root / "run_plan.json")


def planned_tasks(run_root: Path) -> list[dict[str, Any]]:
    plan = load_plan(run_root)
    selected = plan.get("selected") or []
    rows = []
    for item in selected:
        rows.append({
            "task": item.get("task"),
            "subtask": item.get("subtask"),
            "module_id": plan.get("module_id") or item.get("module_id"),
            "proposal_id": item.get("proposal_id") or item.get("id"),
            "source_quality_verdict": (item.get("quality") or {}).get("verdict"),
            "source_quality_score": (item.get("quality") or {}).get("score"),
        })
    return rows


def metric_passed(packet: dict[str, Any], val_metric: Any) -> bool | None:
    if val_metric is None:
        return None
    pass_metric = packet.get("pass_metric")
    if pass_metric is None:
        return None
    try:
        val = float(val_metric)
        threshold = float(pass_metric)
    except Exception:
        return None
    if bool(packet.get("lower_is_better")):
        return val <= threshold
    return val >= threshold


def metric_improvement(packet: dict[str, Any], val_metric: Any) -> float | None:
    baseline = packet.get("baseline_metric")
    if val_metric is None or baseline is None:
        return None
    try:
        val = float(val_metric)
        base = float(baseline)
    except Exception:
        return None
    return base - val if bool(packet.get("lower_is_better")) else val - base


def collect_results(run_root: Path) -> dict[str, Any]:
    output_root = run_root / "output"
    planned = planned_tasks(run_root)
    by_task: dict[tuple[str, str], dict[str, Any]] = {}
    for item in planned:
        by_task[(str(item.get("task")), str(item.get("subtask")))] = dict(item)
    for summary_path in sorted(output_root.glob("*/summary.json")):
        summary = load_json(summary_path)
        worker = summary.get("worker_result") or {}
        key = (str(summary.get("task")), str(summary.get("subtask")))
        row = by_task.setdefault(key, {"task": key[0], "subtask": key[1]})
        row.update({
            "sample_dir": str(summary_path.parent),
            "summary_exists": True,
            "module_id": summary.get("module_id") or row.get("module_id"),
            "proposal_id": summary.get("proposal_id") or row.get("proposal_id"),
            "baseline_metric": summary.get("baseline_metric"),
            "pass_metric": summary.get("pass_metric"),
            "source_quality_verdict": summary.get("source_quality_verdict") or row.get("source_quality_verdict"),
            "source_quality_score": summary.get("source_quality_score") or row.get("source_quality_score"),
            "worker_status": worker.get("status"),
            "val_metric": worker.get("val_metric"),
            "improvement": worker.get("improvement"),
            "passed": worker.get("passed"),
            "elapsed_s": worker.get("elapsed_s"),
            "error": worker.get("error"),
        })
    for sample_dir in sorted(output_root.glob("*")):
        if not sample_dir.is_dir() or (sample_dir / "summary.json").exists():
            continue
        result = load_json(sample_dir / "result.json")
        workspace_result = load_json(sample_dir / "workspace" / "result.json")
        error = load_json(sample_dir / "error.json")
        packet = load_json(sample_dir / "task_packet.json")
        key = (str(packet.get("task") or sample_dir.name.split("__")[0]), str(packet.get("subtask") or ""))
        row = by_task.setdefault(key, {"task": key[0], "subtask": key[1]})
        row.setdefault("sample_dir", str(sample_dir))
        row["summary_exists"] = False
        row.setdefault("baseline_metric", packet.get("baseline_metric"))
        row.setdefault("pass_metric", packet.get("pass_metric"))
        if result:
            parsed = result.get("_parsed") or {}
            row.update({
                "worker_status": parsed.get("status") or "done_without_summary",
                "val_metric": parsed.get("val_metric") or result.get("val_metric"),
                "improvement": parsed.get("improvement"),
                "passed": parsed.get("passed"),
                "error": parsed.get("error"),
            })
        elif workspace_result:
            val_metric = workspace_result.get("val_metric")
            if val_metric is None and workspace_result.get("error"):
                row.update({
                    "worker_status": "error",
                    "error": workspace_result.get("error"),
                    "signed_workspace_result": True,
                })
            else:
                row.update({
                    "worker_status": "done_pending_summary",
                    "val_metric": val_metric,
                    "improvement": metric_improvement(packet, val_metric),
                    "passed": metric_passed(packet, val_metric),
                    "signed_workspace_result": True,
                })
        if error:
            row.update({"worker_status": "error", "error": error.get("error") or error})
    rows = sorted(by_task.values(), key=lambda row: (str(row.get("task")), str(row.get("subtask"))))
    completed = [row for row in rows if row.get("passed") is not None]
    passed = [row for row in completed if row.get("passed") is True]
    errors = [row for row in rows if row.get("worker_status") == "error" or row.get("error")]
    planned_count = max(len(planned), len(rows))
    return {
        "run_root": str(run_root),
        "exists": run_root.exists(),
        "planned_count": planned_count,
        "row_count": len(rows),
        "completed_count": len(completed),
        "passed_count": len(passed),
        "error_count": len(errors),
        "pass_rate_completed": (len(passed) / len(completed)) if completed else None,
        "pass_rate_planned_conservative": (len(passed) / planned_count) if planned_count else None,
        "rows": rows,
        "plan": load_plan(run_root),
        "submission": load_json(run_root / "submission.json"),
        "job_status": load_json(run_root / "worker_eval_job_status.json"),
    }


def row_completeness(row: dict[str, Any]) -> int:
    if row.get("worker_status") == "error" or row.get("error"):
        return 3
    if row.get("passed") is not None:
        return 3
    if row.get("val_metric") is not None:
        return 2
    if row.get("worker_status"):
        return 1
    return 0


def recompute_counts(run: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(run.get("rows") or [], key=lambda row: (str(row.get("task")), str(row.get("subtask"))))
    completed = [row for row in rows if row.get("passed") is not None]
    passed = [row for row in completed if row.get("passed") is True]
    errors = [row for row in rows if row.get("worker_status") == "error" or row.get("error")]
    planned_count = max(int(run.get("planned_count") or 0), len(rows))
    run.update({
        "row_count": len(rows),
        "completed_count": len(completed),
        "passed_count": len(passed),
        "error_count": len(errors),
        "pass_rate_completed": (len(passed) / len(completed)) if completed else None,
        "pass_rate_planned_conservative": (len(passed) / planned_count) if planned_count else None,
        "planned_count": planned_count,
        "rows": rows,
    })
    return run


def collect_merged_results(primary_run: Path, extra_runs: list[Path]) -> dict[str, Any]:
    merged = collect_results(primary_run)
    merged["merged_run_roots"] = [str(primary_run)]
    by_key = {
        (str(row.get("task")), str(row.get("subtask"))): row
        for row in merged.get("rows") or []
    }
    for extra in extra_runs:
        if not extra.exists():
            continue
        extra_payload = collect_results(extra)
        merged["merged_run_roots"].append(str(extra))
        merged["planned_count"] = max(int(merged.get("planned_count") or 0), int(extra_payload.get("planned_count") or 0))
        for row in extra_payload.get("rows") or []:
            key = (str(row.get("task")), str(row.get("subtask")))
            old = by_key.get(key)
            if old is None or row_completeness(row) > row_completeness(old):
                copied = dict(row)
                copied["merged_source_run_root"] = str(extra)
                by_key[key] = copied
    merged["rows"] = list(by_key.values())
    return recompute_counts(merged)


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V3 Base vs V1-Semantics CoT SFT Eval",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "This report separates completed pass rate from a conservative planned pass rate where unfinished samples count as not passed.",
        "",
        "## Summary",
        "",
        "| Run | Planned | Completed | Passed | Errors | Pass rate completed | Pass rate conservative |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in payload["runs"].items():
        lines.append(
            f"| {name} | {run['planned_count']} | {run['completed_count']} | {run['passed_count']} | "
            f"{run['error_count']} | {fmt_rate(run['pass_rate_completed'])} | "
            f"{fmt_rate(run['pass_rate_planned_conservative'])} |"
        )
    lines.extend(["", "## Task Matrix", ""])
    all_keys = sorted({
        (row.get("task"), row.get("subtask"))
        for run in payload["runs"].values()
        for row in run.get("rows") or []
    })
    header = "| Task | Subtask | " + " | ".join(f"{name} metric/pass/status" for name in payload["runs"]) + " |"
    sep = "|---|---|" + "".join("---|" for _ in payload["runs"])
    lines.extend([header, sep])
    for task, subtask in all_keys:
        cells = []
        for run in payload["runs"].values():
            row = next((r for r in run.get("rows") or [] if r.get("task") == task and r.get("subtask") == subtask), None)
            if not row:
                cells.append("missing")
                continue
            passed = row.get("passed")
            verdict = "pass" if passed is True else "fail" if passed is False else str(row.get("worker_status") or "running")
            cells.append(
                f"{fmt_metric(row.get('val_metric'))}/{fmt_metric(row.get('pass_metric'))}/{verdict}"
            )
        lines.append(f"| {task} | {subtask} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Run Roots", ""])
    for name, run in payload["runs"].items():
        submission = run.get("submission") or {}
        lines.append(
            f"- `{name}`: `{run['run_root']}` job={submission.get('job_id') or '-'} "
            f"result={submission.get('result') or (submission.get('latest_status') or {}).get('status') or '-'}"
        )
        for extra_root in run.get("merged_run_roots") or []:
            if extra_root != run.get("run_root"):
                lines.append(f"  - merged extra: `{extra_root}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare V3 base 32B and V1-semantics CoT SFT worker eval runs.")
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-extra-run", type=Path, action="append", default=None)
    parser.add_argument("--sft-run", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--sft-extra-run", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": {
            "base_32b": collect_merged_results(args.base_run, args.base_extra_run if args.base_extra_run is not None else DEFAULT_BASE_EXTRAS),
            "v1sem_cot_sft": collect_merged_results(args.sft_run, args.sft_extra_run if args.sft_extra_run is not None else DEFAULT_SFT_EXTRAS),
        },
    }
    json_output = args.json_output or args.output.with_suffix(".json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(payload))
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "json_output": str(json_output)}, indent=2))


if __name__ == "__main__":
    main()

"""
Run all baselines on the trace-backed harness and collect results.
Replaces the VideoTemporalKVPolicy class in custom_template.py with
the one from each baseline edit.py, runs the evaluator, and writes
leaderboard_trace_v1.csv.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
TEMPLATE = TASK_DIR / "edits" / "custom_template.py"
EVAL_PY = TASK_DIR / "custom_video_eval.py"
OUT_CSV = TASK_DIR / "leaderboard_trace_v1.csv"

BASELINES = ["recent_window", "packcache", "causalcache_vdm", "flowcache"]

WORKLOADS = {
    "ucf101_short_prediction": "medium_history_budget",
    "dmlab_long_prediction": "tight_history_budget",
    "minecraft_scene_cut": "tight_history_budget",
}


def apply_edit(name: str) -> None:
    """Apply a baseline edit by exec-ing the edit file to get _POLICY, then line-replacing."""
    template_lines = TEMPLATE.read_text(encoding="utf-8").splitlines(keepends=True)
    edit_path = TASK_DIR / "edits" / f"{name}.edit.py"

    # exec the edit file in a sandbox to get OPS / _POLICY without importing anything
    ns: dict = {}
    exec(compile(edit_path.read_text(encoding="utf-8"), str(edit_path), "exec"), ns)

    ops = ns.get("OPS", [])
    if not ops:
        raise ValueError(f"No OPS found in {name}.edit.py")

    result_lines = list(template_lines)
    for op in ops:
        if op.get("op") != "replace":
            continue
        start = op["start_line"] - 1   # 1-indexed → 0-indexed
        end = op["end_line"]            # exclusive end
        content = op["content"]
        replacement = [line + ("\n" if not line.endswith("\n") else "") for line in content.splitlines()]
        # Preserve trailing newline from original block
        if result_lines[end - 1].endswith("\n"):
            if replacement and not replacement[-1].endswith("\n"):
                replacement[-1] += "\n"
        result_lines[start:end] = replacement

    EVAL_PY.write_text("".join(result_lines), encoding="utf-8")
    print(f"  Applied edit: {name} (OPS lines {ops[0].get('start_line')}-{ops[0].get('end_line')})", flush=True)


def run_eval(workload: str, budget: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(EVAL_PY), "--workload", workload, "--budget", budget, "--seed", "42"],
        capture_output=True,
        text=True,
        cwd=str(TASK_DIR),
    )
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-400:]}", flush=True)
        return {}
    for line in result.stdout.splitlines():
        if line.startswith("TEST_METRICS:"):
            metrics_str = line[len("TEST_METRICS:"):]
            metrics: dict = {}
            for part in metrics_str.split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        metrics[k] = v
            return metrics
    print(f"    WARNING: no TEST_METRICS line found", flush=True)
    return {}


def main():
    rows = []
    header = "baseline,workload,budget,temporal_quality_main,peak_kv_memory_mb,fps,drift_score,scene_retention_precision,trace_mode"

    for baseline in BASELINES:
        print(f"\n=== Baseline: {baseline} ===", flush=True)
        try:
            apply_edit(baseline)
        except Exception as e:
            print(f"  FAILED to apply edit: {e}", flush=True)
            continue

        for workload, budget in WORKLOADS.items():
            print(f"  -> {workload} / {budget}", flush=True)
            metrics = run_eval(workload, budget)
            if not metrics:
                print(f"    FAILED (no metrics)", flush=True)
                continue
            tq = round(metrics.get("temporal_quality_main", 0), 4)
            mem = round(metrics.get("peak_kv_memory_mb", 0), 3)
            fps = round(metrics.get("fps", 0), 4)
            drift = round(metrics.get("drift_score", 0), 4)
            srp = round(metrics.get("scene_retention_precision", 0), 4)
            print(f"    tq={tq}  mem={mem}  fps={fps}  drift={drift}  srp={srp}", flush=True)
            rows.append(f"{baseline},{workload},{budget},{tq},{mem},{fps},{drift},{srp},video_chunk_trace")

    print(f"\n=== Writing {OUT_CSV} ===", flush=True)
    OUT_CSV.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")
    print(OUT_CSV.read_text(), flush=True)

    # Restore template as custom_video_eval.py
    import shutil
    shutil.copy(TEMPLATE, EVAL_PY)
    print("Restored custom_video_eval.py from template.", flush=True)


if __name__ == "__main__":
    main()

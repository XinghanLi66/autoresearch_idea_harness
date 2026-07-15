#!/usr/bin/env python3
"""Closed-loop MLS eval driver (step 2): run the V2 worker on precomputed proposals, all arms/tasks.

For each arm's proposals_<arm>.jsonl, for each of the 10 MLS tasks, invoke
run_precomputed_proposal_worker.py (worker = Claude Code implements the proposal, runs the task's
run.sh, returns a signed metric + pass/fail). Runs sequentially on the single local GPU. Aggregates
pass counts + per-task metric vs threshold into a comparison table (vs recorded V2.5 baselines).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python"


def load_proposals(path: Path) -> dict[str, dict]:
    return {r["task"]: r for r in (json.loads(l) for l in path.open())}


def run_one(arm: str, rec: dict, out_root: Path, max_turns: int, timeout: int, gpu: str) -> dict:
    task = rec["task"]
    prop_file = out_root / f"_proposal_{arm}_{task}.txt"
    prop_file.write_text(rec["proposal"])
    sample_dir = out_root / arm / task
    cmd = [PY, str(ROOT / "scripts" / "run_precomputed_proposal_worker.py"),
           "--task", task, "--proposal", str(prop_file),
           "--output-root", str(out_root / arm),
           "--module-id", f"mls_eval_{arm}", "--model-id", f"qwen25_32b_{arm}",
           "--worker-mode", "claude", "--gpu", gpu,
           "--max-turns", str(max_turns), "--worker-timeout", str(timeout),
           "--result-wait-timeout", str(timeout), "--sample-id", f"{arm}_{task}", "--force"]
    if rec.get("subtask"):
        cmd += ["--subtask", rec["subtask"]]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 600)
    elapsed = round(time.time() - t0, 1)
    # the worker script writes the authoritative verdict to <sample>/summary.json under worker_result
    passed = metric = status = improvement = baseline = None
    sdir = out_root / arm / f"mls_eval_{arm}"
    cands = sorted((out_root / arm).rglob("summary.json"))
    for sd in cands:
        try:
            d = json.loads(sd.read_text())
        except Exception:
            continue
        if d.get("task") != task or "worker_result" not in d:
            continue
        wr = d["worker_result"]
        passed = wr.get("passed")
        metric = wr.get("val_metric")
        status = wr.get("status")
        improvement = wr.get("improvement")
        baseline = d.get("baseline_metric")
    return {"arm": arm, "task": task, "pass_metric": rec.get("pass_metric"),
            "passed": passed, "metric": metric, "improvement": improvement,
            "baseline_metric": baseline, "status": status,
            "elapsed_s": elapsed, "rc": p.returncode, "stderr_tail": p.stderr[-300:]}


def parse_existing(arm: str, task: str, out_root: Path, pass_metric) -> dict | None:
    """Re-derive a result row from an existing worker summary.json (no re-run)."""
    for sd in sorted((out_root / arm).rglob("summary.json")):
        try:
            d = json.loads(sd.read_text())
        except Exception:
            continue
        if d.get("task") == task and "worker_result" in d:
            wr = d["worker_result"]
            return {"arm": arm, "task": task, "pass_metric": pass_metric,
                    "passed": wr.get("passed"), "metric": wr.get("val_metric"),
                    "improvement": wr.get("improvement"), "baseline_metric": d.get("baseline_metric"),
                    "status": wr.get("status"), "cached": True}
    return None
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposals-dir", type=Path, required=True, help="dir with proposals_<arm>.jsonl")
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs" / "researcher_cot" / "mls_eval" / "worker_runs")
    ap.add_argument("--arms", default="base,sft,rl")
    ap.add_argument("--tasks", default=None, help="comma list to subset (default: all in proposals)")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--worker-timeout", type=int, default=7200)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    results_path = args.out_root / "results.jsonl"

    arms = args.arms.split(",")
    rows = []
    for arm in arms:
        props = load_proposals(args.proposals_dir / f"proposals_{arm}.jsonl")
        tasks = args.tasks.split(",") if args.tasks else list(props)
        for task in tasks:
            existing = parse_existing(arm, task, args.out_root, props[task].get("pass_metric"))
            if existing:
                print(f"[eval] cached {arm}/{task}: passed={existing['passed']} metric={existing['metric']}", flush=True)
                rows.append(existing)
                continue
            print(f"[eval] START {arm}/{task}", flush=True)
            r = run_one(arm, props[task], args.out_root, args.max_turns, args.worker_timeout, args.gpu)
            rows.append(r)
            print(f"[eval] DONE {arm}/{task}: passed={r['passed']} metric={r['metric']} "
                  f"vs {r['pass_metric']} status={r['status']} {r['elapsed_s']}s", flush=True)
            with results_path.open("w") as f:  # rewrite full snapshot after each run
                for rr in rows:
                    f.write(json.dumps(rr, ensure_ascii=False) + "\n")

    with results_path.open("w") as f:
        for rr in rows:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    summary = {}
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"n": len(ar), "passed": sum(1 for r in ar if r["passed"]),
                        "completed": sum(1 for r in ar if r["metric"] is not None)}
    (args.out_root / "summary.json").write_text(json.dumps(
        {"per_arm": summary,
         "baseline_v25": {"base_32b": "1/10", "old_sft": "0/10"},
         "rows": rows}, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", "per_arm": summary}), flush=True)


if __name__ == "__main__":
    main()

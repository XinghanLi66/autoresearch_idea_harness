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
    # find the settlement written by the worker script
    passed = metric = status = None
    for sd in sorted((out_root / arm).rglob("settlement.json")) + sorted((out_root / arm).rglob("result.json")):
        if task in str(sd) or f"{arm}_{task}" in str(sd):
            try:
                d = json.loads(sd.read_text())
                passed = d.get("passed", passed)
                metric = d.get("val_metric", d.get("metric", metric))
                status = d.get("status", status)
            except Exception:
                pass
    return {"arm": arm, "task": task, "pass_metric": rec.get("pass_metric"),
            "passed": passed, "metric": metric, "status": status,
            "elapsed_s": elapsed, "rc": p.returncode, "stderr_tail": p.stderr[-300:]}


def main() -> None:
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
    done = set()
    if results_path.exists():
        for l in results_path.open():
            try:
                r = json.loads(l)
                done.add((r["arm"], r["task"]))
            except Exception:
                pass

    arms = args.arms.split(",")
    for arm in arms:
        props = load_proposals(args.proposals_dir / f"proposals_{arm}.jsonl")
        tasks = args.tasks.split(",") if args.tasks else list(props)
        for task in tasks:
            if (arm, task) in done:
                print(f"[eval] skip {arm}/{task} (cached)", flush=True)
                continue
            print(f"[eval] START {arm}/{task}", flush=True)
            r = run_one(arm, props[task], args.out_root, args.max_turns, args.worker_timeout, args.gpu)
            with results_path.open("a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[eval] DONE {arm}/{task}: passed={r['passed']} metric={r['metric']} "
                  f"vs {r['pass_metric']} status={r['status']} {r['elapsed_s']}s", flush=True)

    # aggregate
    rows = [json.loads(l) for l in results_path.open()]
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

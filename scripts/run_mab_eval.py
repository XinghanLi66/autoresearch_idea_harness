#!/usr/bin/env python3
"""MLAgentBench (MAB) eval driver — the single benchmarking entrypoint.

Two-stage closed loop, mirroring run_mls_lite_eval.py:
  1. baseline:  run the unmodified train.py per (task, seed)          (proposal-independent reference)
  2. eval:      the fable-5 worker IMPLEMENTS each arm's proposal into train.py, runs it, scores it
Metrics are read back from our own results/<task>.csv sink (MAB has no leaderboard). mab_report.py then
computes the direction-signed Δ-over-baseline per (arm, task), aggregated across seeds.

Proposals come from gen_proposals_from_endpoint.py as proposals_<arm>.jsonl (records keyed by task slug).
Task set = the 5 in docs/eval/mab_tasks.json.

The MAB eval env (conda env `mab`, prepared task data) is provisioned by cc002. Use --dry-run to print the
exact dispatch/worker commands first.

Usage:
  python scripts/run_mab_eval.py --arms base,sft,rl --seeds 3 --dispatch cluster --dry-run
  python scripts/run_mab_eval.py --arms base,rl --seeds 1 --dispatch local --dry-run
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _mab_common as mab  # noqa: E402


def run_cluster(args, slugs: list[str], arms: list[str], seeds: list[int]) -> None:
    """Fan the same work items out as PAI DLC jobs via mab_cluster_dispatch.py."""
    from datetime import datetime
    dispatcher = str(ROOT / "scripts" / "mab_cluster_dispatch.py")
    batch = args.batch or f"mab_{datetime.now().strftime('%m%d_%H%M%S')}"
    common = [sys.executable, dispatcher]
    tasks_csv = ",".join(slugs)
    seeds_csv = ",".join(str(s) for s in seeds)

    def run(cmd: list[str]) -> None:
        if args.dry_run:
            cmd = cmd + ["--dry-run"]
        print("  $", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"dispatcher step failed: {' '.join(cmd)}")

    if not args.skip_baseline:
        run(common + ["submit-batch", "--kind", "baseline", "--batch", batch,
                      "--tasks", tasks_csv, "--seeds", seeds_csv,
                      "--max-concurrent", str(args.max_concurrent)])
    run(common + ["submit-batch", "--kind", "agent", "--batch", batch,
                  "--tasks", tasks_csv, "--seeds", seeds_csv,
                  "--arms", ",".join(arms), "--proposals-dir", args.proposals_dir,
                  "--worker-model", args.worker_model,
                  "--hparam-suffix", args.hparam_suffix,
                  "--max-concurrent", str(args.max_concurrent)])
    if not args.dry_run:
        run(common + ["status", "--batch", batch, "--watch"])
        print(f"== cluster batch '{batch}' finished; run mab_report.py to aggregate ==")


def run_local(args, slugs: list[str], arms: list[str], seeds: list[int]) -> None:
    """Serial in-process runs on this machine via mab_worker_edit.py (needs the shim + `mab` env)."""
    worker = str(ROOT / "scripts" / "mab_worker_edit.py")

    def run(cmd: list[str]) -> None:
        print("  $", " ".join(cmd), flush=True)
        if not args.dry_run and subprocess.run(cmd).returncode != 0:
            print(f"[warn] worker step failed (continuing): {' '.join(cmd)}", flush=True)

    if not args.skip_baseline:
        print("== baselines ==")
        for slug in slugs:
            for s in seeds:
                run([sys.executable, worker, "--task", slug, "--arm", "baseline", "--seed", str(s)])

    for arm in arms:
        pfile = Path(args.proposals_dir) / f"proposals_{arm}.jsonl"
        props = {}
        if pfile.exists():
            for line in pfile.open():
                r = json.loads(line)
                props[r["task"]] = r.get("proposal") or r.get("text") or ""
        elif not args.dry_run:
            sys.exit(f"missing {pfile}")
        print(f"== arm={arm} ==")
        pdir = Path(args.proposals_dir) / "_local_md" / arm
        pdir.mkdir(parents=True, exist_ok=True)
        for slug in slugs:
            md = pdir / f"{slug}.md"
            if not args.dry_run:
                md.write_text(props.get(slug, "") + args.hparam_suffix)
            for s in seeds:
                run([sys.executable, worker, "--task", slug, "--arm", arm, "--seed", str(s),
                     "--proposal", str(md), "--worker-model", args.worker_model])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-json", default=str(ROOT / "docs/eval/mab_tasks.json"))
    ap.add_argument("--proposals-dir", default=str(ROOT / "runs/researcher_cot/mab_eval/proposals"))
    ap.add_argument("--arms", default="base,sft,rl")
    ap.add_argument("--tasks", default=None, help="comma slug subset (default: all 5)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--worker-model", default="claude-fable-5")
    ap.add_argument("--hparam-suffix", default="", help="text appended to each proposal ctx")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dispatch", default="cluster", choices=["local", "cluster"],
                    help="cluster = fan out as PAI DLC jobs (mab_cluster_dispatch.py); "
                         "local = serial on this machine (needs the shim + `mab` env)")
    ap.add_argument("--batch", default=None, help="cluster batch name (default: mab_<timestamp>)")
    ap.add_argument("--max-concurrent", type=int, default=64)
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks_json).read_text())
    slugs = args.tasks.split(",") if args.tasks else tasks["slugs"]
    arms = args.arms.split(",")
    seeds = list(range(1, args.seeds + 1))

    if args.dispatch == "cluster":
        run_cluster(args, slugs, arms, seeds)
    else:
        run_local(args, slugs, arms, seeds)


if __name__ == "__main__":
    main()

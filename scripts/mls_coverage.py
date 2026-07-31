#!/usr/bin/env python3
"""MLS coverage sweep — read every task's leaderboard from disk via native `mlsbench score`
and report, per task: #models scored, #models with task_score>0, and the max score.

Purpose: identify tasks where NO arm succeeds (max score == 0 across all models) vs tasks with
real signal. Pulls from disk only (no context/memory). Includes the 30th (humanoid) task so we
can track full-scope coverage.

Usage:
  python scripts/mls_coverage.py                 # all 30 (29 lite + humanoid)
  python scripts/mls_coverage.py --tasks a,b,c   # subset
  python scripts/mls_coverage.py --json          # machine-readable
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUMANOID = "robo-humanoid-sim2real-algo"

# The canonical V3 proposer-zoo 18 arms (bare arm tags, matched against claude-fable-5__<arm>).
ARMS_18 = [
    "base8b", "base14b", "base32b", "base32bq3",
    "sft8b", "sft14b", "sft32b", "rl",
    "d1base", "d1sft", "d1rl",
    "m2base", "m2sft", "m2rl",
    "qwen3-8b-rl", "qwen3-14b-rl", "qwen3-32b-rl", "qwen25sft",
]


def _arm_of(model: str) -> str | None:
    return model.split("__", 1)[1].split(":", 1)[0] if "__" in model else None


def score_task(mlsbench: str, task: str) -> list[dict]:
    try:
        out = subprocess.run([mlsbench, "score", task, "--format", "json"],
                             capture_output=True, text=True, timeout=180)
        if out.returncode != 0 or not out.stdout.strip():
            return []
        data = json.loads(out.stdout)
    except Exception as e:
        print(f"[warn] score {task}: {e}", file=sys.stderr)
        return []
    rows = data.get(task, data if isinstance(data, list) else [])
    return [r for r in rows if "task_score" in r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite-json", default=str(ROOT / "docs/eval/mls_bench_lite_tasks.json"))
    ap.add_argument("--mlsbench", default="mlsbench")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--matrix", action="store_true",
                    help="per-cell 18-arm x 30-task faithfulness grid (OK / '.' true-zero / 'x' MISSING)")
    ap.add_argument("--arms", default=None, help="comma arm subset for --matrix (default: canonical 18)")
    args = ap.parse_args()

    lite = json.loads(Path(args.lite_json).read_text())
    slugs = args.tasks.split(",") if args.tasks else (lite["slugs"] + ([HUMANOID] if HUMANOID not in lite["slugs"] else []))

    if args.matrix:
        arms = args.arms.split(",") if args.arms else ARMS_18
        # cell status per (arm, task): OK(>0), 'z'(scored true-zero), 'x'(missing/no row)
        cells: dict[str, dict[str, str]] = {a: {} for a in arms}
        for slug in slugs:
            rows = score_task(args.mlsbench, slug)
            by_arm = {}
            for r in rows:
                a = _arm_of(r["model"])
                if a in cells:
                    by_arm[a] = max(by_arm.get(a, -1.0), float(r["task_score"]))
            for a in arms:
                v = by_arm.get(a)
                cells[a][slug] = "x" if v is None else ("z" if v <= 0 else "OK")
        # print grid
        short = [s[:11] for s in slugs]
        print("arm\\task".ljust(14) + "".join(f"{s:>12}" for s in short))
        n_missing = n_zero = n_ok = 0
        for a in arms:
            line = a.ljust(14)
            for slug in slugs:
                c = cells[a][slug]
                line += f"{('OK' if c=='OK' else ('.' if c=='z' else 'x')):>12}"
                n_ok += c == "OK"; n_zero += c == "z"; n_missing += c == "x"
            print(line)
        total = len(arms) * len(slugs)
        print(f"\ncells: {n_ok} OK, {n_zero} true-zero(.), {n_missing} MISSING(x)  of {total}")
        # per-task missing/zero summary (the actionable gaps)
        for slug in slugs:
            miss = [a for a in arms if cells[a][slug] == "x"]
            zero = [a for a in arms if cells[a][slug] == "z"]
            if miss or zero:
                print(f"  {slug}: MISSING={len(miss)} zero={len(zero)}"
                      + (f" missing={','.join(miss)}" if 0 < len(miss) <= 6 else ""))
        return

    report, zero, no_data = [], [], []
    for slug in slugs:
        rows = score_task(args.mlsbench, slug)
        # "ours" = arm-tagged runs (claude-fable-5__<arm>); everything else = reference/open models
        ours = [r for r in rows if "__" in r["model"]]
        n_models = len(rows)
        n_ours = len(ours)
        pos = [r for r in rows if float(r["task_score"]) > 0]
        pos_ours = [r for r in ours if float(r["task_score"]) > 0]
        mx = max((float(r["task_score"]) for r in rows), default=0.0)
        mx_ours = max((float(r["task_score"]) for r in ours), default=0.0)
        rec = {"task": slug, "n_models": n_models, "n_ours": n_ours,
               "n_pos": len(pos), "n_pos_ours": len(pos_ours),
               "max_score": round(mx, 4), "max_ours": round(mx_ours, 4)}
        report.append(rec)
        # an "all-arm-zero" task = we have our-arm rows but none scored > 0
        if n_ours == 0:
            no_data.append(slug)
        elif len(pos_ours) == 0:
            zero.append(slug)

    if args.json:
        print(json.dumps({"report": report, "all_zero": zero, "no_data": no_data}, indent=2))
        return

    print(f"{'task':<40} {'#our':>5} {'our>0':>6} {'ourmax':>8} {'refmax':>8}")
    print("-" * 72)
    for r in report:
        flag = "  <-- ALL-ARM ZERO" if (r["n_ours"] and r["n_pos_ours"] == 0) else ("  <-- NO OUR DATA" if not r["n_ours"] else "")
        print(f"{r['task']:<40} {r['n_ours']:>5} {r['n_pos_ours']:>6} {r['max_ours']:>8.4f} {r['max_score']:>8.4f}{flag}")
    print("-" * 72)
    print(f"ALL-ARM-ZERO ({len(zero)}): {', '.join(zero) or '-'}")
    print(f"NO-OUR-DATA  ({len(no_data)}): {', '.join(no_data) or '-'}")
    print(f"tasks where >=1 of our arms scores: {sum(1 for r in report if r['n_pos_ours'] > 0)}/{len(report)}")


if __name__ == "__main__":
    main()

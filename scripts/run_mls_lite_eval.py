#!/usr/bin/env python3
"""MLS-Bench-Lite eval driver — the single benchmarking entrypoint (native mlsbench CLI).

Two-stage closed loop, run entirely through the upstream `mlsbench` CLI (github.com/Imbernoulli/MLS-Bench):
  1. baseline:  mlsbench baseline <slug> --seed S        (proposal-independent reference, multi-seed)
  2. eval:      mlsbench agent <slug> --model <worker> --mode eng --extra-context <our-proposal>
                (worker IMPLEMENTS our checkpoint's proposal; --mode eng = no new ideation)
Metrics are read back from each task's leaderboard.csv (native scoring). We report the direction-signed
Δ-over-baseline per task, aggregated across seeds (mean ± CI) — effect size, not a knife-edge pass.

Proposals come from generate_mls_proposals_3arm.py as proposals_<arm>.jsonl (records keyed by task slug).
Task set = the official 30 in docs/eval/mls_bench_lite_tasks.json.

NOTE: the native eval env (docker images per package, data, provider/model routing) is provisioned by cc002
(see agent-memory/coder/mls_lite_eval_env_request.md). Use --dry-run to print the exact commands first.

Usage:
  python scripts/run_mls_lite_eval.py --arms base,sft,rl --seeds 3 --worker-model claude-sonnet-4-6 --dry-run
"""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _mls_lite_common as mls  # noqa: E402


def load_proposals(path: Path) -> dict[str, str]:
    out = {}
    for line in path.open():
        r = json.loads(line)
        out[r["task"]] = r.get("proposal") or r.get("text") or ""
    return out


def sh(cmd: list[str], dry: bool) -> int:
    print("  $", " ".join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd).returncode


def run_baseline(mlsbench: str, slug: str, seed: int, dry: bool) -> None:
    sh([mlsbench, "baseline", slug, "--seed", str(seed)], dry)


HPARAM_DIRECTIVE = {
    "fixed": ("\n\n[Constraint] Implement ONLY the proposed idea in the task's editable region. Do NOT "
              "change training hyperparameters (learning rate, batch size, epochs) or the eval/split/reporting."),
    "swap":  ("\n\n[Allowed] You MAY also retune training hyperparameters (learning rate, batch size, epochs) "
              "if the proposed idea benefits from it. Do NOT change the eval/split/reporting."),
}


def run_agent(mlsbench: str, slug: str, worker_model: str, mode: str, proposal: str,
              workspace: Path, dry: bool, hparam_mode: str = "fixed") -> None:
    # proposal injected as --extra-context (file), with a fixed/swap hparam directive appended
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, dir=workspace) as fh:
        fh.write(proposal + HPARAM_DIRECTIVE.get(hparam_mode, HPARAM_DIRECTIVE["fixed"]))
        ctx = fh.name
    sh([mlsbench, "agent", slug, "--model", worker_model, "--mode", mode,
        "--extra-context", ctx, "--workspace", str(workspace)], dry)


def aggregate(args, slugs: list[str], arms: list[str]) -> None:
    """Read leaderboards (shared FS) and write the Δ-over-baseline results file."""
    results = []
    for arm in arms:
        for slug in slugs:
            setting = (mls.visible_settings(slug, args.mls_root) or [None])[0]
            base = mls.baseline_metrics(slug, setting, args.mls_root) if setting else {}
            run = mls.run_metrics(slug, setting, args.worker_model, args.mls_root)
            d = mls.signed_delta(slug, setting, base, run, args.mls_root)
            deltas = [d] if d is not None else []
            rec = {"arm": arm, "task": slug, "setting": setting, "n_seeds": len(deltas),
                   "delta_mean_pct": round(statistics.mean(deltas), 3) if deltas else None,
                   "delta_ci_pct": None}
            results.append(rec)
            print(f"  {arm}/{slug}: Δ={rec['delta_mean_pct']}% (n={rec['n_seeds']})")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


def run_cluster(args, slugs: list[str], arms: list[str], seeds: list[int]) -> None:
    """Fan the same work items out as 1-GPU DLC jobs via mls_cluster_dispatch.py.

    Seeds are split by tier: the 6 HEAVY tasks (4-12 GPU-h each = ~58% of the GPU-h budget) get
    only `--heavy-seeds` seeds (default 1), since their run-to-run variance is dominated by the single
    long training run, not seed noise. Cheap tasks keep the full `--seeds` for a real CI. This is the
    dominant lever on wall-clock because the farm is GPU-quota-bound, not throughput-bound.
    """
    from datetime import datetime
    import mls_cluster_dispatch as mcd
    dispatcher = str(ROOT / "scripts" / "mls_cluster_dispatch.py")
    batch = args.batch or f"mlslite_{datetime.now().strftime('%m%d_%H%M%S')}"
    common = [sys.executable, dispatcher]

    heavy = [s for s in slugs if s in mcd.HEAVY_TASKS]
    cheap = [s for s in slugs if s not in mcd.HEAVY_TASKS]
    heavy_seeds_csv = ",".join(str(s) for s in seeds[:max(1, args.heavy_seeds)])
    cheap_seeds_csv = ",".join(str(s) for s in seeds)
    # (task subset, seeds) groups; heavy tasks capped at heavy_seeds to shorten the tail
    groups = [(cheap, cheap_seeds_csv), (heavy, heavy_seeds_csv)]

    def run(cmd: list[str]) -> None:
        print("  $", " ".join(cmd), flush=True)
        if not args.dry_run and subprocess.run(cmd).returncode != 0:
            sys.exit(f"dispatcher step failed: {' '.join(cmd)}")

    for group_slugs, group_seeds in groups:
        if not group_slugs:
            continue
        tasks_csv = ",".join(group_slugs)
        if not args.skip_baseline:
            run(common + ["submit-batch", "--kind", "baseline", "--batch", batch,
                          "--tier", "full", "--tasks", tasks_csv, "--seeds", group_seeds,
                          "--max-concurrent", str(args.max_concurrent)])
        agent_cmd = common + ["submit-batch", "--kind", "agent", "--batch", batch,
                      "--tier", "full", "--tasks", tasks_csv, "--seeds", group_seeds,
                      "--arms", args.arms, "--model", args.worker_model, "--mode", args.mode,
                      "--max-concurrent", str(args.max_concurrent)]
        if args.no_proposal:
            agent_cmd += ["--no-proposal"]   # native MLS-Bench agent: worker solves from scratch, no master proposal
        else:
            agent_cmd += ["--proposals-dir", args.proposals_dir,
                          "--hparam-suffix", HPARAM_DIRECTIVE.get(args.hparam_mode, "")]
        run(agent_cmd)
    run(common + ["status", "--batch", batch, "--watch"])
    if not args.dry_run:
        print(f"== cluster batch '{batch}' finished; aggregating ==")
        aggregate(args, slugs, arms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite-json", default=str(ROOT / "docs/eval/mls_bench_lite_tasks.json"))
    ap.add_argument("--proposals-dir", default=str(ROOT / "runs/researcher_cot/mls_eval/proposals"))
    ap.add_argument("--arms", default="base,sft,rl")
    ap.add_argument("--tasks", default=None, help="comma slug subset (default: all 30)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--heavy-seeds", type=int, default=1,
                    help="seeds for the 6 HEAVY tasks (cluster only); capped low since their "
                         "wall-clock dominates the GPU-quota-bound tail (default 1)")
    ap.add_argument("--worker-model", default="claude-fable-5")
    ap.add_argument("--mode", default="eng", choices=["eng", "sci"])
    ap.add_argument("--no-proposal", action="store_true",
                    help="native-agent arm (e.g. purefable5): no master proposal; worker solves from "
                         "MLS-Bench's native prompt. Use with --mode sci to match the original benchmark.")
    ap.add_argument("--hparam-mode", default="fixed", choices=["fixed", "swap"],
                    help="fixed = idea only (attribution); swap = idea may retune lr/batch/epochs (ablation)")
    ap.add_argument("--mls-root", default=mls.DEFAULT_MLS_ROOT)
    ap.add_argument("--mlsbench", default="mlsbench")
    ap.add_argument("--out", default=str(ROOT / "runs/researcher_cot/mls_lite_eval/results.json"))
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dispatch", default="local", choices=["local", "cluster"],
                    help="local = serial on this GPU; cluster = fan out as 1-GPU PAI DLC "
                         "jobs on the L20Z farm (quota1shcr2h7uae) via mls_cluster_dispatch.py")
    ap.add_argument("--tier", default="full", choices=["fast", "full", "heavy"],
                    help="fast = 24 cheap tasks; heavy = the 6 long ones; full = all 30")
    ap.add_argument("--batch", default=None,
                    help="cluster batch name (default: mlslite_<timestamp>)")
    ap.add_argument("--max-concurrent", type=int, default=64)  # 96->64: MaaS degrades under 96-way load (empty completions); see mls_cluster_dispatch.py
    args = ap.parse_args()

    lite = json.loads(Path(args.lite_json).read_text())
    slugs = args.tasks.split(",") if args.tasks else lite["slugs"]
    if args.tier != "full":
        import mls_cluster_dispatch as mcd
        slugs = [s for s in slugs
                 if (s in mcd.HEAVY_TASKS) == (args.tier == "heavy")]
    arms = args.arms.split(",")
    seeds = list(range(1, args.seeds + 1))

    if args.dispatch == "cluster":
        run_cluster(args, slugs, arms, seeds)
        return

    ws_root = Path(tempfile.mkdtemp(prefix="mls_lite_ws_")) if not args.dry_run else Path("/tmp/mls_lite_ws")
    results = []

    # baselines are proposal-independent -> run once per (task, seed)
    if not args.skip_baseline:
        print("== baselines ==")
        for slug in slugs:
            for s in seeds:
                run_baseline(args.mlsbench, slug, s, args.dry_run)

    for arm in arms:
        props = load_proposals(Path(args.proposals_dir) / f"proposals_{arm}.jsonl") if not args.dry_run else {}
        print(f"== arm={arm} ==")
        for slug in slugs:
            setting = (mls.visible_settings(slug, args.mls_root) or [None])[0]
            base = mls.baseline_metrics(slug, setting, args.mls_root) if setting else {}
            deltas = []
            for s in seeds:
                ws = ws_root / arm / slug / f"seed{s}"
                ws.mkdir(parents=True, exist_ok=True)
                run_agent(args.mlsbench, slug, args.worker_model, args.mode,
                          props.get(slug, ""), ws, args.dry_run, hparam_mode=args.hparam_mode)
                if not args.dry_run:
                    run = mls.run_metrics(slug, setting, args.worker_model, args.mls_root)
                    d = mls.signed_delta(slug, setting, base, run, args.mls_root)
                    if d is not None:
                        deltas.append(d)
            rec = {"arm": arm, "task": slug, "setting": setting, "n_seeds": len(deltas),
                   "delta_mean_pct": round(statistics.mean(deltas), 3) if deltas else None,
                   "delta_ci_pct": round(1.96 * statistics.pstdev(deltas) / (len(deltas) ** 0.5), 3)
                   if len(deltas) > 1 else None}
            results.append(rec)
            print(f"  {arm}/{slug}: Δ={rec['delta_mean_pct']}% ±{rec['delta_ci_pct']} (n={rec['n_seeds']})")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

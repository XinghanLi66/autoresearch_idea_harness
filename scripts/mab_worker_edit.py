#!/usr/bin/env python3
"""MAB in-pod worker: implement ONE arm's proposal for ONE (task, seed), run it, score it.

MLAgentBench has no `agent --extra-context` CLI, so we implement the worker-edit ourselves:
  1. copy the task's benchmarks/<task>/env/ -> a fresh workspace under /newcpfs;
  2. (arm != baseline) read research_problem.txt + baseline train.py + the arm's proposal .md, then POST
     to the local MaaS shim (127.0.0.1:18791, Anthropic /v1/messages, model claude-fable-5__<arm>) asking
     the fable-5 worker for the FULL rewritten train.py that implements the proposal, respecting
     read_only_files.txt. Strip code fences, write train.py.
     (baseline) skip the edit and run the unmodified train.py.
  3. run `python train.py` in the workspace (produces submission.csv);
  4. score with scripts/eval.py::get_score(workspace) via _mab_common;
  5. append the metric row to runs/researcher_cot/mab_eval/results/<task>.csv.

The "__<arm>" suffix is stripped by the shim so a single fable-5 worker serves every arm; the arm name is
what we store in the results sink (so mab_report can group by arm). Baseline rows are stored as model="baseline".

Usage (run inside the mab conda env, with the shim already listening on 127.0.0.1:18791):
  python scripts/mab_worker_edit.py --task cifar10 --arm exp09rl --seed 1 --proposal /newcpfs/.../p.md
  python scripts/mab_worker_edit.py --task cifar10 --arm baseline --seed 1
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _mab_common as mab  # noqa: E402

DEFAULT_SHIM = "http://127.0.0.1:18791"
DEFAULT_WS_ROOT = os.environ.get("MAB_WS_ROOT", "/newcpfs/lxh/mab_ws")

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

WORKER_SYSTEM = (
    "You are an elite ML engineer implementing a research proposal into a single training script. "
    "You are given a task description, the current baseline `train.py`, and ONE research idea to implement. "
    "Rewrite `train.py` in full so that it faithfully implements the idea while still producing the required "
    "`submission.csv` in the exact same format as the baseline. Do NOT change the evaluation/split/reporting "
    "logic and do NOT read or modify any read-only files. Keep it runnable end-to-end with `python train.py`. "
    "Output ONLY the complete new contents of train.py inside a single ```python code block, with no prose.")


def build_user_prompt(task: str, research_problem: str, baseline: str,
                      proposal: str, read_only: str) -> str:
    ro = read_only.strip() or "(none listed)"
    return (
        f"# Task: {task}\n\n"
        f"## Research problem\n{research_problem.strip()}\n\n"
        f"## Read-only files (must NOT be edited or overwritten)\n{ro}\n\n"
        f"## Proposal to implement (the ONLY change to make)\n{proposal.strip()}\n\n"
        f"## Current baseline train.py\n```python\n{baseline}\n```\n\n"
        "Return the FULL rewritten `train.py` implementing the proposal above. Output ONLY one "
        "```python code block containing the entire file.")


NAIVE_SYSTEM = (
    "You are a competent ML engineer completing a starter training script. You are given a task description "
    "and a `train.py` that may be a skeleton with unfilled sections. Fill it in with a STANDARD, straightforward, "
    "competent baseline — the obvious default approach a solid engineer would write. Do NOT add novel, creative, "
    "or research-grade ideas; this is the plain reference baseline. Produce the required `submission.csv` in the "
    "specified format, keep it runnable end-to-end with `python train.py`, and do NOT modify read-only files. "
    "Output ONLY the complete new contents of train.py inside a single ```python code block, with no prose.")


def build_naive_prompt(task: str, research_problem: str, skeleton: str, read_only: str) -> str:
    ro = read_only.strip() or "(none listed)"
    return (
        f"# Task: {task}\n\n"
        f"## Research problem\n{research_problem.strip()}\n\n"
        f"## Read-only files (must NOT be edited or overwritten)\n{ro}\n\n"
        f"## Starter train.py (fill in / complete it with a STANDARD baseline, no novel ideas)\n"
        f"```python\n{skeleton}\n```\n\n"
        "Return the FULL completed `train.py` (standard baseline only). Output ONLY one ```python code block.")


NATIVE_SYSTEM = (
    "You are an elite ML researcher-engineer competing to maximize a task's metric. You are given a task "
    "description and the current baseline `train.py`. There is NO proposal — devise your OWN best approach and "
    "implement it: improve the model/training/features however you judge best to beat the baseline. Rewrite "
    "`train.py` in full, still producing the required `submission.csv` in the exact same format; do NOT change "
    "the evaluation/split/reporting logic and do NOT read or modify any read-only files. Keep it runnable "
    "end-to-end with `python train.py`. Output ONLY the complete new contents of train.py inside a single "
    "```python code block, with no prose.")


def build_native_prompt(task: str, research_problem: str, baseline: str, read_only: str) -> str:
    ro = read_only.strip() or "(none listed)"
    return (
        f"# Task: {task}\n\n"
        f"## Research problem\n{research_problem.strip()}\n\n"
        f"## Read-only files (must NOT be edited or overwritten)\n{ro}\n\n"
        f"## Current baseline train.py (improve it with your OWN approach — no proposal given)\n"
        f"```python\n{baseline}\n```\n\n"
        "Devise and implement your own best idea to maximize the metric. Return the FULL rewritten `train.py`. "
        "Output ONLY one ```python code block containing the entire file.")


def strip_code_fences(text: str) -> str:
    """Extract the first fenced code block; fall back to the raw text with any stray fences removed."""
    m = _FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip("\n")
    return re.sub(r"^```(?:python|py)?\s*|```\s*$", "", (text or "").strip(), flags=re.MULTILINE).strip("\n")


def call_worker(shim_url: str, model: str, system: str, user: str,
                max_tokens: int, timeout: int, max_retries: int) -> str:
    """POST Anthropic /v1/messages to the local shim; return the text content. Retries with backoff."""
    import urllib.request
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    last = ""
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(shim_url.rstrip("/") + "/v1/messages", data=body,
                                         headers={"Content-Type": "application/json",
                                                  "x-api-key": "local",
                                                  "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            blocks = data.get("content") or []
            text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
            if text.strip():
                return text
            last = f"empty completion (attempt {attempt})"
        except Exception as exc:  # noqa: BLE001 — retry transient gateway/HTTP errors
            last = f"{type(exc).__name__}: {exc}"
        print(f"  [retry] worker call attempt {attempt}/{max_retries}: {last}", flush=True)
        time.sleep(min(4 * attempt, 30))
    raise RuntimeError(f"worker call failed after {max_retries} attempts: {last}")


def prepare_workspace(task: str, workspace: Path, mab_root: str) -> Path:
    """Copy benchmarks/<task>/env/ -> a fresh workspace (carries prepared data/ + networks/)."""
    env_dir = mab.task_dir(task, mab_root) / "env"
    if not env_dir.exists():
        sys.exit(f"missing env dir for {task}: {env_dir}")
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(env_dir, workspace)
    return workspace


def run_train(workspace: Path, seed: int, timeout: int) -> None:
    env = dict(os.environ)
    env["SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    print(f"  running train.py in {workspace} (seed={seed}, timeout={timeout}s)", flush=True)
    proc = subprocess.run([sys.executable, "train.py"], cwd=str(workspace),
                          env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"train.py exited rc={proc.returncode}")
    if not (workspace / "submission.csv").exists():
        raise RuntimeError("train.py finished but produced no submission.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--arm", required=True, help="arm name, or 'baseline' to run the unmodified train.py")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--proposal", default=None, help="proposal .md path (required unless --arm baseline)")
    ap.add_argument("--native", action="store_true",
                    help="no proposal: worker devises + implements its OWN approach (native MLAgentBench agent)")
    ap.add_argument("--workspace", default=None, help="workspace dir (default: <MAB_WS_ROOT>/<task>_<arm>_s<seed>)")
    ap.add_argument("--mab-root", default=mab.DEFAULT_MAB_ROOT)
    ap.add_argument("--shim-url", default=DEFAULT_SHIM)
    ap.add_argument("--worker-model", default="claude-fable-5")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--http-timeout", type=int, default=1200, help="per worker-call HTTP timeout (s)")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--train-timeout", type=int, default=7200, help="train.py wall-clock cap (s)")
    ap.add_argument("--keep-workspace", action="store_true")
    args = ap.parse_args()

    is_baseline = args.arm == "baseline"
    worker_fill_baseline = is_baseline and mab.baseline_mode(args.task) == "worker_fill"
    if not is_baseline and not args.native and not args.proposal:
        sys.exit("--proposal is required unless --arm baseline or --native")

    ws = Path(args.workspace) if args.workspace else Path(DEFAULT_WS_ROOT) / f"{args.task}_{args.arm}_s{args.seed}"
    ws = ws.resolve()
    prepare_workspace(args.task, ws, args.mab_root)

    # invoke the worker for: arm runs (implement proposal) OR skeleton-baseline (fill a naive baseline)
    if not is_baseline or worker_fill_baseline:
        sdir = mab.task_dir(args.task, args.mab_root) / "scripts"
        research_problem = (sdir / "research_problem.txt").read_text(errors="ignore")
        ro_file = sdir / "read_only_files.txt"
        read_only = ro_file.read_text(errors="ignore") if ro_file.exists() else ""
        baseline_src = (ws / mab.edit_file(args.task)).read_text(errors="ignore")
        model = f"{args.worker_model}__{args.arm}"  # shim strips __<arm>; one worker serves all arms
        if worker_fill_baseline:
            system = NAIVE_SYSTEM
            user = build_naive_prompt(args.task, research_problem, baseline_src, read_only)
        elif args.native:
            system = NATIVE_SYSTEM
            user = build_native_prompt(args.task, research_problem, baseline_src, read_only)
        else:
            proposal = Path(args.proposal).read_text(errors="ignore")
            system = WORKER_SYSTEM
            user = build_user_prompt(args.task, research_problem, baseline_src, proposal, read_only)
        raw = call_worker(args.shim_url, model, system, user,
                          args.max_tokens, args.http_timeout, args.max_retries)
        new_train = strip_code_fences(raw)
        if len(new_train) < 50:
            sys.exit(f"worker returned too little code ({len(new_train)}c) — aborting")
        (ws / mab.edit_file(args.task)).write_text(new_train)
        print(f"  wrote {'naive-baseline' if worker_fill_baseline else 'edited'} "
              f"{mab.edit_file(args.task)} ({len(new_train)}c)", flush=True)

    run_train(ws, args.seed, args.train_timeout)
    score = mab.get_score(args.task, ws, args.mab_root)
    model_col = "baseline" if is_baseline else args.arm
    out = mab.append_result(args.task, model_col, args.seed, score, args.results_dir)
    print(f"MAB_SCORE task={args.task} arm={model_col} seed={args.seed} "
          f"{mab.metric_name(args.task)}={score} -> {out}", flush=True)

    if not args.keep_workspace and not args.workspace:
        shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    main()

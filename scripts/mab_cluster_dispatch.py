#!/usr/bin/env python3
"""MLAgentBench (MAB) cluster dispatcher — fan worker-edit runs out as PAI DLC jobs.

Mirrors mls_cluster_dispatch.py: same farm (PAI DLC workspace 262162 / quota quota1shcr2h7uae, 16 x 8
L20Z), same /newcpfs shared mount, same in-pod MaaS shim on 127.0.0.1:18791. The difference: MAB has no
`mlsbench` CLI, so each pod activates the `mab` conda env and runs scripts/mab_worker_edit.py, which
implements the arm's proposal into train.py, runs it, and scores it into our own results/<task>.csv sink.

Subcommands:
    submit-one    submit a single baseline/agent (task,arm,seed) run  -> prints JobId
    submit-batch  fan out (tasks x arms x seeds)                        -> batch dir
    status        poll job states for a batch
    stop          stop all non-terminal jobs of a batch

Examples:
    python scripts/mab_cluster_dispatch.py submit-batch --kind baseline --batch mab_bl \
        --seeds 1,2,3 --dry-run
    python scripts/mab_cluster_dispatch.py submit-batch --kind agent --batch mab_run \
        --arms base,rl --proposals-dir runs/researcher_cot/mab_eval/proposals --seeds 1,2,3 --dry-run
    python scripts/mab_cluster_dispatch.py status --batch mab_run --watch
"""
from __future__ import annotations

import argparse
import json
import os
import re as _re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
MAB_ROOT = Path(os.environ.get("MAB_ROOT", "/newcpfs/lxh/MLAgentBench"))
MINICONDA = "/newcpfs/lxh/miniconda3"
CONDA_ENV = "mab"
PAI_MANAGE = "/root/.claude/skills/pai/scripts/pai_manage.py"
CRED_URI = "http://localhost:7002/api/v1/credentials/0"

WORKSPACE_ID = "262162"
QUOTA_ID = "quota1shcr2h7uae"
IMAGE = "ali-sg-acr-registry-vpc.ap-southeast-1.cr.aliyuncs.com/xhs-llm/guangsu:ngc2506_te2.8_h100_v3"

BATCH_ROOT = HARNESS_ROOT / "runs" / "mab_cluster"
TASKS_JSON = HARNESS_ROOT / "docs" / "eval" / "mab_tasks.json"
WORKER = HARNESS_ROOT / "scripts" / "mab_worker_edit.py"
SHIM = HARNESS_ROOT / "scripts" / "anthropic_maas_shim.py"

TERMINAL = {"Succeeded", "Failed", "Stopped", "Untracked"}

_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


def strip_think(text: str) -> str:
    """Inject the IDEA, not the raw CoT. Remove complete <think>...</think> blocks; if an unclosed
    <think> remains (truncated proposal), drop everything from it onward. Applied to every arm so the
    worker implements the proposal's idea, not the generating model's reasoning trace."""
    t = _THINK_RE.sub("", text)
    if "<think>" in t:
        t = t.split("<think>", 1)[0]
    return t.strip()


# GPU vs CPU tasks. cifar10 / ogbn-arxiv / imdb train on 1 GPU; the tabular tasks run on CPU (0 GPU).
GPU_TASKS = {"cifar10", "ogbn-arxiv", "imdb"}


def _task_gpus(task: str) -> int:
    return 1 if task in GPU_TASKS else 0


# ---------------------------------------------------------------------------
# PAI plumbing (verbatim from mls_cluster_dispatch.py)
# ---------------------------------------------------------------------------

def _pai(args: list[str], *, parse: bool = True):
    env = dict(os.environ)
    env["ALIBABA_CLOUD_CREDENTIALS_URI"] = CRED_URI
    env.pop("PAI_WORKSPACE_ID", None)  # DSW exports its own workspace; must not leak
    proc = subprocess.run(
        [sys.executable, PAI_MANAGE, *args],
        capture_output=True, text=True, env=env,
    )
    if not parse:
        return proc
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"pai_manage {args[0]} produced no output; stderr tail: "
                           f"{proc.stderr.strip()[-400:]}")
    return json.loads(out)


def job_status(job_id: str) -> str:
    body = _pai(["get-job", "--job-id", job_id])["body"]
    return body.get("Status", "Unknown")


def pod_log_tail(job_id: str, n: int = 30) -> str:
    try:
        body = _pai(["get-job", "--job-id", job_id])["body"]
        pods = [p for p in body.get("Pods", []) if p.get("Type") != "aimaster"]
        if not pods:
            return "(no pods)"
        logs = _pai(["get-pod-logs", "--job-id", job_id, "--pod-id", pods[0]["PodId"]])
        lines = logs.get("body", {}).get("Logs", [])
        return "\n".join(lines[-n:])
    except Exception as exc:  # noqa: BLE001 — diagnostics only
        return f"(log fetch failed: {exc})"


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def build_command(task: str, arm: str, *, seed: int, proposal: str | None,
                  workspace: str, worker_model: str, tag: str, native: bool = False) -> str:
    """In-pod command: start the MaaS shim (base env), activate `mab`, run mab_worker_edit.py."""
    worker_args = [f"--task {task}", f"--arm {arm}", f"--seed {seed}",
                   f"--workspace {workspace}", f"--worker-model {worker_model}",
                   "--shim-url http://127.0.0.1:18791"]
    if native:
        worker_args.append("--native")   # no proposal: worker devises + implements its own approach
    elif arm != "baseline":
        if not proposal:
            raise ValueError(f"agent run for {tag} needs a proposal file")
        worker_args.append(f"--proposal {proposal}")
    # use the mab env's python EXPLICITLY (conda activate alone left bare `python` = system py3.12,
    # whose torch is ABI-broken → GPU tasks failed on `import torch`). sys.executable then propagates
    # the mab python to train.py inside run_train.
    worker_cmd = f"{MINICONDA}/envs/{CONDA_ENV}/bin/python {WORKER} " + " ".join(worker_args)

    return f"""set -o pipefail
export PATH={MINICONDA}/bin:$PATH
unset PYTHONPATH
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HOME=/root
export MAB_ROOT={MAB_ROOT}
{MINICONDA}/bin/python {SHIM} --port 18791 &
SHIM_PID=$!
sleep 6
curl -sS -m 10 http://127.0.0.1:18791/health || (echo SHIM_DEAD; exit 91)
source {MINICONDA}/etc/profile.d/conda.sh
conda activate {CONDA_ENV}
cd {HARNESS_ROOT}
{worker_cmd}
RC=$?
kill $SHIM_PID 2>/dev/null
echo "MAB_RUN_EXIT=$RC"
exit $RC
"""


def submit_job(name: str, command: str, *, max_minutes: int, cpus: int = 16,
               memory: str = "100Gi", gpus: int = 1, dry_run: bool = False) -> str | None:
    if dry_run:
        print(f"  [dry-run] would submit {name} (gpus={gpus} cpus={cpus} mem={memory} "
              f"max_minutes={max_minutes})")
        print("  --- in-pod command ---")
        print("\n".join("    " + ln for ln in command.strip().splitlines()))
        print("  ----------------------")
        return None
    cmd_file = Path("/tmp") / f"mab_dispatch_{os.getpid()}_{time.monotonic_ns()}.sh"
    cmd_file.write_text(command)
    try:
        _pai([
            "create-job", "--name", name,
            "--workspace-id", WORKSPACE_ID, "--resource-id", QUOTA_ID,
            "--image", IMAGE,
            "--command-file", str(cmd_file),
            "--gpus", str(gpus), "--cpus", str(cpus), "--memory", memory,
            "--shared-memory", "32Gi", "--pod-count", "1",
            "--max-running-minutes", str(max_minutes),
            "--priority", "9",
            "--enable-rdma", "false",
        ], parse=False)
        # create-job returns an empty body; resolve the JobId by listing (list-jobs has indexing lag).
        for _ in range(12):
            jobs = _pai(["list-jobs", "--set", f"WorkspaceId={WORKSPACE_ID}",
                         "--set", "PageSize=100", "--compact"])["body"].get("Jobs", [])
            for j in jobs:
                if j.get("DisplayName") == name and j.get("Status") not in TERMINAL:
                    return j["JobId"]
            time.sleep(5)
        # Submitted but JobId unresolved under API lag — the job still runs + writes results/<task>.csv.
        # Do NOT crash the batch (scoring reads the CSV sink, not this tracking id).
        print(f"[warn] submitted '{name}' but JobId unresolved after retries (untracked; job still runs)",
              flush=True)
        return None
    finally:
        cmd_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Batch state
# ---------------------------------------------------------------------------

def batch_dir(batch: str) -> Path:
    d = BATCH_ROOT / batch
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_jobs(batch: str) -> list[dict]:
    f = batch_dir(batch) / "jobs.json"
    return json.loads(f.read_text()) if f.exists() else []


def save_jobs(batch: str, jobs: list[dict]) -> None:
    (batch_dir(batch) / "jobs.json").write_text(json.dumps(jobs, indent=1))


def all_slugs() -> list[str]:
    return json.loads(TASKS_JSON.read_text())["slugs"]


def select_tasks(tasks_arg: str | None) -> list[str]:
    if tasks_arg:
        return [t.strip() for t in tasks_arg.split(",") if t.strip()]
    return all_slugs()


def default_minutes(task: str, kind: str) -> int:
    # GPU tasks train a real model; tabular are quick. Agent runs add the worker LLM turn.
    if task in GPU_TASKS:
        return 240 if kind == "baseline" else 360   # 4 h / 6 h
    return 120 if kind == "baseline" else 180       # 2 h / 3 h


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _ws_for(batch: str, tag: str) -> str:
    return str(batch_dir(batch) / "ws" / tag)


def cmd_submit_one(args) -> None:
    kind = "baseline" if args.arm == "baseline" else "agent"
    tag = args.tag or f"{args.arm}_{args.task}_s{args.seed}"
    if args.proposal and not str(Path(args.proposal).resolve()).startswith("/newcpfs"):
        sys.exit(f"--proposal must live under /newcpfs (pods can't see local disks): {args.proposal}")
    ws = args.workspace or _ws_for(args.batch, tag)
    command = build_command(args.task, args.arm, seed=args.seed, proposal=args.proposal,
                            workspace=ws, worker_model=args.worker_model, tag=tag)
    minutes = args.max_minutes or default_minutes(args.task, kind)
    name = f"mab-{tag}"[:60].replace("_", "-")
    g = _task_gpus(args.task)
    job_id = submit_job(name, command, max_minutes=minutes, gpus=g,
                        cpus=max(8, 16 * max(g, 1)), memory=f"{100 * max(g, 1)}Gi",
                        dry_run=args.dry_run)
    if args.dry_run:
        return
    entry = {"job_id": job_id, "name": name, "kind": kind, "task": args.task,
             "seed": args.seed, "arm": args.arm, "tag": tag, "workspace": ws,
             "proposal": args.proposal, "status": "Submitted",
             "submitted_at": datetime.now(timezone.utc).isoformat()}
    jobs = load_jobs(args.batch)
    jobs.append(entry)
    save_jobs(args.batch, jobs)
    print(job_id)


def cmd_submit_batch(args) -> None:
    tasks = select_tasks(args.tasks)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.kind == "baseline":
        arms = ["baseline"]
    else:
        arms = args.arms.split(",") if args.arms else []
        if not arms:
            sys.exit("--arms required for agent batches")

    proposals: dict[str, dict[str, str]] = {}
    if args.kind == "agent" and not args.native:
        if not args.proposals_dir:
            sys.exit("--proposals-dir required for agent batches (proposals_<arm>.jsonl)")
        pdir = Path(args.proposals_dir)
        for arm in arms:
            f = pdir / f"proposals_{arm}.jsonl"
            if not f.exists():
                sys.exit(f"missing {f}")
            proposals[arm] = {}
            for line in f.open():
                r = json.loads(line)
                proposals[arm][r["task"]] = r.get("proposal") or r.get("text") or ""

    jobs = load_jobs(args.batch)
    seen_tags = {j["tag"] for j in jobs}
    n_submitted = 0
    for task in tasks:
        for arm in arms:
            for seed in seeds:
                tag = f"{arm}_{task}_s{seed}"
                if tag in seen_tags:
                    print(f"[skip] {tag} already in batch")
                    continue
                proposal_path = None
                if arm != "baseline" and not args.native:
                    text = strip_think(proposals[arm].get(task, ""))
                    if not text:
                        print(f"[warn] no proposal (or empty after <think>-strip) for {arm}/{task}, skipping")
                        continue
                    pfile = batch_dir(args.batch) / "proposals" / f"{tag}.md"
                    pfile.parent.mkdir(parents=True, exist_ok=True)
                    pfile.write_text(text + args.hparam_suffix)
                    proposal_path = str(pfile)
                ws = _ws_for(args.batch, tag)
                command = build_command(task, arm, seed=seed, proposal=proposal_path,
                                        workspace=ws, worker_model=args.worker_model, tag=tag,
                                        native=args.native)
                minutes = args.max_minutes or default_minutes(task, args.kind)
                name = f"mab-{tag}"[:60].replace("_", "-")
                g = _task_gpus(task)
                if not args.dry_run:
                    while True:
                        in_flight = sum(1 for j in jobs if j["status"] not in TERMINAL)
                        if in_flight < args.max_concurrent:
                            break
                        print(f"[throttle] {in_flight} in flight >= {args.max_concurrent}; polling...")
                        refresh_statuses(jobs)
                        save_jobs(args.batch, jobs)
                        time.sleep(args.poll_interval)
                job_id = submit_job(name, command, max_minutes=minutes, gpus=g,
                                    cpus=max(8, 16 * max(g, 1)), memory=f"{100 * max(g, 1)}Gi",
                                    dry_run=args.dry_run)
                if args.dry_run:
                    n_submitted += 1
                    continue
                entry = {"job_id": job_id, "name": name, "kind": args.kind, "task": task,
                         "seed": seed, "arm": arm, "tag": tag, "workspace": ws,
                         "proposal": proposal_path, "status": "Submitted",
                         "submitted_at": datetime.now(timezone.utc).isoformat()}
                jobs.append(entry)
                seen_tags.add(tag)
                save_jobs(args.batch, jobs)
                n_submitted += 1
                print(f"[submit] {tag} -> {job_id}")
    if args.dry_run:
        print(f"\n[dry-run] batch '{args.batch}': {n_submitted} jobs would be submitted")
    else:
        print(f"\nbatch '{args.batch}': {n_submitted} new jobs, {len(jobs)} total "
              f"(state: {batch_dir(args.batch)/'jobs.json'})")


def refresh_statuses(jobs: list[dict]) -> None:
    for j in jobs:
        if j["status"] in TERMINAL:
            continue
        if not j.get("job_id"):
            j["status"] = "Untracked"  # submitted but JobId unresolved; runs + writes results sink
            continue
        try:
            j["status"] = job_status(j["job_id"])
        except Exception as exc:  # noqa: BLE001 — keep polling the rest
            print(f"[warn] status({j['job_id']}) failed: {exc}")


def cmd_status(args) -> None:
    while True:
        jobs = load_jobs(args.batch)
        if not jobs:
            sys.exit(f"no jobs recorded for batch '{args.batch}'")
        refresh_statuses(jobs)
        save_jobs(args.batch, jobs)
        counts: dict[str, int] = {}
        for j in jobs:
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
        if args.verbose:
            for j in jobs:
                print(f"  {j['status']:12s} {j['tag']:40s} {j['job_id']}")
        for j in jobs:
            if j["status"] == "Failed" and not j.get("_dumped"):
                print(f"\n--- log tail of FAILED {j['tag']} ({j['job_id']}) ---")
                print(pod_log_tail(j["job_id"]))
                j["_dumped"] = True
        save_jobs(args.batch, jobs)
        if not args.watch or all(j["status"] in TERMINAL for j in jobs):
            break
        time.sleep(args.poll_interval)


def cmd_stop(args) -> None:
    jobs = load_jobs(args.batch)
    for j in jobs:
        if j["status"] not in TERMINAL and j.get("job_id"):
            print(f"stopping {j['tag']} ({j['job_id']})")
            _pai(["stop-job", "--job-id", j["job_id"]], parse=False)
    refresh_statuses(jobs)
    save_jobs(args.batch, jobs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("submit-one", help="submit a single (task,arm,seed) run")
    p1.add_argument("--task", required=True)
    p1.add_argument("--arm", required=True, help="arm name, or 'baseline'")
    p1.add_argument("--seed", type=int, default=1)
    p1.add_argument("--proposal", default=None, help="proposal .md (MUST be under /newcpfs; agent only)")
    p1.add_argument("--worker-model", default="claude-fable-5")
    p1.add_argument("--workspace", default=None)
    p1.add_argument("--tag", default=None)
    p1.add_argument("--batch", default="adhoc")
    p1.add_argument("--max-minutes", type=int, default=None)
    p1.add_argument("--dry-run", action="store_true")
    p1.set_defaults(func=cmd_submit_one)

    p2 = sub.add_parser("submit-batch", help="fan out tasks x arms x seeds")
    p2.add_argument("--kind", required=True, choices=["baseline", "agent"])
    p2.add_argument("--tasks", default=None, help="comma slugs override (default: all 5)")
    p2.add_argument("--seeds", default="1", help="comma seeds")
    p2.add_argument("--arms", default=None, help="comma arms (agent kind)")
    p2.add_argument("--proposals-dir", default=None)
    p2.add_argument("--native", action="store_true",
                    help="no proposal: worker devises+implements its own approach (native MLAgentBench agent)")
    p2.add_argument("--hparam-suffix", default="", help="text appended to each proposal ctx")
    p2.add_argument("--worker-model", default="claude-fable-5")
    p2.add_argument("--batch", required=True)
    p2.add_argument("--max-concurrent", type=int, default=64)
    p2.add_argument("--max-minutes", type=int, default=None)
    p2.add_argument("--poll-interval", type=int, default=60)
    p2.add_argument("--dry-run", action="store_true")
    p2.set_defaults(func=cmd_submit_batch)

    p3 = sub.add_parser("status", help="poll batch job states")
    p3.add_argument("--batch", required=True)
    p3.add_argument("--watch", action="store_true")
    p3.add_argument("--verbose", action="store_true")
    p3.add_argument("--poll-interval", type=int, default=120)
    p3.set_defaults(func=cmd_status)

    p4 = sub.add_parser("stop", help="stop all non-terminal jobs in a batch")
    p4.add_argument("--batch", required=True)
    p4.set_defaults(func=cmd_stop)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

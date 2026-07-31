#!/usr/bin/env python3
"""MLS-Bench L20Z cluster dispatcher — fan mlsbench runs out as 1-GPU PAI DLC jobs.

Farm: PAI DLC, workspace 262162 (Data_Agent), quota quota1shcr2h7uae (16 nodes x 8
NVIDIA L20Z = 128 GPUs). Every pod mounts the same /newcpfs as the DSW machines, so
the MLS-Bench checkout, the per-package conda envs (mlsbench-<pkg>), task data, and
leaderboard CSVs are all shared — no replication. Each job runs its own Anthropic->
MaaS shim on 127.0.0.1:18791 (same port the MLS-Bench config.yaml points at).

Subcommands:
    submit-one    submit a single mlsbench baseline/agent run       -> prints JobId
    submit-batch  fan out (tasks x arms x seeds) with a tier filter -> batch dir
    status        poll job states for a batch (or --job-id)
    stop          stop all non-terminal jobs of a batch

Examples:
    # one baseline run
    python scripts/mls_cluster_dispatch.py submit-one --kind baseline \
        --task dl-activation-function --seed 1

    # one agent run with a proposal file (must be under /newcpfs!)
    python scripts/mls_cluster_dispatch.py submit-one --kind agent \
        --task dl-activation-function --model claude-fable-5 \
        --proposal /newcpfs/.../proposal.md --tag base_s1

    # fast-24 tier, 3 baseline seeds
    python scripts/mls_cluster_dispatch.py submit-batch --kind baseline \
        --tier fast --seeds 1,2,3 --batch bl_fast

    python scripts/mls_cluster_dispatch.py status --batch bl_fast --watch
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
MLS_ROOT = Path(os.environ.get("MLS_BENCH_ROOT", "/newcpfs/lxh/MLS-Bench"))
MINICONDA = "/newcpfs/lxh/miniconda3"
PAI_MANAGE = "/root/.claude/skills/pai/scripts/pai_manage.py"
CRED_URI = "http://localhost:7002/api/v1/credentials/0"

WORKSPACE_ID = "262162"
QUOTA_ID = "quota1shcr2h7uae"
IMAGE = "ali-sg-acr-registry-vpc.ap-southeast-1.cr.aliyuncs.com/xhs-llm/guangsu:ngc2506_te2.8_h100_v3"
NEWCPFS_DATASET = "d-s5sllzhztbq48bhexm"

BATCH_ROOT = HARNESS_ROOT / "runs" / "mls_cluster"
LITE_JSON = HARNESS_ROOT / "docs" / "eval" / "mls_bench_lite_tasks.json"

# The 6 heavy tasks = 58% of the ~73 GPU-h/seed budget (cc001's cost profile).
HEAVY_TASKS = {
    "llm-pretrain-optimizer",      # ~12 h
    "robomimic-bc-loss",           # ~8 h
    "jepa-planning",               # ~6 h
    "robo-diffusion-policy",       # ~6 h
    "llm-rl-importance-sampling",  # ~6 h
    "robo-diffusion-guidance",     # ~4 h
}

TERMINAL = {"Succeeded", "Failed", "Stopped", "Untracked"}

import re as _re
_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


def strip_think(text: str) -> str:
    """Inject the IDEA, not the raw CoT. Remove complete <think>...</think> blocks; if an unclosed
    <think> remains (truncated proposal), drop everything from it onward. Applied to every arm so the
    worker implements the proposal's idea, not the generating model's reasoning trace (also removes the
    CoT-length confound between thinking base arms and concise SFT arms)."""
    t = _THINK_RE.sub("", text)
    if "<think>" in t:
        t = t.split("<think>", 1)[0]
    return t.strip()

# Tasks whose conda env ships a cu130 torch wheel (torch 2.13+cu130). Those need
# GPU driver >= 580; the pool's default 535 driver reports "no GPUs found" /
# "CUDAGuardImpl non-CUDA cpu" and every arm fails. Give ONLY these task jobs the
# 580 driver (serving already uses it) so the 14 cu12x tasks keep their default
# scheduling. cu12x wheels also run fine on 580 (backward-compatible), so this is
# safe to widen later if more envs go cu130.
CU130_TASKS = {
    "cv-3dgs-densification",   # mlsbench-gsplat  (torch 2.13+cu130)
    "cv-vae-loss",             # mlsbench-diffusers-main (torch 2.13+cu130)
}

# Tasks that need >1 GPU. llm-rl (verl GRPO) OOMs the colocated vLLM+FSDP-actor pair on
# one 79GB GPU unless vLLM sleep_mode is on — and sleep_mode's per-step sleep/wake
# overhead pushes the heavy ~6h run over the wall clock. With 2 GPUs the FSDP actor
# update shards across GPUs so vLLM + shard co-reside WITHOUT sleep_mode (result-neutral:
# same global batch/steps). train_1gpu.sh auto-detects N_GPUS and drops sleep_mode at >=2.
GPU_OVERRIDE = {
    "llm-rl-importance-sampling": 2,
    # ⚠️ L20Z ONLY SUPPORTS 1/2/4/8-GPU JOBS. A 3- (or 5/6/7-) GPU job HANGS FOREVER in
    # admission AND head-of-line-BLOCKS every job queued behind it (caused a full-farm
    # starvation 2026-07-26). Only ever use 1/2/4/8 here.
    # ai4sci-pla's 3 PDBbind settings run in PARALLEL (group=1, ThreadPoolExecutor); each is
    # an 800-epoch training that contends ~3x on 1 GPU and blows the 59min budget (blank
    # metrics). Give it 4 GPUs (valid size; mlsbench's _allocate_group_gpu_assignments pins
    # one GPU per parallel setting, 4th idle) -> no contention -> full-quality runs finish.
    "ai4sci-pla-binding-affinity": 4,
}


def _task_gpus(task: str) -> int:
    return GPU_OVERRIDE.get(task, 1)


# Host-RAM overrides (pod --memory). Default is 100Gi/GPU. cv-3dgs's `bonsai` scene
# (292 images + the largest SfM point cloud) OOM-kills at 100Gi during image-load /
# gaussian init -> exit=137 (SIGKILL), zeroing the 4-scene gmean for every arm. Other
# scenes (109-169 imgs) pass at 100Gi. Bump only bonsai's task. (GPU has 80GB — this is
# HOST RAM, not CUDA-OOM.)
MEM_OVERRIDE = {
    "cv-3dgs-densification": "256Gi",
}


def _task_mem(task: str, gpus: int) -> str:
    return MEM_OVERRIDE.get(task, f"{100 * gpus}Gi")


# ---------------------------------------------------------------------------
# PAI plumbing
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

def build_command(kind: str, task: str, *, seed: int | None, model: str,
                  mode: str, proposal: str | None, workspace: str | None,
                  tag: str, extra_args: str = "") -> str:
    """In-pod command: start shim, run one mlsbench invocation, propagate rc."""
    if kind == "baseline":
        mls_cmd = f"{MINICONDA}/bin/mlsbench baseline {task}"
        if seed is not None:
            mls_cmd += f" --seed {seed}"
        mls_cmd += " --resume"
    elif kind == "agent":
        mls_cmd = (f"{MINICONDA}/bin/mlsbench agent {task} --model {model} --mode {mode}")
        if proposal:
            mls_cmd += f" --extra-context {proposal}"
        if workspace:
            mls_cmd += f" --workspace {workspace}"
    else:
        raise ValueError(f"unknown kind: {kind}")
    if extra_args:
        mls_cmd += f" {extra_args}"

    # Multi-GPU tasks (GPU_OVERRIDE) must expose ALL pod GPUs to the run. mlsbench sets
    # CUDA_VISIBLE_DEVICES=config.gpu_devices, and the default config.yaml is "0" (1 GPU),
    # so a 2-GPU pod would still see only GPU 0. Point them at config.<N>gpu.yaml
    # (gpu_devices="0,..,N-1"). Applies to baseline + agent alike.
    _g = _task_gpus(task)
    if _g > 1:
        mls_cmd += f" --config configs/config.{_g}gpu.yaml"

    return f"""set -o pipefail
export PATH={MINICONDA}/bin:$PATH
export MLSBENCH_LOG_LABEL={tag}
export HOME=/root
{MINICONDA}/bin/python {HARNESS_ROOT}/scripts/anthropic_maas_shim.py --port 18791 &
SHIM_PID=$!
sleep 6
curl -sS -m 10 http://127.0.0.1:18791/health || (echo SHIM_DEAD; exit 91)
cd {MLS_ROOT}
{mls_cmd}
RC=$?
kill $SHIM_PID 2>/dev/null
echo "MLS_RUN_EXIT=$RC"
exit $RC
"""


def submit_job(name: str, command: str, *, max_minutes: int, cpus: int = 16,
               memory: str = "100Gi", config_file: str | None = None,
               gpus: int = 1) -> str:
    cmd_file = Path("/tmp") / f"mls_dispatch_{os.getpid()}_{time.monotonic_ns()}.sh"
    cmd_file.write_text(command)
    try:
        extra = ["--config", config_file] if config_file else []
        _pai([
            "create-job", "--name", name, *extra,
            "--workspace-id", WORKSPACE_ID, "--resource-id", QUOTA_ID,
            "--image", IMAGE,
            "--command-file", str(cmd_file),
            "--gpus", str(gpus), "--cpus", str(cpus), "--memory", memory,
            "--shared-memory", "32Gi", "--pod-count", "1",
            "--max-running-minutes", str(max_minutes),
            "--priority", "9",   # L20Z is the priority pool (user 2026-07); schedule ahead of others
            "--enable-rdma", "false",
        ], parse=False)
        # create-job returns an empty body; resolve the JobId by listing (with retry — list-jobs
        # has indexing lag, so a just-submitted job may not appear on the first poll).
        for attempt in range(12):
            try:
                jobs = _pai(["list-jobs", "--set", f"WorkspaceId={WORKSPACE_ID}",
                             "--set", "PageSize=100", "--compact"])["body"].get("Jobs", [])
            except Exception as exc:  # noqa: BLE001 — PAI list-jobs times out under farm load; the
                # create-job already succeeded (job runs), so NEVER crash the batch on a resolution
                # timeout — retry, then fall through to untracked.
                print(f"[warn] list-jobs failed (attempt {attempt+1}/12, retrying): {str(exc)[:100]}",
                      flush=True)
                time.sleep(5)
                continue
            for j in jobs:
                if j.get("DisplayName") == name and j.get("Status") not in TERMINAL:
                    return j["JobId"]
            time.sleep(5)
        # The job WAS submitted (create-job succeeded) and will run + write to the leaderboard;
        # we just couldn't resolve its tracking JobId under API lag. Do NOT crash the whole batch —
        # warn and continue (scoring reads the leaderboard, not this tracking id).
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


def tier_tasks(tier: str, tasks_arg: str | None) -> list[str]:
    if tasks_arg:
        slugs = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    else:
        slugs = json.loads(LITE_JSON.read_text())["slugs"]
    if tier == "fast":
        return [s for s in slugs if s not in HEAVY_TASKS]
    if tier == "heavy":
        return [s for s in slugs if s in HEAVY_TASKS]
    return slugs  # full


def default_minutes(task: str, kind: str) -> int:
    # L20Z ≈ half an H100 → pad generously; agent runs include LLM turns + up to
    # max_tests training rounds.
    if task in HEAVY_TASKS:
        return 2880 if kind == "baseline" else 4320   # 48 h / 72 h
    return 720 if kind == "baseline" else 1440        # 12 h / 24 h


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_submit_one(args) -> None:
    tag = args.tag or f"{args.kind}_{args.task}_s{args.seed or 'x'}"
    if args.proposal and not str(Path(args.proposal).resolve()).startswith("/newcpfs"):
        sys.exit(f"--proposal must live under /newcpfs (pods can't see local disks): {args.proposal}")
    ws = args.workspace
    if args.kind == "agent" and not ws:
        ws = str(batch_dir(args.batch) / "ws" / tag)
    command = build_command(
        args.kind, args.task, seed=args.seed, model=args.model, mode=args.mode,
        proposal=args.proposal, workspace=ws, tag=tag, extra_args=args.extra_args,
    )
    minutes = args.max_minutes or default_minutes(args.task, args.kind)
    name = f"mls-{tag}"[:60].replace("_", "-")
    _g = _task_gpus(args.task)
    job_id = submit_job(name, command, max_minutes=minutes,
                        config_file=_task_config_file(args.task),
                        gpus=_g, cpus=16 * _g, memory=_task_mem(args.task, _g))
    entry = {"job_id": job_id, "name": name, "kind": args.kind, "task": args.task,
             "seed": args.seed, "model": args.model, "tag": tag, "workspace": ws,
             "proposal": args.proposal, "status": "Submitted",
             "submitted_at": datetime.now(timezone.utc).isoformat()}
    jobs = load_jobs(args.batch)
    jobs.append(entry)
    save_jobs(args.batch, jobs)
    print(job_id)


def cmd_submit_batch(args) -> None:
    tasks = tier_tasks(args.tier, args.tasks)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [None]
    arms = args.arms.split(",") if args.arms else [None]
    proposals: dict[str, dict[str, str]] = {}
    if args.kind == "agent" and not args.no_proposal:
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
                if args.kind == "baseline":
                    tag = f"bl_{task}_s{seed}"
                else:
                    tag = f"{arm}_{task}_s{seed}"
                if tag in seen_tags:
                    print(f"[skip] {tag} already in batch")
                    continue
                proposal_path = None
                if args.kind == "agent" and not args.no_proposal:
                    text = strip_think(proposals[arm].get(task, ""))
                    if not text:
                        print(f"[warn] no proposal (or empty after <think>-strip) for {arm}/{task}, skipping")
                        continue
                    pfile = batch_dir(args.batch) / "proposals" / f"{tag}.md"
                    pfile.parent.mkdir(parents=True, exist_ok=True)
                    pfile.write_text(text + args.hparam_suffix)
                    proposal_path = str(pfile)
                ws = str(batch_dir(args.batch) / "ws" / tag) if args.kind == "agent" else None
                # Arm-distinct leaderboard tag: worker stays the same upstream model,
                # but 'mlsbench score' sees one row-group per arm. The shim strips
                # the "__<arm>" suffix when routing.
                model = args.model
                if args.kind == "agent" and arm and not args.no_arm_tag:
                    model = f"{args.model}__{arm}"
                command = build_command(
                    args.kind, task, seed=seed, model=model, mode=args.mode,
                    proposal=proposal_path, workspace=ws, tag=tag,
                )
                minutes = args.max_minutes or default_minutes(task, args.kind)
                name = f"mls-{tag}"[:60].replace("_", "-")
                while True:
                    in_flight = sum(1 for j in jobs if j["status"] not in TERMINAL)
                    if in_flight < args.max_concurrent:
                        break
                    print(f"[throttle] {in_flight} in flight >= {args.max_concurrent}; polling...")
                    refresh_statuses(jobs)
                    save_jobs(args.batch, jobs)
                    time.sleep(args.poll_interval)
                _g = _task_gpus(task)
                job_id = submit_job(name, command, max_minutes=minutes,
                                    config_file=_task_config_file(task),
                                    gpus=_g, cpus=16 * _g, memory=_task_mem(task, _g))
                entry = {"job_id": job_id, "name": name, "kind": args.kind, "task": task,
                         "seed": seed, "arm": arm, "model": args.model, "tag": tag,
                         "workspace": ws, "proposal": proposal_path, "status": "Submitted",
                         "submitted_at": datetime.now(timezone.utc).isoformat()}
                jobs.append(entry)
                seen_tags.add(tag)
                save_jobs(args.batch, jobs)
                n_submitted += 1
                print(f"[submit] {tag} -> {job_id}")
    print(f"\nbatch '{args.batch}': {n_submitted} new jobs, {len(jobs)} total "
          f"(state: {batch_dir(args.batch)/'jobs.json'})")


def refresh_statuses(jobs: list[dict]) -> None:
    for j in jobs:
        if j["status"] in TERMINAL:
            continue
        if not j.get("job_id"):
            j["status"] = "Untracked"  # submitted but JobId unresolved; runs + writes leaderboard
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
        if j["status"] not in TERMINAL:
            print(f"stopping {j['tag']} ({j['job_id']})")
            _pai(["stop-job", "--job-id", j["job_id"]], parse=False)
    refresh_statuses(jobs)
    save_jobs(args.batch, jobs)


# ---------------------------------------------------------------------------
# vLLM serving on the farm
# ---------------------------------------------------------------------------

VLLM_ENV = "qwen-serving"   # shared conda env: vllm 0.20.1 / transformers 5.7 / torch 2.11+cu130
VLLM_DRIVER = "580.95.05-open"  # cu130 wheels need driver >= 580 (pods default to 535)


def _driver_config_file(out_name: str) -> str:
    """Write a pai config = base infra + the newer GPU driver (580)."""
    import yaml
    cfg = {}
    base = Path.home() / ".pai_config.yaml"
    if base.exists():
        cfg = yaml.safe_load(base.read_text()) or {}
    cfg.setdefault("settings", {})["Driver"] = VLLM_DRIVER
    out = Path("/tmp") / out_name
    out.write_text(yaml.safe_dump(cfg))
    return str(out)


def _serve_config_file() -> str:
    """pai config for serving jobs: base infra + the newer GPU driver."""
    return _driver_config_file("mls_serve_pai_config.yaml")


def _task_config_file(task: str) -> str | None:
    """Driver-580 config for cu130 task envs; None for the default-535 majority."""
    if task in CU130_TASKS:
        return _driver_config_file("mls_task580_pai_config.yaml")
    return None


def cmd_serve(args) -> None:
    """Launch a vLLM OpenAI-compatible server as a long-running DLC job."""
    ckpt = str(Path(args.ckpt).resolve())
    if not ckpt.startswith("/newcpfs"):
        sys.exit(f"--ckpt must live under /newcpfs (pods can't see local disks): {ckpt}")
    tag = args.tag or f"serve_{args.name}"
    command = f"""set -o pipefail
source {MINICONDA}/etc/profile.d/conda.sh
conda activate {VLLM_ENV}
export HOME=/root
export VLLM_USE_DEEP_GEMM=0
nvidia-smi | head -4
vllm serve {ckpt} --served-model-name {args.name} \
  --host 0.0.0.0 --port {args.port} \
  --tensor-parallel-size {args.gpus} \
  --gpu-memory-utilization 0.92 --max-model-len {args.max_model_len} \
  --dtype bfloat16
"""
    name = f"mls-{tag}"[:60].replace("_", "-")
    job_id = submit_job(name, command, max_minutes=args.max_minutes,
                        cpus=16 * args.gpus, memory=f"{100 * args.gpus}Gi",
                        gpus=args.gpus, config_file=_serve_config_file())
    jobs = load_jobs(args.batch)
    jobs.append({"job_id": job_id, "name": name, "kind": "serve", "task": args.name,
                 "seed": None, "model": args.name, "tag": tag, "ckpt": ckpt,
                 "port": args.port, "status": "Submitted",
                 "submitted_at": datetime.now(timezone.utc).isoformat()})
    save_jobs(args.batch, jobs)
    print(job_id)


def cmd_endpoint(args) -> None:
    """Print the ip:port of a serving job and probe /v1/models."""
    jobs = [j for j in load_jobs(args.batch) if j["kind"] == "serve"]
    if args.tag:
        jobs = [j for j in jobs if j["tag"] == args.tag]
    if not jobs:
        sys.exit("no serving jobs recorded in this batch")
    for j in jobs:
        body = _pai(["get-job", "--job-id", j["job_id"]])["body"]
        status = body.get("Status")
        pods = [p for p in body.get("Pods", []) if p.get("Type") != "aimaster"]
        ip = None
        if pods:
            # Prefer the DSW-routable VPC leg (net0, 10.39.0.0/17) over the
            # overlay (eth0, 22.x) and the API's top-level Ip (11.x, unroutable
            # from DSW).
            pod_ips = {e.get("InterfaceName"): e.get("Ip")
                       for e in pods[0].get("PodIps", [])}
            ip = pod_ips.get("net0") or pod_ips.get("eth0") or pods[0].get("Ip")
        url = f"http://{ip}:{j['port']}/v1" if ip else None
        line = f"{j['tag']}: status={status} endpoint={url}"
        if url and status == "Running" and not args.no_probe:
            probe = subprocess.run(
                ["curl", "-sS", "-m", "10", f"{url}/models"],
                capture_output=True, text=True)
            ok = probe.returncode == 0 and '"id"' in probe.stdout
            line += f" probe={'OK' if ok else 'UNREACHABLE'}"
            if ok:
                line += f" ({probe.stdout.strip()[:120]})"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("submit-one", help="submit a single run")
    p1.add_argument("--kind", required=True, choices=["baseline", "agent"])
    p1.add_argument("--task", required=True)
    p1.add_argument("--seed", type=int, default=None)
    p1.add_argument("--model", default="claude-fable-5")
    p1.add_argument("--mode", default="eng", choices=["eng", "sci"])
    p1.add_argument("--proposal", default=None, help="context file (MUST be under /newcpfs)")
    p1.add_argument("--workspace", default=None)
    p1.add_argument("--tag", default=None)
    p1.add_argument("--batch", default="adhoc")
    p1.add_argument("--max-minutes", type=int, default=None)
    p1.add_argument("--extra-args", default="", help="extra raw mlsbench args")
    p1.set_defaults(func=cmd_submit_one)

    p2 = sub.add_parser("submit-batch", help="fan out tasks x arms x seeds")
    p2.add_argument("--kind", required=True, choices=["baseline", "agent"])
    p2.add_argument("--tier", default="fast", choices=["fast", "full", "heavy"])
    p2.add_argument("--tasks", default=None, help="comma slugs override")
    p2.add_argument("--seeds", default="1", help="comma seeds (baseline) / replicate ids (agent)")
    p2.add_argument("--arms", default=None, help="comma arms (agent kind)")
    p2.add_argument("--proposals-dir", default=None)
    p2.add_argument("--hparam-suffix", default="", help="text appended to each proposal ctx")
    p2.add_argument("--model", default="claude-fable-5")
    p2.add_argument("--no-arm-tag", action="store_true",
                    help="do NOT append __<arm> to the leaderboard model tag")
    p2.add_argument("--no-proposal", action="store_true",
                    help="native-agent arm: NO --extra-context (worker solves from MLS-Bench's native "
                         "prompt, no master proposal). Pair with --mode sci to match the original benchmark.")
    p2.add_argument("--mode", default="eng", choices=["eng", "sci"])
    p2.add_argument("--batch", required=True)
    # 64 (was 96): at 96 concurrent pods the shared MaaS gateway degrades on large
    # (~33k-token) requests and returns empty completions (~36% flaky). 64 lowers the
    # empty rate; defense-in-depth with the shim's empty-retry+backoff. Reliability>throughput.
    p2.add_argument("--max-concurrent", type=int, default=64)
    p2.add_argument("--max-minutes", type=int, default=None)
    p2.add_argument("--poll-interval", type=int, default=60)
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

    p5 = sub.add_parser("serve", help="launch a vLLM OpenAI server job on the farm")
    p5.add_argument("--ckpt", required=True, help="model dir under /newcpfs")
    p5.add_argument("--name", required=True, help="served model name")
    p5.add_argument("--gpus", type=int, default=1)
    p5.add_argument("--port", type=int, default=8000)
    p5.add_argument("--max-model-len", type=int, default=16384)
    p5.add_argument("--max-minutes", type=int, default=10080, help="default 7 days")
    p5.add_argument("--tag", default=None)
    p5.add_argument("--batch", default="serving")
    p5.set_defaults(func=cmd_serve)

    p6 = sub.add_parser("endpoint", help="show serving job endpoint(s) + probe")
    p6.add_argument("--batch", default="serving")
    p6.add_argument("--tag", default=None)
    p6.add_argument("--no-probe", action="store_true")
    p6.set_defaults(func=cmd_endpoint)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

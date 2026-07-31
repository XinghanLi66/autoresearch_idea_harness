#!/usr/bin/env python3
"""MLS-Bench QS (QuickSilver 532 / GB200-aarch64) eval dispatcher — the QS-analog of
mls_cluster_dispatch.py, for when the L20Z farm is unusable (hidden-tenant throttle).

Submits each `mlsbench baseline|agent` invocation as a queue-532 Trial via `qs training create`.
Resource specs (from cc000's live 532 serve trial, confirmed by cc002):
  queue-id 532 · cluster-id 70 (rcjp-gpu) · cloud 小红书 · resource-package-id 234 (4GPU/140C/900Gi)
  worker-num 1 (4 GPU/pod → 1 GPU/task via CUDA_VISIBLE_DEVICES=0) · ARM image tutu_cybertron:...3fs_v1
Per-trial: activate the task's aarch64 conda env on /mnt/3fs, start the fable-5 MaaS shim (agent only),
health-check it, then run one mlsbench invocation reading the proposal .md from 3fs. MaaS keys are
injected via `qs training create --env` (pods have no harness/.env).

ONLY the ARM-PORTABLE task subset runs here (sklearn/numpy/std-torch); gsplat/mujoco/dgl/isaacgym and
heavy robo/diffusion tasks are NOT portable to aarch64 and stay on L20Z/pending.

GATES before real submission (see cc002/cc000):
  1. MaaS egress from QS pods (maas.devops.rednote.life reachable) — else agent arms can't run.
  2. /mnt/3fs/lxh/mlsbench staged (MLS-Bench + miniconda + mlsbench-<pkg> envs) + proposals/*.md present.
Use --dry-run to print the exact `qs training create` commands without submitting.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

# ---- confirmed QS 532 resources ----
QUEUE_ID = "532"
CLUSTER_ID = "70"
CLOUD_NAME = "小红书"
RESOURCE_PACKAGE_ID = "234"     # 4 GPU / 140C / 900Gi
WORKER_NUM = "1"
IMAGE = ("artifactory.devops.xiaohongshu.com/quicksilver/images/training/media/nvidia/"
         "tutu_cybertron:ngc2510_xray_newep_3fs_v1")
MLSBENCH_ROOT = "/mnt/3fs/lxh/mlsbench"          # shared stage on 3fs (cc002)
SHIM = f"{MLSBENCH_ROOT}/harness/scripts/anthropic_maas_shim.py"
PROPOSALS_3FS = f"{MLSBENCH_ROOT}/proposals"     # per-(arm,task) <tag>.md live here
MAAS_ENV_KEYS = ["RUNWAY_FABLE5_MAAS_URL", "RUNWAY_FABLE5_MAAS_API_KEY", "RUNWAY_FABLE5_MAAS_MODEL"]

# task -> aarch64 conda env (mlsbench-<pkg>); package resolved from MLS-Bench config.json test_cmds
TASK_ENV = {
    "ml-clustering-algorithm": "mlsbench-scikit-learn",
    "ml-dimensionality-reduction": "mlsbench-scikit-learn",
    "causal-discovery-discrete": "mlsbench-causal-bnlearn",
    "optimization-multi-objective": "mlsbench-deap",
    "optimization-variance-reduction": "mlsbench-opt-vr-bench",
    "dl-activation-function": "mlsbench-pytorch-vision",
    "cv-pooling-aggregation": "mlsbench-pytorch-vision",
    "ts-imputation": "mlsbench-Time-Series-Library",
    "ts-exogenous-forecast": "mlsbench-Time-Series-Library",
    "security-membership-inference-defense": "mlsbench-pytorch-vision",
}
CPU_TASKS = {"ml-clustering-algorithm", "ml-dimensionality-reduction", "causal-discovery-discrete",
             "optimization-multi-objective", "optimization-variance-reduction"}  # Phase-1, no GPU


def maas_env_str() -> str:
    """Read MaaS keys from the local harness .env → 'K=V,K=V' for qs --env (format TBD-confirm w/ cc002)."""
    env = {}
    for cand in (".env", "harness/.env"):
        p = Path(cand)
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in MAAS_ENV_KEYS:
                    env[k.strip()] = v.strip().strip('"')
    return ",".join(f"{k}={v}" for k, v in env.items())


TEMPLATE_TRIAL_ID = "1724258"  # cc000's proven 532 serve trial; inherit image/cloud/cluster/pack/mount


def build_command(kind: str, task: str, arm: str, seed: int) -> str:
    env_name = TASK_ENV.get(task, "mlsbench-base")
    tag = f"{arm}_{task}"          # proposal is per (arm,task); seed is the eval seed, not a proposal variant
    lines = [
        "set -o pipefail",
        f"source {MLSBENCH_ROOT}/miniconda3/etc/profile.d/conda.sh",
        f"conda activate {env_name}",
        "export CUDA_VISIBLE_DEVICES=0",
        f"cd {MLSBENCH_ROOT}/MLS-Bench",
    ]
    if kind == "agent":
        # start shim (worker=fable-5) + health-gate, then implement THIS proposal
        lines += [
            f"python3 {SHIM} --port 18791 & sleep 6",
            'curl -sf 127.0.0.1:18791/health || { echo "shim/MaaS unreachable"; exit 91; }',
            f"mlsbench agent {task} --model claude-fable-5 --mode eng "
            f"--extra-context {PROPOSALS_3FS}/{tag}.md --seed {seed}",
        ]
    else:  # baseline is proposal-independent + needs no worker/shim
        lines += [f"mlsbench baseline {task} --seed {seed} --resume"]
    return "\n".join(lines)


def submit_one(kind: str, task: str, arm: str, seed: int, priority: str, dry: bool) -> None:
    name = f"mls-{arm}-{task}-s{seed}" if kind == "agent" else f"mls-bl-{task}-s{seed}"
    command = build_command(kind, task, arm, seed)
    # --from-trial-id inherits image/cloud/cluster/resource-pack/mount (cc002); override name/cmd/env/priority.
    # qs training create has NO --command-file → pass cmd via a temp file: --command "$(cat f)".
    if dry:
        print(f"  $ qs training create --from-trial-id {TEMPLATE_TRIAL_ID} --name {name} "
              f"--priority {priority} --env '<MaaS 3 keys>' --command \"$(cat <tmpl>)\"")
        print("    " + command.replace("\n", "\n    "))
        return
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(command); tmpl = fh.name
    cmd = ["qs", "training", "create", "--from-trial-id", TEMPLATE_TRIAL_ID,
           "--name", name, "--priority", priority, "--env", maas_env_str(),
           "--command", Path(tmpl).read_text()]
    rc = subprocess.run(cmd).returncode
    os.unlink(tmpl)
    print(f"  [{'ok' if rc == 0 else 'FAIL'}] {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["agent", "baseline"], required=True)
    ap.add_argument("--arms", default="d1sft,d1rl,m2sft", help="comma arms (agent only)")
    ap.add_argument("--tasks", default=None, help="comma tasks (default: all portable)")
    ap.add_argument("--phase1-cpu", action="store_true", help="only the 5 CPU-portable tasks")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--priority", default="0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = (args.tasks.split(",") if args.tasks
             else sorted(CPU_TASKS) if args.phase1_cpu else list(TASK_ENV))
    arms = args.arms.split(",") if args.kind == "agent" else ["baseline"]
    seeds = list(range(1, args.seeds + 1))
    print(f"== QS dispatch: kind={args.kind} tasks={len(tasks)} arms={arms} seeds={seeds} "
          f"({'DRY-RUN' if args.dry_run else 'LIVE'}) ==")
    if not args.dry_run and not maas_env_str():
        sys.exit("no MaaS env keys found in .env — cannot submit agent trials")
    for task in tasks:
        for arm in arms:
            for s in seeds:
                submit_one(args.kind, task, arm, s, args.priority, args.dry_run)


if __name__ == "__main__":
    main()

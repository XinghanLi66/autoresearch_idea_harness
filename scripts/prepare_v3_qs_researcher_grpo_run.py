#!/usr/bin/env python3
"""Prepare a QS GRPO RL run (single GPU) for the researcher-CoT policy.

Generates runs/qs_researcher_grpo/<run_id>/ with qs_command_grpo.sh (pod-side script),
qs_create_dry_run.sh and qs_create_job.sh (guarded by CONFIRM_SUBMIT_V3_QS_GRPO=1).
The pod: clones autoresearch_idea_harness@V3, pip-installs sentence-transformers,
verifies policy/data/reward-heads paths on /mnt/3fs, runs train_v3_researcher_cot_grpo.py
on GPU 0, and validates the summary (status ok + reward logged).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autoresearch_idea_harness.io import load_config, write_json  # noqa: E402


def _safe(v: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", v).strip("_") or "qs_grpo"


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_researcher_grpo"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--policy-model", required=True, help="remote dir: full-SFT checkpoint (or base for smoke)")
    ap.add_argument("--train-jsonl", default="/mnt/3fs/lxh/agentic-training/data/researcher_cot/anchored_v1/train.jsonl")
    ap.add_argument("--reward-heads-dir", default="/mnt/3fs/lxh/agentic-training/data/reward_heads_v1")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1400)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--sleep-seconds", type=int, default=120)
    args = ap.parse_args()

    cfg = load_config(args.config)
    qs = dict(cfg.get("v3_training", {}).get("qs") or {})
    repo = next(r for r in qs["git_repos"] if r["name"] == "autoresearch_idea_harness")
    run_id = _safe(args.run_id)
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    remote_root = str(qs["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/autoresearch_idea_harness/runs/qs/qs_researcher_grpo/{run_id}"
    clone_dir = f"$REMOTE_RUN_DIR/src/autoresearch_idea_harness"
    out_dir = f"{remote_run_dir}/rl_output"
    head = _git_head()

    trainer_args = (
        f"  --policy-model {shlex.quote(args.policy_model)} \\\n"
        f"  --train-jsonl {shlex.quote(args.train_jsonl)} \\\n"
        f"  --reward-heads-dir {shlex.quote(args.reward_heads_dir)} \\\n"
        f"  --output-dir {shlex.quote(out_dir)} \\\n"
        f"  --max-steps {args.max_steps} --prompts-per-step {args.prompts_per_step} \\\n"
        f"  --group-size {args.group_size} --max-new-tokens {args.max_new_tokens} \\\n"
        f"  --kl-coef {args.kl_coef} --lr {args.lr} --save-steps {args.save_steps} \\\n"
        f"  --micro-batch {args.micro_batch}"
    )

    command = f"""#!/usr/bin/env bash
set -euo pipefail
export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
export HF_HOME="${{HF_HOME:-{remote_root}/hf_cache}}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF="${{PYTORCH_ALLOC_CONF:-expandable_segments:True}}"
mkdir -p "$REMOTE_RUN_DIR/src" "$HF_HOME"
cd "$REMOTE_RUN_DIR"
echo "[qs-grpo] start $(date -Is) run_id=$RUN_ID host=$(hostname)" | tee grpo.log
nvidia-smi 2>&1 | tee -a grpo.log
test -d {shlex.quote(args.policy_model)}
test -f {shlex.quote(args.train_jsonl)}
test -f {shlex.quote(args.reward_heads_dir)}/reward_heads.npz
test -f {shlex.quote(args.reward_heads_dir)}/reward_heads_meta.json

rm -rf {clone_dir}
git clone --depth 1 --branch {shlex.quote(repo["ref"])} {shlex.quote(repo["url"])} {clone_dir} 2>&1 | tee -a grpo.log
cd {clone_dir}
git rev-parse HEAD | tee "$REMOTE_RUN_DIR/git_head.txt"
echo "[qs-grpo] expected_commit={head}" | tee -a "$REMOTE_RUN_DIR/grpo.log"

python3 -c "import transformers, numpy; print('deps ok')" | tee -a "$REMOTE_RUN_DIR/grpo.log"
# NOTE: do not import peft here — on this image peft requires the transformers.modeling_layers
# shim, which the trainer applies before importing peft.
python3 -m py_compile scripts/train_v3_researcher_cot_grpo.py

python3 scripts/train_v3_researcher_cot_grpo.py \\
{trainer_args} 2>&1 | tee -a "$REMOTE_RUN_DIR/grpo.log"

python3 - <<'PY' "$REMOTE_RUN_DIR/result.json" {shlex.quote(out_dir)}
import json, sys
from pathlib import Path
out = Path(sys.argv[2])
summary_path = out / "grpo_summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
events = (out / "events.jsonl")
n_steps = sum(1 for l in events.open() if '"event": "step"' in l) if events.exists() else 0
adapters = sorted(p.name for p in out.glob("adapter-step*") if p.is_dir())
result = {{
    "status": "ok" if (summary and summary.get("status") == "ok" and adapters) else "failed_validation",
    "summary": summary, "logged_steps": n_steps, "adapters": adapters,
}}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps(result, indent=2, ensure_ascii=False))
raise SystemExit(0 if result["status"] == "ok" else 2)
PY
echo "[qs-grpo] sleeping {args.sleep_seconds}s" | tee -a "$REMOTE_RUN_DIR/grpo.log"
sleep {args.sleep_seconds}
"""
    cmd_path = run_dir / "qs_command_grpo.sh"
    cmd_path.write_text(command)
    cmd_path.chmod(0o755)

    def create_script(dry: bool) -> str:
        qargs = ["qs", "training", "create"]
        if dry:
            qargs.append("--dry-run")
        qargs += ["--name", run_id, "--image", str(qs["image"]),
                  "--queue-id", str(qs["queue_id"]), "--cloud-id", str(qs["cloud_id"]),
                  "--cluster-id", str(qs["cluster_id"]), "--resource-package-id", str(qs["resource_package_id"]),
                  "--job-type", str(qs.get("job_type", "PytorchJob")), "--worker-num", "1",
                  "--priority", str(qs.get("priority", 0)), "--yes", "-o", "json", "-q"]
        base = " ".join(shlex.quote(a) for a in qargs) + ' --command "$QS_COMMAND"'
        lines = ["#!/usr/bin/env bash", "set -euo pipefail",
                 f"QS_COMMAND=$(cat {shlex.quote(str(cmd_path))})"]
        if dry:
            lines.append(base)
        else:
            lines += [': "${CONFIRM_SUBMIT_V3_QS_GRPO:?Set to 1 after reviewing dry-run output.}"',
                      f"{base} | tee {shlex.quote(str(run_dir / 'submission.json'))}"]
        return "\n".join(lines) + "\n"

    (run_dir / "qs_create_dry_run.sh").write_text(create_script(True))
    (run_dir / "qs_create_job.sh").write_text(create_script(False))
    for f in ("qs_create_dry_run.sh", "qs_create_job.sh"):
        (run_dir / f).chmod(0o755)

    plan = {"run_id": run_id, "expected_commit": head, "remote_run_dir": remote_run_dir,
            "policy_model": args.policy_model, "train_jsonl": args.train_jsonl,
            "reward_heads_dir": args.reward_heads_dir,
            "grpo": {k: getattr(args, k) for k in ("max_steps", "prompts_per_step", "group_size",
                                                   "max_new_tokens", "kl_coef", "lr", "save_steps", "micro_batch")},
            "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_GRPO=1"}
    write_json(run_dir / "run_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

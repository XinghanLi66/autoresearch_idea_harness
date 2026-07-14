#!/usr/bin/env python3
"""Prepare a QS 32B full-parameter FSDP SFT run.

The generated QS startup command launches `train_v3_researcher_cot_full_fsdp.py`
with torchrun on the four GPUs provided by one research_agent worker. For smoke
runs it also resumes from the first saved FSDP sharded checkpoint and runs one
additional step, which validates save/restore/eval in the same QS trial.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config, write_json  # noqa: E402


DEFAULT_BASE_MODEL = "/mnt/3fs/lxh/agentic-training/models/Qwen2.5-32B-Instruct"
DEFAULT_REMOTE_TRAIN = "/mnt/3fs/lxh/agentic-training/data/researcher_cot/prejudge_v1/train.jsonl"
DEFAULT_REMOTE_VAL = "/mnt/3fs/lxh/agentic-training/data/researcher_cot/prejudge_v1/val.jsonl"


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_full_sft"


def _local_git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            timeout=10,
        ).strip()
    except Exception:
        return "unknown"


def _find_harness_repo(qs_cfg: dict[str, Any]) -> dict[str, Any]:
    for repo in qs_cfg.get("git_repos") or []:
        if repo.get("name") == "autoresearch_idea_harness":
            return dict(repo)
    raise SystemExit("v3_training.qs.git_repos must include autoresearch_idea_harness")


def _torchrun_block(
    *,
    nproc_per_node: int,
    master_port: int,
    train_jsonl: str,
    val_jsonl: str | None,
    output_dir: str,
    base_model: str,
    limit_rows: int,
    max_steps: int,
    max_seq_length: int,
    lr: float,
    grad_accum: int,
    save_steps: int,
    eval_steps: int,
    summary_name: str,
    resume_from: str | None = None,
) -> str:
    val_arg = f"  --val-jsonl {shlex.quote(val_jsonl)} \\\n" if val_jsonl else ""
    if resume_from and resume_from.startswith("$"):
        resume_value = resume_from
    elif resume_from:
        resume_value = shlex.quote(resume_from)
    else:
        resume_value = ""
    resume_arg = f"  --resume-from-checkpoint {resume_value} \\\n" if resume_value else ""
    return f"""$TORCHRUN_BIN --standalone --nproc_per_node={nproc_per_node} --master_port={master_port} \\
  scripts/train_v3_researcher_cot_full_fsdp.py \\
  --train-jsonl {shlex.quote(train_jsonl)} \\
{val_arg}\
  --output-dir {shlex.quote(output_dir)} \\
  --base-model {shlex.quote(base_model)} \\
  --limit {limit_rows} \\
  --max-steps {max_steps} \\
  --max-seq-length {max_seq_length} \\
  --per-device-batch-size 1 \\
  --grad-accum {grad_accum} \\
  --lr {lr} \\
  --save-steps {save_steps} \\
  --eval-steps {eval_steps} \\
  --save-total-limit 3 \\
{resume_arg}\
  --summary-name {shlex.quote(summary_name)}"""


def _command_text(
    *,
    run_id: str,
    qs_cfg: dict[str, Any],
    harness_repo: dict[str, Any],
    expected_commit: str,
    base_model: str,
    remote_train_jsonl: str,
    remote_val_jsonl: str | None,
    limit_rows: int,
    max_steps: int,
    resume_check: bool,
    resume_max_steps: int,
    max_seq_length: int,
    lr: float,
    grad_accum: int,
    nproc_per_node: int,
    master_port: int,
    remote_run_family: str,
) -> str:
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/{remote_run_family}/{run_id}"
    sleep_seconds = int(qs_cfg.get("smoke_sleep_seconds", 60))
    repo_url = str(harness_repo["url"])
    repo_ref = str(harness_repo.get("ref") or "V3")
    clone_dir = f"{remote_run_dir}/src/autoresearch_idea_harness"
    output_dir = f"{remote_run_dir}/output"
    save_steps = 1 if max_steps <= 2 else 50
    eval_steps = 1 if max_steps <= 2 else 50
    first_train = _torchrun_block(
        nproc_per_node=nproc_per_node,
        master_port=master_port,
        train_jsonl=remote_train_jsonl,
        val_jsonl=remote_val_jsonl,
        output_dir=output_dir,
        base_model=base_model,
        limit_rows=limit_rows,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        lr=lr,
        grad_accum=grad_accum,
        save_steps=save_steps,
        eval_steps=eval_steps,
        summary_name="train_summary_initial.json",
    )
    resume_train = _torchrun_block(
        nproc_per_node=nproc_per_node,
        master_port=master_port + 1,
        train_jsonl=remote_train_jsonl,
        val_jsonl=remote_val_jsonl,
        output_dir=output_dir,
        base_model=base_model,
        limit_rows=limit_rows,
        max_steps=resume_max_steps,
        max_seq_length=max_seq_length,
        lr=lr,
        grad_accum=grad_accum,
        save_steps=save_steps,
        eval_steps=eval_steps,
        summary_name="train_summary_resume.json",
        resume_from="$RESUME_CKPT",
    )
    resume_block = ""
    if resume_check:
        resume_block = f"""
RESUME_CKPT=$(python3 - <<'PY' {shlex.quote(output_dir)}
import re
import sys
from pathlib import Path
out = Path(sys.argv[1])
ckpts = []
for path in out.glob("checkpoint-*"):
    if path.is_dir():
        m = re.search(r"checkpoint-(\\d+)$", path.name)
        ckpts.append((int(m.group(1)) if m else -1, path))
if not ckpts:
    raise SystemExit("no checkpoint directory found after initial full-SFT run")
print(str(sorted(ckpts)[-1][1]))
PY
)
echo "[qs-full-sft] resume_ckpt=$RESUME_CKPT" | tee -a "$REMOTE_RUN_DIR/full_sft.log"
{resume_train} 2>&1 | tee -a "$REMOTE_RUN_DIR/full_sft.log"
"""
    return f"""#!/usr/bin/env bash
set -euo pipefail

export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
export HF_HOME="${{HF_HOME:-{remote_root}/hf_cache}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-$HF_HOME/transformers}}"
export HF_HUB_ENABLE_HF_TRANSFER="${{HF_HUB_ENABLE_HF_TRANSFER:-0}}"
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${{TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}}"
export NCCL_DEBUG="${{NCCL_DEBUG:-WARN}}"
export PYTORCH_ALLOC_CONF="${{PYTORCH_ALLOC_CONF:-expandable_segments:True}}"
export TORCHRUN_BIN="${{TORCHRUN_BIN:-torchrun}}"
SMOKE_SLEEP_SECONDS=${{QS_SMOKE_SLEEP_SECONDS:-{sleep_seconds}}}

mkdir -p "$REMOTE_RUN_DIR/src" "$HF_HOME"
cd "$REMOTE_RUN_DIR"

echo "[qs-full-sft] start $(date -Is)" | tee full_sft.log
echo "[qs-full-sft] run_id=$RUN_ID" | tee -a full_sft.log
echo "[qs-full-sft] hostname=$(hostname)" | tee -a full_sft.log
echo "[qs-full-sft] uname=$(uname -a)" | tee -a full_sft.log
echo "[qs-full-sft] expected_commit={shlex.quote(expected_commit)}" | tee -a full_sft.log
echo "[qs-full-sft] base_model={shlex.quote(base_model)}" | tee -a full_sft.log
echo "[qs-full-sft] train_jsonl={shlex.quote(remote_train_jsonl)}" | tee -a full_sft.log
echo "[qs-full-sft] val_jsonl={shlex.quote(remote_val_jsonl or '')}" | tee -a full_sft.log
echo "[qs-full-sft] limit_rows={limit_rows}" | tee -a full_sft.log
echo "[qs-full-sft] max_steps={max_steps}" | tee -a full_sft.log
echo "[qs-full-sft] resume_check={str(resume_check).lower()}" | tee -a full_sft.log
echo "[qs-full-sft] max_seq_length={max_seq_length}" | tee -a full_sft.log
echo "[qs-full-sft] nproc_per_node={nproc_per_node}" | tee -a full_sft.log
echo "[qs-full-sft] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a full_sft.log
echo "[qs-full-sft] TORCHRUN_BIN=$TORCHRUN_BIN" | tee -a full_sft.log
echo "[qs-full-sft] net_ifaces=$(ls /sys/class/net 2>/dev/null | tr '\\n' ',' || true)" | tee -a full_sft.log
if command -v ip >/dev/null 2>&1; then
  ip -o addr show 2>&1 | tee -a full_sft.log || true
fi

select_nccl_iface() {{
  local iface path state
  if [[ -n "${{QS_NCCL_SOCKET_IFNAME:-}}" && -d "/sys/class/net/${{QS_NCCL_SOCKET_IFNAME}}" ]]; then
    printf '%s\\n' "$QS_NCCL_SOCKET_IFNAME"
    return 0
  fi
  for path in /sys/class/net/*; do
    [[ -e "$path" ]] || continue
    iface=$(basename "$path")
    case "$iface" in
      lo|docker*|veth*|cni*|flannel*|tun*|tap*) continue ;;
    esac
    state=$(cat "$path/operstate" 2>/dev/null || true)
    if [[ "$state" == "up" || "$state" == "unknown" ]]; then
      printf '%s\\n' "$iface"
      return 0
    fi
  done
  printf 'lo\\n'
}}
export NCCL_SOCKET_IFNAME="$(select_nccl_iface)"
export GLOO_SOCKET_IFNAME="${{GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}}"
unset NCCL_ASYNC_ERROR_HANDLING || true
unset PYTORCH_CUDA_ALLOC_CONF || true
echo "[qs-full-sft] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" | tee -a full_sft.log
echo "[qs-full-sft] GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME" | tee -a full_sft.log
echo "[qs-full-sft] TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING" | tee -a full_sft.log
echo "[qs-full-sft] PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF" | tee -a full_sft.log

test -d {shlex.quote(base_model)}
test -f {shlex.quote(remote_train_jsonl)}
if [[ -n {shlex.quote(remote_val_jsonl or '')} ]]; then
  test -f {shlex.quote(remote_val_jsonl or '')}
fi

rm -rf {shlex.quote(clone_dir)}
git clone --depth 1 --branch {shlex.quote(repo_ref)} {shlex.quote(repo_url)} {shlex.quote(clone_dir)} 2>&1 | tee -a full_sft.log
cd {shlex.quote(clone_dir)}
git rev-parse HEAD | tee "$REMOTE_RUN_DIR/git_head.txt"

python3 -m py_compile scripts/train_v3_researcher_cot_full_fsdp.py 2>&1 | tee -a "$REMOTE_RUN_DIR/full_sft.log"
$TORCHRUN_BIN --help >/dev/null
nvidia-smi 2>&1 | tee -a "$REMOTE_RUN_DIR/full_sft.log"

{first_train} 2>&1 | tee -a "$REMOTE_RUN_DIR/full_sft.log"
{resume_block}
python3 - <<'PY' "$REMOTE_RUN_DIR/result.json" {shlex.quote(output_dir)} {str(resume_check).lower()!r}
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
resume_required = sys.argv[3] == "true"
initial_path = output_dir / "train_summary_initial.json"
resume_path = output_dir / "train_summary_resume.json"
initial = json.loads(initial_path.read_text()) if initial_path.exists() else None
resume = json.loads(resume_path.read_text()) if resume_path.exists() else None
checkpoints = sorted(p.name for p in output_dir.glob("checkpoint-*") if p.is_dir())
eval_ok = bool(initial and initial.get("eval_metrics"))
if resume_required:
    eval_ok = eval_ok and bool(resume and resume.get("eval_metrics"))
result = {{
    "status": "ok",
    "output_dir": str(output_dir),
    "initial_summary": str(initial_path) if initial else None,
    "resume_summary": str(resume_path) if resume else None,
    "checkpoint_dirs": checkpoints,
    "full_checkpoint_saved": bool(checkpoints),
    "resume_required": resume_required,
    "resume_ok": (not resume_required) or bool(resume),
    "eval_ok": eval_ok,
    "initial": initial,
    "resume": resume,
}}
if not result["full_checkpoint_saved"] or not result["resume_ok"] or not result["eval_ok"]:
    result["status"] = "failed_validation"
result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if result["status"] == "ok" else 2)
PY

echo "[qs-full-sft] sleeping $SMOKE_SLEEP_SECONDS seconds for log/exec inspection" | tee -a "$REMOTE_RUN_DIR/full_sft.log"
sleep "$SMOKE_SLEEP_SECONDS"
echo "[qs-full-sft] done $(date -Is)" | tee -a "$REMOTE_RUN_DIR/full_sft.log"
"""


def _script_text(
    *,
    run_id: str,
    qs_cfg: dict[str, Any],
    command_path: Path,
    submit: bool,
) -> str:
    args = ["qs", "training", "create"]
    if not submit:
        args.append("--dry-run")
    args.extend(
        [
            "--name",
            run_id,
            "--image",
            str(qs_cfg["image"]),
            "--queue-id",
            str(qs_cfg["queue_id"]),
            "--cloud-id",
            str(qs_cfg["cloud_id"]),
            "--cluster-id",
            str(qs_cfg["cluster_id"]),
            "--resource-package-id",
            str(qs_cfg["resource_package_id"]),
            "--job-type",
            str(qs_cfg.get("job_type", "PytorchJob")),
            "--worker-num",
            str(qs_cfg.get("worker_num", 1)),
            "--priority",
            str(qs_cfg.get("priority", 0)),
        ]
    )
    if qs_cfg.get("overuse", True):
        args.append("--overuse")
    args.extend(["--yes", "-o", "json", "-q"])
    prefix = f"QS_COMMAND=$(cat {shlex.quote(str(command_path))})"
    command = " ".join(shlex.quote(a) for a in args) + ' --command "$QS_COMMAND"'
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", prefix]
    if submit:
        lines.extend(
            [
                ': "${CONFIRM_SUBMIT_V3_QS_FULL_SFT:?Set to 1 after reviewing run_plan.json and dry-run output.}"',
                'if [[ "${CONFIRM_SUBMIT_V3_QS_FULL_SFT}" != "1" ]]; then',
                '  echo "CONFIRM_SUBMIT_V3_QS_FULL_SFT must equal 1" >&2',
                "  exit 2",
                "fi",
                f"{command} | tee {shlex.quote(str(command_path.parent / 'submission.json'))}",
            ]
        )
    else:
        lines.append(command)
    return "\n".join(lines) + "\n"


def prepare(
    *,
    config_path: Path,
    output_dir: Path,
    run_id: str | None,
    base_model: str,
    remote_train_jsonl: str,
    remote_val_jsonl: str | None,
    limit_rows: int,
    max_steps: int,
    resume_check: bool,
    resume_max_steps: int,
    max_seq_length: int,
    lr: float,
    grad_accum: int,
    nproc_per_node: int,
    master_port: int,
    remote_run_family: str,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    qs_cfg = dict(cfg.get("v3_training", {}).get("qs") or {})
    required = [
        "queue_id",
        "cloud_id",
        "cluster_id",
        "resource_package_id",
        "image",
        "remote_project_root",
        "git_repos",
    ]
    missing = [key for key in required if not qs_cfg.get(key)]
    if missing:
        raise SystemExit(f"missing v3_training.qs config keys: {', '.join(missing)}")

    if run_id is None:
        run_id = "v3_qs_researcher_full_sft_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    harness_repo = _find_harness_repo(qs_cfg)
    expected_commit = _local_git_head()
    command = _command_text(
        run_id=run_id,
        qs_cfg=qs_cfg,
        harness_repo=harness_repo,
        expected_commit=expected_commit,
        base_model=base_model,
        remote_train_jsonl=remote_train_jsonl,
        remote_val_jsonl=remote_val_jsonl,
        limit_rows=limit_rows,
        max_steps=max_steps,
        resume_check=resume_check,
        resume_max_steps=resume_max_steps,
        max_seq_length=max_seq_length,
        lr=lr,
        grad_accum=grad_accum,
        nproc_per_node=nproc_per_node,
        master_port=master_port,
        remote_run_family=remote_run_family,
    )
    command_path = run_dir / "qs_command_full_sft.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)

    dry_run_path = run_dir / "qs_create_dry_run.sh"
    dry_run_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, command_path=command_path, submit=False))
    dry_run_path.chmod(0o755)

    submit_path = run_dir / "qs_create_job.sh"
    submit_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, command_path=command_path, submit=True))
    submit_path.chmod(0o755)

    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/{remote_run_family}/{run_id}"
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "purpose": "QS V3 32B full-parameter SFT with FSDP full_shard; includes checkpoint resume/eval validation.",
        "qs": {
            "queue_id": int(qs_cfg["queue_id"]),
            "queue_name": qs_cfg.get("queue_name"),
            "cloud_id": int(qs_cfg["cloud_id"]),
            "cloud_name": qs_cfg.get("cloud_name"),
            "cluster_id": int(qs_cfg["cluster_id"]),
            "resource_package_id": int(qs_cfg["resource_package_id"]),
            "resource_package_name": qs_cfg.get("resource_package_name"),
            "worker_num": int(qs_cfg.get("worker_num", 1)),
            "overuse": bool(qs_cfg.get("overuse", False)),
            "image": qs_cfg["image"],
        },
        "source": {
            "repo": harness_repo,
            "expected_commit": expected_commit,
        },
        "training": {
            "base_model": base_model,
            "remote_train_jsonl": remote_train_jsonl,
            "remote_val_jsonl": remote_val_jsonl,
            "limit": limit_rows,
            "max_steps": max_steps,
            "resume_check": resume_check,
            "resume_max_steps": resume_max_steps,
            "max_seq_length": max_seq_length,
            "lr": lr,
            "grad_accum": grad_accum,
            "nproc_per_node": nproc_per_node,
            "fsdp": "full_shard auto_wrap",
            "remote_run_family": remote_run_family,
        },
        "artifacts": {
            "command": str(command_path),
            "dry_run": str(dry_run_path),
            "submit": str(submit_path),
            "expected_remote_run_dir": remote_run_dir,
            "expected_remote_result": f"{remote_run_dir}/result.json",
            "expected_remote_log": f"{remote_run_dir}/full_sft.log",
        },
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_FULL_SFT=1",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_researcher_full_sft"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--remote-train-jsonl", default=DEFAULT_REMOTE_TRAIN)
    parser.add_argument("--remote-val-jsonl", default=DEFAULT_REMOTE_VAL)
    parser.add_argument("--limit-rows", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--resume-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-max-steps", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--remote-run-family", default="qs_researcher_full_sft")
    args = parser.parse_args()
    summary = prepare(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        base_model=args.base_model,
        remote_train_jsonl=args.remote_train_jsonl,
        remote_val_jsonl=args.remote_val_jsonl,
        limit_rows=args.limit_rows,
        max_steps=args.max_steps,
        resume_check=args.resume_check,
        resume_max_steps=args.resume_max_steps,
        max_seq_length=args.max_seq_length,
        lr=args.lr,
        grad_accum=args.grad_accum,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        remote_run_family=args.remote_run_family,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

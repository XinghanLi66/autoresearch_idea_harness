#!/usr/bin/env python3
"""Prepare a QS run that executes the real V3 researcher-CoT LoRA script.

This smoke is heavier than `qs_v3_bootstrap_smoke.py`: it uses the actual
`train_v3_researcher_cot_lora.py` entrypoint and a small public Qwen Instruct
model, but limits training to one step and skips merge/generation.
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


DEFAULT_SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_lora_smoke"


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


def _synthetic_rows_json() -> str:
    rows = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Richard Sutton, a world-class AI researcher. "
                        "Given a research situation, reason toward one genuinely novel idea."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "It is 1988. Learning from delayed reward is sample-inefficient. "
                        "Find a better way to learn useful predictions before final outcomes."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I would make predictions learn from later predictions. The Core idea "
                        "is temporal-difference bootstrapping: update the present estimate toward "
                        "a partly learned future estimate instead of waiting for a final label. "
                        "The Non-trivial crux is accepting controlled bias in exchange for a much "
                        "denser learning signal."
                    ),
                },
            ],
            "researcher": "Richard Sutton",
            "source": "qs_real_lora_smoke",
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Kaiming He, a world-class AI researcher. "
                        "Given a research situation, reason toward one genuinely novel idea."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "It is 2015. As convolutional nets get much deeper, training error gets worse. "
                        "Find an architecture change that makes depth easier to optimize."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I would let each block learn only the residual correction. The Core idea "
                        "is an identity skip path, so a deep block can preserve the current solution "
                        "and learn a delta. The Non-trivial crux is that this is an optimization "
                        "intervention, not just an ensemble or parameter-count trick."
                    ),
                },
            ],
            "researcher": "Kaiming He",
            "source": "qs_real_lora_smoke",
        },
    ]
    return json.dumps(rows, ensure_ascii=False)


def _command_text(
    run_id: str,
    qs_cfg: dict[str, Any],
    harness_repo: dict[str, Any],
    expected_commit: str,
    base_model: str,
) -> str:
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_researcher_lora_smoke/{run_id}"
    sleep_seconds = int(qs_cfg.get("smoke_sleep_seconds", 60))
    repo_url = str(harness_repo["url"])
    repo_ref = str(harness_repo.get("ref") or "V3")
    clone_dir = f"{remote_run_dir}/src/autoresearch_idea_harness"
    train_jsonl = f"{remote_run_dir}/data/train.jsonl"
    output_dir = f"{remote_run_dir}/output"
    rows_json = _synthetic_rows_json()
    return f"""#!/usr/bin/env bash
set -euo pipefail

export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
export HF_HOME="${{HF_HOME:-{remote_root}/hf_cache}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-$HF_HOME/transformers}}"
export HF_HUB_ENABLE_HF_TRANSFER="${{HF_HUB_ENABLE_HF_TRANSFER:-0}}"
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0}}"
SMOKE_SLEEP_SECONDS=${{QS_SMOKE_SLEEP_SECONDS:-{sleep_seconds}}}

mkdir -p "$REMOTE_RUN_DIR/src" "$REMOTE_RUN_DIR/data" "$HF_HOME"
cd "$REMOTE_RUN_DIR"

echo "[qs-lora-smoke] start $(date -Is)" | tee lora_smoke.log
echo "[qs-lora-smoke] run_id=$RUN_ID" | tee -a lora_smoke.log
echo "[qs-lora-smoke] hostname=$(hostname)" | tee -a lora_smoke.log
echo "[qs-lora-smoke] uname=$(uname -a)" | tee -a lora_smoke.log
echo "[qs-lora-smoke] expected_commit={shlex.quote(expected_commit)}" | tee -a lora_smoke.log
echo "[qs-lora-smoke] base_model={shlex.quote(base_model)}" | tee -a lora_smoke.log
echo "[qs-lora-smoke] HF_HOME=$HF_HOME" | tee -a lora_smoke.log
echo "[qs-lora-smoke] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a lora_smoke.log

rm -rf {shlex.quote(clone_dir)}
git clone --depth 1 --branch {shlex.quote(repo_ref)} {shlex.quote(repo_url)} {shlex.quote(clone_dir)} 2>&1 | tee -a lora_smoke.log
cd {shlex.quote(clone_dir)}
git rev-parse HEAD | tee "$REMOTE_RUN_DIR/git_head.txt"

python -m py_compile scripts/train_v3_researcher_cot_lora.py 2>&1 | tee -a "$REMOTE_RUN_DIR/lora_smoke.log"

python - <<'PY' {shlex.quote(train_jsonl)}
import json
import sys
rows = json.loads({rows_json!r})
with open(sys.argv[1], "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")
print(json.dumps({{"train_jsonl": sys.argv[1], "rows": len(rows)}}, ensure_ascii=False))
PY

python scripts/train_v3_researcher_cot_lora.py \
  --train-jsonl {shlex.quote(train_jsonl)} \
  --output-dir {shlex.quote(output_dir)} \
  --base-model {shlex.quote(base_model)} \
  --max-steps 1 \
  --limit 2 \
  --per-device-batch-size 1 \
  --grad-accum 1 \
  --max-seq-length 256 \
  --no-merge \
  --skip-gen-check 2>&1 | tee -a "$REMOTE_RUN_DIR/lora_smoke.log"

python - <<'PY' "$REMOTE_RUN_DIR/result.json" {shlex.quote(output_dir)}
import json
import sys
from pathlib import Path
summary_path = Path(sys.argv[2]) / "train_summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {{}}
result = {{
    "status": "ok" if summary else "missing_train_summary",
    "output_dir": sys.argv[2],
    "train_summary": summary,
}}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if summary else 2)
PY

echo "[qs-lora-smoke] sleeping $SMOKE_SLEEP_SECONDS seconds for log/exec inspection" | tee -a "$REMOTE_RUN_DIR/lora_smoke.log"
sleep "$SMOKE_SLEEP_SECONDS"
echo "[qs-lora-smoke] done $(date -Is)" | tee -a "$REMOTE_RUN_DIR/lora_smoke.log"
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
    args.extend([
        "--name", run_id,
        "--image", str(qs_cfg["image"]),
        "--queue-id", str(qs_cfg["queue_id"]),
        "--cloud-id", str(qs_cfg["cloud_id"]),
        "--cluster-id", str(qs_cfg["cluster_id"]),
        "--resource-package-id", str(qs_cfg["resource_package_id"]),
        "--job-type", str(qs_cfg.get("job_type", "PytorchJob")),
        "--worker-num", str(qs_cfg.get("worker_num", 1)),
        "--priority", str(qs_cfg.get("priority", 0)),
    ])
    if qs_cfg.get("overuse", True):
        args.append("--overuse")
    args.extend(["--yes", "-o", "json", "-q"])
    prefix = f"QS_COMMAND=$(cat {shlex.quote(str(command_path))})"
    command = " ".join(shlex.quote(a) for a in args) + ' --command "$QS_COMMAND"'
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", prefix]
    if submit:
        lines.extend([
            ': "${CONFIRM_SUBMIT_V3_QS_LORA_SMOKE:?Set to 1 after reviewing run_plan.json and dry-run output.}"',
            'if [[ "${CONFIRM_SUBMIT_V3_QS_LORA_SMOKE}" != "1" ]]; then',
            '  echo "CONFIRM_SUBMIT_V3_QS_LORA_SMOKE must equal 1" >&2',
            "  exit 2",
            "fi",
            f"{command} | tee {shlex.quote(str(command_path.parent / 'submission.json'))}",
        ])
    else:
        lines.append(command)
    return "\n".join(lines) + "\n"


def prepare(config_path: Path, output_dir: Path, run_id: str | None, base_model: str) -> dict[str, Any]:
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
        run_id = "v3_qs_researcher_lora_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    harness_repo = _find_harness_repo(qs_cfg)
    expected_commit = _local_git_head()
    command = _command_text(run_id, qs_cfg, harness_repo, expected_commit, base_model)
    command_path = run_dir / "qs_command_lora_smoke.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)

    dry_run_path = run_dir / "qs_create_dry_run.sh"
    dry_run_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, command_path=command_path, submit=False))
    dry_run_path.chmod(0o755)

    submit_path = run_dir / "qs_create_job.sh"
    submit_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, command_path=command_path, submit=True))
    submit_path.chmod(0o755)

    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_researcher_lora_smoke/{run_id}"
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "purpose": "QS V3 real-script smoke: run train_v3_researcher_cot_lora.py for one step with a small Qwen model.",
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
            "max_steps": 1,
            "limit": 2,
            "max_seq_length": 256,
            "no_merge": True,
            "skip_gen_check": True,
        },
        "artifacts": {
            "command": str(command_path),
            "dry_run": str(dry_run_path),
            "submit": str(submit_path),
            "expected_remote_run_dir": remote_run_dir,
            "expected_remote_result": f"{remote_run_dir}/result.json",
            "expected_remote_log": f"{remote_run_dir}/lora_smoke.log",
        },
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_LORA_SMOKE=1",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_researcher_lora_smoke"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--base-model", default=DEFAULT_SMALL_MODEL)
    args = parser.parse_args()
    summary = prepare(Path(args.config), Path(args.output_dir), args.run_id, args.base_model)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

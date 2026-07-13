#!/usr/bin/env python3
"""Prepare a QS smoke run that verifies pod-side source access.

This is the second QS migration smoke: after queue/GPU/filesystem validation,
it checks whether a QS pod can reach GitHub over HTTPS and shallow-clone the
small source repos into /mnt/3fs without relying on local DSW mounts.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config, write_json  # noqa: E402


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_code_smoke"


def _command_text(run_id: str, qs_cfg: dict[str, Any]) -> str:
    remote_root = str(qs_cfg.get("remote_code_smoke_root") or qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/{run_id}"
    repos = qs_cfg.get("git_repos") or []
    sleep_seconds = int(qs_cfg.get("smoke_sleep_seconds", 60))
    repos_json = json.dumps(repos, ensure_ascii=False)
    return f"""#!/usr/bin/env bash
set -euo pipefail

export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
SMOKE_SLEEP_SECONDS=${{QS_SMOKE_SLEEP_SECONDS:-{sleep_seconds}}}

mkdir -p "$REMOTE_RUN_DIR"
cd "$REMOTE_RUN_DIR"

echo "[qs-code-smoke] start $(date -Is)" | tee code_smoke.log
echo "[qs-code-smoke] run_id=$RUN_ID" | tee -a code_smoke.log
echo "[qs-code-smoke] pwd=$(pwd)" | tee -a code_smoke.log
echo "[qs-code-smoke] hostname=$(hostname)" | tee -a code_smoke.log
echo "[qs-code-smoke] uname=$(uname -a)" | tee -a code_smoke.log

cat > code_smoke.py <<'PY'
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

REPOS = json.loads({repos_json!r})
ROOT = Path(os.environ["REMOTE_RUN_DIR"])


def run(cmd: list[str], log_path: Path, timeout: int = 600) -> dict[str, object]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout
        returncode = proc.returncode
        error = None
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\\n[TIMEOUT]\\n" + (exc.stderr or "")
        returncode = 124
        error = f"timeout after {{timeout}}s"
    log_path.write_text(output or "")
    return {{
        "command": cmd,
        "returncode": returncode,
        "ok": returncode == 0,
        "seconds": round(time.time() - start, 3),
        "log": str(log_path),
        "error": error,
    }}


def main() -> None:
    result: dict[str, object] = {{
        "status": "running",
        "run_id": os.environ.get("RUN_ID"),
        "remote_run_dir": str(ROOT),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "repos": [],
    }}
    result["git_version"] = run(["git", "--version"], ROOT / "git_version.log", timeout=30)
    result["df_3fs"] = run(["df", "-h", "/mnt/3fs"], ROOT / "df_3fs.log", timeout=30)

    all_ok = bool(result["git_version"]["ok"]) and bool(result["df_3fs"]["ok"])  # type: ignore[index]
    for repo in REPOS:
        name = str(repo["name"])
        url = str(repo["url"])
        ref = str(repo.get("ref") or "HEAD")
        clone_dir = ROOT / name
        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        repo_result: dict[str, object] = {{
            "name": name,
            "url": url,
            "ref": ref,
            "clone_dir": str(clone_dir),
        }}
        repo_result["ls_remote"] = run(
            ["git", "ls-remote", "--heads", url, ref],
            ROOT / f"{{name}}.ls_remote.log",
            timeout=120,
        )
        repo_result["clone"] = run(
            ["git", "clone", "--depth", "1", "--branch", ref, url, str(clone_dir)],
            ROOT / f"{{name}}.clone.log",
            timeout=900,
        )
        if bool(repo_result["clone"]["ok"]):  # type: ignore[index]
            repo_result["rev_parse"] = run(
                ["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
                ROOT / f"{{name}}.rev_parse.log",
                timeout=30,
            )
            compile_check = repo.get("compile_check")
            if compile_check:
                check_path = clone_dir / str(compile_check)
                if check_path.exists():
                    repo_result["compile_check"] = run(
                        ["python", "-m", "py_compile", str(check_path)],
                        ROOT / f"{{name}}.py_compile.log",
                        timeout=120,
                    )
                else:
                    repo_result["compile_check"] = {{
                        "ok": False,
                        "returncode": 2,
                        "error": f"missing compile_check path: {{check_path}}",
                    }}
        repo_ok = all(
            bool(repo_result.get(key, {{}}).get("ok"))
            for key in ["ls_remote", "clone"]
        )
        if "compile_check" in repo_result:
            repo_ok = repo_ok and bool(repo_result["compile_check"]["ok"])  # type: ignore[index]
        repo_result["ok"] = repo_ok
        all_ok = all_ok and repo_ok
        result["repos"].append(repo_result)  # type: ignore[union-attr]

    result["status"] = "ok" if all_ok else "failed"
    (ROOT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
PY

python code_smoke.py 2>&1 | tee -a code_smoke.log
echo "[qs-code-smoke] sleeping $SMOKE_SLEEP_SECONDS seconds for log/exec inspection" | tee -a code_smoke.log
sleep "$SMOKE_SLEEP_SECONDS"
echo "[qs-code-smoke] done $(date -Is)" | tee -a code_smoke.log
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
            ': "${CONFIRM_SUBMIT_V3_QS_CODE_SMOKE:?Set to 1 after reviewing run_plan.json and queue availability.}"',
            'if [[ "${CONFIRM_SUBMIT_V3_QS_CODE_SMOKE}" != "1" ]]; then',
            '  echo "CONFIRM_SUBMIT_V3_QS_CODE_SMOKE must equal 1" >&2',
            "  exit 2",
            "fi",
            f"{command} | tee {shlex.quote(str(command_path.parent / 'submission.json'))}",
        ])
    else:
        lines.append(command)
    return "\n".join(lines) + "\n"


def prepare(config_path: Path, output_dir: Path, run_id: str | None) -> dict[str, Any]:
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
        run_id = "v3_qs_code_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    command = _command_text(run_id, qs_cfg)
    command_path = run_dir / "qs_command_code_smoke.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)

    dry_run_path = run_dir / "qs_create_dry_run.sh"
    dry_run_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, command_path=command_path, submit=False))
    dry_run_path.chmod(0o755)

    submit_path = run_dir / "qs_create_job.sh"
    submit_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, command_path=command_path, submit=True))
    submit_path.chmod(0o755)

    remote_root = str(qs_cfg.get("remote_code_smoke_root") or qs_cfg["remote_project_root"]).rstrip("/")
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "purpose": "QS V3 migration code smoke: verify pod-side HTTPS git access and shallow clones into /mnt/3fs.",
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
        "repos": qs_cfg["git_repos"],
        "artifacts": {
            "command": str(command_path),
            "dry_run": str(dry_run_path),
            "submit": str(submit_path),
            "expected_remote_result": f"{remote_root}/{run_id}/result.json",
            "expected_remote_log": f"{remote_root}/{run_id}/code_smoke.log",
        },
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_CODE_SMOKE=1",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_code_smoke"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    summary = prepare(Path(args.config), Path(args.output_dir), args.run_id)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

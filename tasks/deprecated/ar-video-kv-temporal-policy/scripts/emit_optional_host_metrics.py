#!/usr/bin/env python3
"""Optionally emit a standardized HOST_METRICS line for video task scripts.

Default behavior is a no-op. This keeps replay-first task scripts unchanged
unless the caller explicitly provides one of the supported sidecar sources.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent.parent
SUMMARIZER = TASK_DIR / "scripts" / "summarize_external_validation.py"


def emit_from_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    host_lines = [line for line in lines if line.startswith("HOST_METRICS:")]
    if host_lines:
        print(host_lines[-1])
        return
    payload = json.loads(text)
    print("HOST_METRICS:", json.dumps(payload, sort_keys=True))


def emit_from_args(args_blob: str) -> None:
    command = [sys.executable, str(SUMMARIZER), *shlex.split(args_blob)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    output = result.stdout.strip()
    if output:
        print(output)


def main() -> int:
    host_metrics_line = os.environ.get("HOST_METRICS_LINE", "").strip()
    if host_metrics_line:
        print(host_metrics_line)
        return 0

    host_metrics_file = os.environ.get("HOST_METRICS_FILE", "").strip()
    if host_metrics_file:
        emit_from_file(Path(host_metrics_file).expanduser())
        return 0

    host_summarizer_args = os.environ.get("HOST_SUMMARIZER_ARGS", "").strip()
    if host_summarizer_args:
        emit_from_args(host_summarizer_args)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

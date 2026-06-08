#!/usr/bin/env python3
"""
MLS-Bench evaluation runner — called by run.sh in the worker workspace.

Reads editable_region.py (the worker's function implementation), splices it
into the training template, runs the evaluation, and writes result.json with
an HMAC signature so the harness can verify the result is genuine.

Usage (from worker workspace):
  python run_mls_eval.py \
    --editable editable_region.py \
    --template /path/to/custom_template.py \
    --edit-start 246 --edit-end 269 \
    --subtasks '["resnet20-cifar10"]' \
    --data-root /path/to/data \
    --python /path/to/python \
    --gpu "${CUDA_VISIBLE_DEVICES:-0}" \
    --out-json result.json
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


_FALLBACK_SUBTASK_MAP: dict[str, dict] = {
    "resnet20-cifar10": {"arch": "resnet20", "dataset": "cifar10", "epochs": 200, "data_subdir": "cifar"},
    "resnet56-cifar100": {"arch": "resnet56", "dataset": "cifar100", "epochs": 200, "data_subdir": "cifar"},
    "mobilenetv2-fmnist": {"arch": "mobilenetv2", "dataset": "fmnist", "epochs": 200, "data_subdir": "fmnist"},
}


def splice_edit(template_path: Path, edit_start: int, edit_end: int,
                replacement: str) -> str:
    """Replace lines edit_start..edit_end (1-indexed, inclusive) in template."""
    lines = template_path.read_text().splitlines()
    new_lines = (
        lines[:edit_start - 1]
        + replacement.rstrip("\n").splitlines()
        + lines[edit_end:]
    )
    return "\n".join(new_lines) + "\n"


def run_subtask(
    script_text: str,
    script_name: str,
    spec: dict,
    data_root: Path,
    python_bin: str,
    gpu: str,
    seed: int,
    log_path: Path,
) -> tuple[float | None, str]:
    """Run one subtask in a temp dir, stream output to log_path, return test_acc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / script_name
        script_path.write_text(script_text)
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        data_subdir = spec.get("data_subdir") or ("fmnist" if spec.get("dataset") == "fmnist" else "cifar")
        cmd = [python_bin, str(script_path)]
        if spec.get("arch"):
            cmd += ["--arch", str(spec["arch"])]
        if spec.get("dataset") and not spec.get("multitask"):
            cmd += ["--dataset", str(spec["dataset"])]
        if spec.get("imbalance_ratio") is not None:
            cmd += ["--imbalance-ratio", str(spec["imbalance_ratio"])]
        cmd = [
            *cmd,
            "--data-root", str(data_root / data_subdir),
            "--epochs", str(int(spec.get("epochs", 200))),
            "--batch-size", str(int(spec.get("batch_size", 128))),
            "--lr", str(spec.get("lr", 0.1)),
            "--momentum", str(spec.get("momentum", 0.9)),
            "--weight-decay", str(spec.get("weight_decay", 5e-4)),
            "--seed", str(seed),
            "--output-dir", str(output_dir),
        ]
        extra_args = spec.get("extra_args") or []
        if extra_args:
            cmd.extend([str(x) for x in extra_args])
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}

        output_parts: list[str] = []
        with open(log_path, "a", buffering=1) as log_f:
            log_f.write("[runner] command: " + " ".join(cmd) + "\n")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=tmpdir, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                output_parts.append(line)
            proc.wait()

        output = "".join(output_parts)
        for line in reversed(output.splitlines()):
            m = re.search(r"TEST_METRICS:.*test_acc=([\d.]+)", line)
            if m:
                return float(m.group(1)), output

        return None, output


def _load_specs(subtasks: list[str], raw_specs: str | None) -> dict[str, dict]:
    if raw_specs:
        rows = json.loads(raw_specs)
        return {row["label"]: row for row in rows}
    return {
        label: _FALLBACK_SUBTASK_MAP.get(label, {"label": label, "epochs": 200, "data_subdir": "cifar"})
        for label in subtasks
    }


def _normalize_spec(label: str, spec: dict) -> dict:
    out = {**spec, "label": label}
    if "data_subdir" not in out:
        out["data_subdir"] = "fmnist" if out.get("dataset") == "fmnist" else "cifar"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--editable", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--edit-start", type=int, required=True)
    parser.add_argument("--edit-end", type=int, required=True)
    parser.add_argument("--subtasks", default='["resnet20-cifar10"]')
    parser.add_argument("--subtask-specs", default=None,
                        help="JSON list of {label, arch, dataset, data_subdir, ...} rows.")
    parser.add_argument("--script-name", default="custom_schedule.py")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="result.json")
    args = parser.parse_args()
    gpu = args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    editable = Path(args.editable).read_text()
    template = Path(args.template)
    out_json = Path(args.out_json)
    log_path = out_json.parent / "eval.log"
    subtasks: list[str] = json.loads(args.subtasks)
    specs_by_label = _load_specs(subtasks, args.subtask_specs)

    # Splice worker's implementation into the training script
    try:
        modified_script = splice_edit(template, args.edit_start, args.edit_end, editable)
    except Exception as exc:
        out_json.write_text(json.dumps({"val_metric": None, "error": f"splice failed: {exc}"}))
        sys.exit(1)

    results: dict[str, float] = {}
    errors: dict[str, str] = {}

    for label in subtasks:
        spec = specs_by_label.get(label)
        if spec is None:
            errors[label] = f"unknown subtask: {label}"
            continue
        spec = _normalize_spec(label, spec)

        print(f"\n=== Running {label} ===", flush=True)
        with open(log_path, "a") as f:
            f.write(f"\n{'='*60}\n=== {label} ===\n{'='*60}\n")

        acc, output = run_subtask(
            script_text=modified_script,
            script_name=args.script_name,
            spec=spec,
            data_root=Path(args.data_root),
            python_bin=args.python,
            gpu=gpu,
            seed=args.seed,
            log_path=log_path,
        )

        if acc is not None:
            results[label] = acc
            print(f"  {label}: test_acc={acc:.2f}%", flush=True)
        else:
            tail = output[-500:] if output else "(no output)"
            errors[label] = f"TEST_METRICS not found\n{tail}"
            print(f"  {label}: FAILED", file=sys.stderr, flush=True)

    # Determine primary metric
    primary_label = subtasks[0]
    primary_acc = results.get(primary_label)

    if primary_acc is None:
        err_msg = errors.get(primary_label, "unknown error")
        out_json.write_text(json.dumps({"val_metric": None, "error": err_msg}))
        sys.exit(1)

    # Sign result with HMAC so harness can verify it was produced by this script
    secret = os.environ.get("BENCHMARK_HMAC_KEY", "benchmark-eval-secret")
    payload = f"{primary_acc:.6f}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    result = {"val_metric": primary_acc, "_sig": sig}
    result.update({f"test_acc_{k}": v for k, v in results.items()})
    if errors:
        result["subtask_errors"] = errors

    out_json.write_text(json.dumps(result))
    print(f"\nResult written → {out_json}: val_metric={primary_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()

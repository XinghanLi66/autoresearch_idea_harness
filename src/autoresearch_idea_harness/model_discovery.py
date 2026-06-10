from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .io import write_json
from .model_registry import infer_model_size_b


SKIP_DIR_NAMES = {
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    "data",
    "dataset",
    "datasets",
    "log",
    "logs",
    "runs",
    "tmp",
    "trash",
    "wandb",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _depth(root: Path, path: Path) -> int:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return 999
    if str(rel) == ".":
        return 0
    return len(rel.parts)


def _guess_hf_id(path: Path) -> str | None:
    parts = list(path.parts)
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx > 0:
            model_dir = parts[idx - 1]
            if model_dir.startswith("models--"):
                return model_dir.removeprefix("models--").replace("--", "/")
            return model_dir
    for part in reversed(parts):
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return None


def _model_name(path: Path, hf_id: str | None) -> str:
    if hf_id:
        return hf_id
    if path.parent.name == "snapshots" and path.parent.parent.name:
        return path.parent.parent.name
    return path.name


def _has_tokenizer(path: Path) -> bool:
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json"):
        if (path / name).exists():
            return True
    return False


def _summarize_config(path: Path, root: Path) -> dict[str, Any]:
    config = _load_json(path / "config.json")
    hf_id = _guess_hf_id(path)
    model_id = _model_name(path, hf_id)
    record = {"name": model_id, "hf_id": hf_id}
    size_b = infer_model_size_b(model_id, path, record)
    architectures = config.get("architectures")
    if isinstance(architectures, str):
        architectures = [architectures]
    elif not isinstance(architectures, list):
        architectures = []
    return {
        "model_id_guess": model_id,
        "hf_id_guess": hf_id,
        "path": str(path),
        "root": str(root),
        "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "size_b": size_b,
        "formal_32b_candidate": bool(size_b is not None and 30.0 <= size_b <= 40.0),
        "model_type": config.get("model_type"),
        "architectures": architectures[:4],
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "has_tokenizer": _has_tokenizer(path),
        "release_date": None,
        "release_date_source": "manual_required",
    }


def discover_local_base_models(
    roots: list[str | Path],
    max_depth: int = 6,
    include_all: bool = False,
    target_min_b: float = 30.0,
    target_max_b: float = 40.0,
) -> dict[str, Any]:
    started = time.time()
    normalized_roots = [Path(root).expanduser() for root in roots]
    models: list[dict[str, Any]] = []
    missing_roots: list[str] = []
    scanned_dirs = 0
    errors: list[dict[str, str]] = []

    for root in normalized_roots:
        if not root.exists():
            missing_roots.append(str(root))
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            path = Path(dirpath)
            current_depth = _depth(root, path)
            if current_depth > max_depth:
                dirnames[:] = []
                continue
            scanned_dirs += 1
            if "config.json" in filenames:
                try:
                    models.append(_summarize_config(path, root))
                except Exception as exc:
                    errors.append({"path": str(path), "error": str(exc)})
            if current_depth >= max_depth:
                dirnames[:] = []
            else:
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)

    def _sort_key(item: dict[str, Any]) -> tuple[float, str]:
        size = item.get("size_b")
        size_f = float(size) if size is not None else -1.0
        return (-size_f, str(item.get("model_id_guess") or item.get("path")))

    candidates = [
        item for item in models
        if item.get("size_b") is not None and target_min_b <= float(item["size_b"]) <= target_max_b
    ]
    candidates = sorted(candidates, key=_sort_key)
    all_models = sorted(models, key=_sort_key)
    size_buckets = Counter()
    for item in all_models:
        size = item.get("size_b")
        if size is None:
            size_buckets["unknown"] += 1
        elif float(size) < target_min_b:
            size_buckets["below_target"] += 1
        elif float(size) <= target_max_b:
            size_buckets["target_30b_40b"] += 1
        else:
            size_buckets["above_target"] += 1

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - started, 3),
        "roots": [str(root) for root in normalized_roots],
        "missing_roots": missing_roots,
        "max_depth": max_depth,
        "target_size_range_b": [target_min_b, target_max_b],
        "counts": {
            "scanned_dirs": scanned_dirs,
            "config_count": len(models),
            "candidate_count": len(candidates),
            "error_count": len(errors),
        },
        "size_buckets": dict(size_buckets),
        "candidates": candidates,
        "all_models": all_models if include_all else all_models[:40],
        "all_models_truncated": not include_all and len(all_models) > 40,
        "errors": errors[:40],
        "notes": [
            "Discovery is read-only and does not register a model.",
            "release_date is intentionally left null; fill it from a trusted model card or release record before training.",
        ],
    }


def render_discovery_markdown(report: dict[str, Any]) -> str:
    counts = report.get("counts") or {}
    lines = [
        "# V3 Base Model Candidates",
        "",
        "This is a read-only local discovery report. It does not register any model for training.",
        "",
        "## Summary",
        "",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Roots: {len(report.get('roots') or [])}",
        f"- Scanned dirs: {counts.get('scanned_dirs', 0)}",
        f"- HF config files: {counts.get('config_count', 0)}",
        f"- 30B-40B candidates: {counts.get('candidate_count', 0)}",
        f"- Errors: {counts.get('error_count', 0)}",
        f"- Size buckets: `{json.dumps(report.get('size_buckets') or {}, ensure_ascii=False)}`",
        "",
        "## Search Roots",
        "",
    ]
    for root in report.get("roots") or []:
        missing = root in set(report.get("missing_roots") or [])
        suffix = " missing" if missing else ""
        lines.append(f"- `{root}`{suffix}")

    lines.extend(["", "## 30B-40B Candidates", ""])
    candidates = report.get("candidates") or []
    if not candidates:
        lines.append("No local 30B-40B model candidates were found in the configured roots.")
    else:
        lines.append("| Model guess | Size | Tokenizer | Path |")
        lines.append("|---|---:|---:|---|")
        for item in candidates:
            lines.append(
                f"| `{item.get('model_id_guess')}` | {item.get('size_b')}B | "
                f"{'yes' if item.get('has_tokenizer') else 'no'} | `{item.get('path')}` |"
            )

    lines.extend(["", "## Other Detected Models", ""])
    for item in report.get("all_models") or []:
        if item.get("formal_32b_candidate"):
            continue
        size = f"{item.get('size_b')}B" if item.get("size_b") is not None else "unknown"
        lines.append(f"- `{item.get('model_id_guess')}`: {size}, `{item.get('path')}`")
    if report.get("all_models_truncated"):
        lines.append("- Other detected models truncated in this report; rerun with `--include-all` for full JSON.")

    lines.extend([
        "",
        "## Manual Registration Step",
        "",
        "Before formal V3 training, choose one candidate and add it to `configs/default.yaml` under "
        "`base_models.registry` with `path`, trusted `release_date`, `size_b`, and `smoke_only: false`.",
        "Do not infer `release_date` from the local path.",
        "",
    ])
    return "\n".join(lines)


def write_discovery_report(report: dict[str, Any], output_json: Path, output_md: Path | None = None) -> None:
    write_json(output_json, report)
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_discovery_markdown(report))

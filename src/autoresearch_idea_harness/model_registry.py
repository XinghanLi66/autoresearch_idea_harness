from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any


MODEL_SIZE_RE = re.compile(r"(?<![a-zA-Z0-9])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])")


def _date_text(value: str | date | None) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _parse_date(value: str | date | None) -> date:
    if not value:
        raise ValueError("base model release_date is required")
    text = _date_text(value)
    try:
        parsed = date.fromisoformat((text or "")[:10])
    except Exception as exc:
        raise ValueError(f"invalid base model release_date {value!r}; expected YYYY-MM-DD") from exc
    if parsed > date.today():
        raise ValueError(f"base model release_date {value!r} is in the future")
    return parsed


def _registry(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_models = cfg.get("base_models") or {}
    registry = base_models.get("registry") or {}
    if not isinstance(registry, dict):
        return {}
    return registry


def _load_model_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except Exception:
        return {}


def _size_from_text(*parts: str | None) -> float | None:
    for part in parts:
        if not part:
            continue
        m = MODEL_SIZE_RE.search(part)
        if m:
            return float(m.group(1))
    return None


def _estimate_size_from_config(config: dict[str, Any]) -> float | None:
    try:
        hidden = int(config.get("hidden_size") or 0)
        inter = int(config.get("intermediate_size") or 0)
        layers = int(config.get("num_hidden_layers") or 0)
        vocab = int(config.get("vocab_size") or 0)
    except Exception:
        return None
    if min(hidden, inter, layers) <= 0:
        return None
    # Approximate decoder-only dense transformer parameters. It is intentionally
    # conservative: this is a validator, not an accounting tool.
    attn = 4 * hidden * hidden
    mlp = 3 * hidden * inter
    norms = 2 * hidden
    embeddings = vocab * hidden
    total = layers * (attn + mlp + norms) + embeddings
    if not config.get("tie_word_embeddings", False):
        total += embeddings
    return round(total / 1e9, 2)


def infer_model_size_b(model_id: str, path: Path, record: dict[str, Any]) -> float | None:
    explicit = record.get("size_b")
    if explicit is not None:
        try:
            return float(explicit)
        except Exception:
            pass
    text_size = _size_from_text(model_id, str(path), record.get("name"), record.get("hf_id"))
    if text_size is not None:
        return text_size
    return _estimate_size_from_config(_load_model_config(path))


def resolve_base_model(
    cfg: dict[str, Any],
    model_id: str | None = None,
    model_path: str | None = None,
    release_date: str | None = None,
    require_32b: bool = True,
    allow_missing_model: bool = False,
    allow_smoke_model: bool = False,
) -> dict[str, Any]:
    """Resolve and validate base-model metadata for leakage-safe V3 training."""
    record: dict[str, Any] = {}
    if model_id:
        registry = _registry(cfg)
        if model_id not in registry:
            available = ", ".join(sorted(registry)) or "(empty registry)"
            raise KeyError(f"unknown base model id {model_id!r}; available: {available}")
        record = dict(registry[model_id] or {})
    path_text = model_path or record.get("path")
    is_hub_id = False
    if not path_text and record.get("hf_id"):
        # No local path configured: prefer $MODEL_DIR/<hf_id> when MODEL_DIR is
        # set, otherwise fall back to the HuggingFace hub id itself so that
        # transformers can download the model on demand.
        model_dir = os.environ.get("MODEL_DIR")
        candidate = Path(model_dir) / str(record["hf_id"]) if model_dir else None
        if candidate is not None and candidate.exists():
            path_text = str(candidate)
        else:
            path_text = str(record["hf_id"])
            is_hub_id = True
    path = Path(path_text or "")
    if not path_text:
        raise ValueError("base model path is required")
    resolved_release = release_date or record.get("release_date")
    parsed_date = _parse_date(resolved_release)
    if release_date and record.get("release_date") and _date_text(release_date) != _date_text(record.get("release_date")):
        raise ValueError(
            f"release date mismatch for {model_id}: CLI={release_date}, registry={record.get('release_date')}"
        )
    if record.get("smoke_only") and not allow_smoke_model:
        raise ValueError(f"base model {model_id!r} is marked smoke_only; pass --allow-smoke-model for tests")
    config_exists = (path / "config.json").exists()
    if not config_exists and not allow_missing_model and not is_hub_id:
        raise FileNotFoundError(f"{path}/config.json not found")

    size_b = infer_model_size_b(model_id or path.name, path, record)
    if require_32b:
        if size_b is None:
            raise ValueError(
                "could not infer base model size; set base_models.registry.<id>.size_b or pass --allow-non-32b"
            )
        if not (30.0 <= size_b <= 40.0):
            raise ValueError(f"expected a 30B-40B base model for V3, got inferred size {size_b}B")

    return {
        "model_id": model_id,
        "path": str(path),
        "release_date": parsed_date.isoformat(),
        "size_b": size_b,
        "config_exists": config_exists,
        "smoke_only": bool(record.get("smoke_only")),
        "record": _json_safe(record),
    }

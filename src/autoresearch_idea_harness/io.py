from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_env_file(path: str | Path | None = None) -> None:
    """Load local KEY=VALUE secrets without overriding existing environment."""
    env_path = Path(path) if path else project_root() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Config keys that hold filesystem paths. Relative values are resolved against
# the repository root so the default config works from any checkout location.
_TOP_LEVEL_PATH_KEYS = (
    "project_root",
    "repo_root",
    "proposal_rl_root",
    "mls_bench_root",
    "mle_data_dir",
    "benchmark_python",
    "classified_papers",
    "dataset_dir",
    "arxiv_root",
    "runs_dir",
)


def _expand_path(value: Any) -> Any:
    """Expand ~ and ${ENV_VAR} references; resolve relative paths to repo root."""
    if not isinstance(value, str) or not value:
        return value
    expanded = os.path.expandvars(os.path.expanduser(value))
    p = Path(expanded)
    if not p.is_absolute():
        p = (project_root() / p).resolve()
    return str(p)


def _resolve_config_paths(cfg: dict[str, Any]) -> None:
    for key in _TOP_LEVEL_PATH_KEYS:
        if cfg.get(key):
            cfg[key] = _expand_path(cfg[key])
    base_models = cfg.get("base_models") or {}
    roots = base_models.get("search_roots")
    if isinstance(roots, list):
        base_models["search_roots"] = [_expand_path(r) for r in roots if r]
    for record in (base_models.get("registry") or {}).values():
        if isinstance(record, dict) and record.get("path"):
            record["path"] = _expand_path(record["path"])
    v2_3 = cfg.get("v2_3") or {}
    if v2_3.get("formal_sweep_root"):
        v2_3["formal_sweep_root"] = _expand_path(v2_3["formal_sweep_root"])
    end_to_end = cfg.get("end_to_end") or {}
    claude_cmd = end_to_end.get("claude_cmd")
    if isinstance(claude_cmd, str) and ("/" in claude_cmd or claude_cmd.startswith("~")):
        end_to_end["claude_cmd"] = _expand_path(claude_cmd)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_env_file()
    cfg_path = Path(path) if path else project_root() / "configs" / "default.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    _resolve_config_paths(cfg)
    threshold_path = cfg_path.parent / "v2_3_thresholds.json"
    if threshold_path.exists():
        try:
            thresholds = json.loads(threshold_path.read_text())
            cfg.setdefault("v2_3", {})
            cfg["v2_3"].setdefault("task_thresholds", {})
            cfg["v2_3"]["task_thresholds"].update(thresholds.get("task_thresholds", thresholds))
            cfg["v2_3"]["thresholds_path"] = str(threshold_path)
        except Exception:
            pass
    cfg["_config_path"] = str(cfg_path)
    return cfg


def ensure_src_paths(cfg: dict[str, Any]) -> None:
    """Expose reusable legacy modules without making V2 depend on proposal_rl runtime."""
    root = Path(cfg["proposal_rl_root"])
    for p in (root, root / "scripts"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def stable_id(prefix: str, *parts: Any, length: int = 12) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:length]
    return f"{prefix}_{digest}"


def short_text(text: str | None, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def load_dataset_records(dataset_dir: Path, splits: list[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for split in splits:
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            aid = row.get("arxiv_id")
            if aid:
                records[aid] = {**row, "_source_split": split}
    return records

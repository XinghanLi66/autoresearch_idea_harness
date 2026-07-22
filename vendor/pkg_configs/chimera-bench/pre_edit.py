"""Pre-edit operations for chimera-bench package.
1. Inject TRAIN_METRICS into baseline training wrappers so training progress
   is visible as text feedback.
2. Inject TEST_METRICS into evaluation so final metrics are parseable.

Since chimera-bench baselines each have their own chimera_trainer.py and
chimera_evaluate.py, and we run a unified custom_cdr.py template instead,
these pre-edits are minimal — the template itself handles TRAIN_METRICS
and TEST_METRICS printing.
"""

_DYMEAN_EVALUATION_METRICS_SHIM = '''\
"""Bridge dyMEAN's local evaluation namespace to CHIMERA-Bench metrics.

dyMEAN imports evaluation.rmsd from baselines/dymean/evaluation before
chimera_trainer.py imports evaluation.metrics from the top-level package. Once
that local namespace is loaded, evaluation.metrics is resolved inside the local
namespace too. Re-export the benchmark metrics here without adding
evaluation/__init__.py, so dyMEAN's original evaluation modules still work.
"""

import importlib.util as _importlib_util
from pathlib import Path as _Path

_METRICS_PATH = _Path(__file__).resolve().parents[3] / "evaluation" / "metrics.py"
_SPEC = _importlib_util.spec_from_file_location("_chimera_bench_metrics", _METRICS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load CHIMERA metrics from {_METRICS_PATH}")
_MODULE = _importlib_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

__all__ = [name for name in dir(_MODULE) if not name.startswith("_")]
globals().update({name: getattr(_MODULE, name) for name in __all__})
'''

_TMSCORE_REPLACEMENT = '''\
def _tm_score_source_path():
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    for rel in (
        "baselines/dymean/evaluation/TMscore.cpp",
        "baselines/mean/evaluation/TMscore.cpp",
        "baselines/dyab/evaluation/TMscore.cpp",
    ):
        path = root / rel
        if path.exists():
            return path
    return None


def _tm_score_binary_path():
    """Compile and cache the official Zhang lab TMscore binary on demand."""
    import hashlib as _hashlib
    import os as _os
    import subprocess as _subprocess
    import tempfile as _tempfile
    from pathlib import Path as _Path

    if _os.environ.get("CHIMERA_USE_OFFICIAL_TMSCORE", "1") == "0":
        return None
    cached = globals().get("_CHIMERA_TMSCORE_BINARY")
    if cached:
        return cached
    if globals().get("_CHIMERA_TMSCORE_BINARY_FAILED"):
        return None

    source = _tm_score_source_path()
    if source is None:
        globals()["_CHIMERA_TMSCORE_BINARY_FAILED"] = True
        return None

    digest = _hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    build_dir = _Path(_os.environ.get("CHIMERA_TMSCORE_BUILD_DIR", _tempfile.gettempdir()))
    binary = build_dir / f"chimera_bench_TMscore_{digest}"
    if binary.exists() and _os.access(binary, _os.X_OK):
        globals()["_CHIMERA_TMSCORE_BINARY"] = str(binary)
        return str(binary)

    build_dir.mkdir(parents=True, exist_ok=True)
    tmp_binary = build_dir / f".{binary.name}.{_os.getpid()}"
    cmd = ["g++", "-O3", "-ffast-math", "-lm", "-o", str(tmp_binary), str(source)]
    try:
        _subprocess.run(cmd, check=True, stdout=_subprocess.PIPE,
                        stderr=_subprocess.PIPE, text=True, timeout=120)
        _os.chmod(tmp_binary, 0o755)
        _os.replace(tmp_binary, binary)
    except Exception:
        try:
            tmp_binary.unlink(missing_ok=True)
        except Exception:
            pass
        globals()["_CHIMERA_TMSCORE_BINARY_FAILED"] = True
        return None

    globals()["_CHIMERA_TMSCORE_BINARY"] = str(binary)
    return str(binary)


def _write_tmscore_ca_pdb(path, coords):
    coords = np.asarray(coords, dtype=float)
    with open(path, "w") as handle:
        for i, (x, y, z) in enumerate(coords, 1):
            handle.write(
                f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\\n"
            )
        handle.write("TER\\nEND\\n")


def _tm_score_official(pred_coords, true_coords):
    import re as _re
    import subprocess as _subprocess
    import tempfile as _tempfile
    from pathlib import Path as _Path

    binary = _tm_score_binary_path()
    if binary is None:
        return None

    with _tempfile.TemporaryDirectory(prefix="chimera_tmscore_") as tmpdir:
        tmpdir = _Path(tmpdir)
        pred_pdb = tmpdir / "pred.pdb"
        true_pdb = tmpdir / "native.pdb"
        _write_tmscore_ca_pdb(pred_pdb, pred_coords)
        _write_tmscore_ca_pdb(true_pdb, true_coords)
        try:
            proc = _subprocess.run(
                [binary, str(pred_pdb), str(true_pdb)],
                check=True, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
                text=True, timeout=30,
            )
        except Exception:
            return None

    for line in proc.stdout.splitlines():
        if line.startswith("TM-score"):
            match = _re.search(r"=\\s*([0-9]*\\.?[0-9]+)", line)
            if match:
                return float(match.group(1))
    return None


def _tm_score_python_fallback(pred_coords, true_coords):
    """Fast fallback: Zhang-Skolnick sum after corrected Kabsch alignment.

    The official TMscore source uses d0 = 1.24*cbrt(L-15)-1.8 for normal
    protein lengths, plus its documented short-chain lower bound. CDR loops are
    often short enough that this lower bound materially changes the score.
    """
    pred = np.asarray(pred_coords, dtype=float)
    true = np.asarray(true_coords, dtype=float)
    L = min(len(pred), len(true))
    if L == 0:
        return 0.0
    pred = pred[:L]
    true = true[:L]

    l_min = L
    d0_min = 0.168 if l_min <= 19 else 1.24 * np.cbrt(l_min - 15.0) - 1.8
    d0_min += 0.8
    d0 = 0.5 if L <= 21 else 1.24 * np.cbrt(L - 15.0) - 1.8
    d0 = max(d0, d0_min)

    pred_c = pred - pred.mean(axis=0)
    true_c = true - true.mean(axis=0)
    H = pred_c.T @ true_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    pred_aligned = pred_c @ R

    di = np.sqrt(((pred_aligned - true_c) ** 2).sum(axis=1))
    score = (1.0 / (1.0 + (di / d0) ** 2)).sum() / L
    return float(score)


def tm_score(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """TM-score using the official Zhang lab TMscore binary when available."""
    official = _tm_score_official(pred_coords, true_coords)
    if official is not None:
        return float(official)
    return _tm_score_python_fallback(pred_coords, true_coords)
'''

_MEAN_ROBUST_SUMMARY_REPLACEMENT = '''\
    # Exclude clearly failed coordinate generations from structure aggregates.
    # A 50 Angstrom CDR CA RMSD is far outside physically meaningful loop
    # placement errors; sequence metrics remain aggregated over all predictions.
    structure_mask = None
    if all_metrics.get("rmsd"):
        rmsd_vals = np.asarray(all_metrics["rmsd"], dtype=float)
        keep = np.isfinite(rmsd_vals) & (rmsd_vals <= 50.0)
        if keep.any():
            structure_mask = keep

    summary = {}
    for k, v in all_metrics.items():
        if not v:
            continue
        vals = np.asarray(v, dtype=float)
        if k in {"rmsd", "tm_score"} and structure_mask is not None and len(vals) == len(structure_mask):
            vals = vals[structure_mask]
        if len(vals):
            summary[k] = float(np.mean(vals))
'''

_DYMEAN_ROBUST_SUMMARY_REPLACEMENT = '''\
    summary_by_cdr = {}
    for cdr_type, metrics in metrics_by_cdr.items():
        if metrics["ppl"]:
            # Exclude clearly failed coordinate generations from structure
            # aggregates. A 50 Angstrom CDR CA RMSD is far outside physically
            # meaningful loop placement errors; sequence metrics keep all rows.
            structure_mask = None
            if metrics.get("rmsd"):
                rmsd_vals = np.asarray(metrics["rmsd"], dtype=float)
                keep = np.isfinite(rmsd_vals) & (rmsd_vals <= 50.0)
                if keep.any():
                    structure_mask = keep

            summary = {}
            for k, v in metrics.items():
                if not v:
                    continue
                vals = np.asarray(v, dtype=float)
                if k in {"rmsd", "tm_score"} and structure_mask is not None and len(vals) == len(structure_mask):
                    vals = vals[structure_mask]
                if len(vals):
                    summary[k] = float(np.mean(vals))
            summary_by_cdr[cdr_type] = summary
'''

_CHIMERA_UTILS_ROBUST_SUMMARY_REPLACEMENT = '''\
    # Summary with robust structure aggregation. Predictions with CDR CA RMSD
    # above 50 Angstrom are failed coordinate generations and are excluded from
    # RMSD/TM-score means; non-structure metrics retain all evaluated rows.
    rmsd_outlier_cutoff = 50.0
    structure_outlier_ids = set()
    for result in per_complex:
        try:
            rmsd_val = float(result.get("rmsd", float("inf")))
        except (TypeError, ValueError):
            rmsd_val = float("inf")
        if not np.isfinite(rmsd_val) or rmsd_val > rmsd_outlier_cutoff:
            structure_outlier_ids.add(result.get("complex_id"))

    summary = {}
    for k in FULL_METRIC_KEYS:
        values = all_metrics[k]
        if k in {"rmsd", "tm_score"} and structure_outlier_ids:
            filtered = []
            raw = []
            for result in per_complex:
                if k not in result or result[k] == float("inf"):
                    continue
                raw.append(result[k])
                if result.get("complex_id") not in structure_outlier_ids:
                    filtered.append(result[k])
            values = filtered if filtered else raw
        if values:
            summary[k] = mean_std(values)

    if structure_outlier_ids:
        log.info("Excluded %d structure outlier(s) with RMSD > %.1f A from RMSD/TM-score aggregation",
                 len(structure_outlier_ids), rmsd_outlier_cutoff)
'''


OPS = [
    # Replace the simplified Kabsch-only TM-score with the official Zhang lab
    # TMscore binary path used for publication. A corrected Python
    # Zhang-Skolnick fallback remains for environments without a compiler.
    {
        "op": "replace",
        "file": "chimera-bench/evaluation/metrics.py",
        "start_line": 66,
        "end_line": 90,
        "content": _TMSCORE_REPLACEMENT,
    },
    # Robust structure aggregation for the MEAN inline trainer summary.
    {
        "op": "replace",
        "file": "chimera-bench/baselines/mean/chimera_trainer.py",
        "start_line": 261,
        "end_line": 261,
        "content": _MEAN_ROBUST_SUMMARY_REPLACEMENT,
    },
    # Robust structure aggregation for dyMEAN's inline validation/test summary.
    {
        "op": "replace",
        "file": "chimera-bench/baselines/dymean/chimera_trainer.py",
        "start_line": 419,
        "end_line": 422,
        "content": _DYMEAN_ROBUST_SUMMARY_REPLACEMENT,
    },
    # dymean and diffab have a 'data/' subpackage without __init__.py.
    # When PYTHONPATH includes both /workspace/chimera-bench (which has its
    # own 'data/' as a regular package via __init__.py) AND the baseline dir,
    # Python's import resolution picks the regular package at chimera-bench/data,
    # shadowing the namespace 'data' inside baselines/{dymean,diffab}/.
    # 'mean' baseline already ships data/__init__.py upstream so it's unaffected.
    # Add empty __init__.py so dymean/diffab data/ become regular packages and
    # win over chimera-bench/data because they appear first in sys.path
    # (chimera_trainer.py inserts _SCRIPT_DIR at sys.path[0]).
    # 'data' subpackage: dymean/diffab need __init__.py so their local data/
    # wins over chimera-bench/data/ (which has __init__.py and shadows otherwise).
    {"op": "create", "file": "chimera-bench/baselines/dymean/data/__init__.py", "content": ""},
    {"op": "create", "file": "chimera-bench/baselines/diffab/data/__init__.py", "content": ""},
    # 'evaluation' subpackage: chimera-bench/evaluation/ ships __init__.py so
    # it's a REGULAR package, which wins sys.path resolution over dymean's
    # namespace evaluation/ (no __init__.py). Result: `from evaluation.rmsd`
    # in dymean/data/framework_templates.py:11 fails because chimera-bench
    # /evaluation/ doesn't have rmsd.py (it's in dymean/evaluation/).
    # Fix: make dymean/evaluation/ a regular package (add __init__.py) so it
    # wins sys.path[0] (dymean dir is inserted at 0 by chimera_trainer.py:30
    # — but that insert happens AFTER framework_templates.py imports run.
    # Actually: dymean/ is also script-dir which Python prepends automatically
    # to sys.path when running python /path/to/chimera_trainer.py).
    # With dymean/evaluation/__init__.py, dymean wins. Then add a metrics.py
    # shim inside it that re-exports top-level chimera-bench/evaluation/
    # metrics so chimera_trainer.py:47 `from evaluation.metrics` still works.
    {"op": "create", "file": "chimera-bench/baselines/dymean/evaluation/__init__.py", "content": ""},
    {
        "op": "create",
        "file": "chimera-bench/baselines/dymean/evaluation/metrics.py",
        "content": _DYMEAN_EVALUATION_METRICS_SHIM,
    },
    # Robust structure aggregation for full CHIMERA evaluation used by dyMEAN
    # and DiffAb final CSV reporting. This must run before the load_natives
    # replacement below because both mutate baselines/chimera_utils.py.
    {
        "op": "replace",
        "file": "chimera-bench/baselines/chimera_utils.py",
        "start_line": 712,
        "end_line": 717,
        "content": _CHIMERA_UTILS_ROBUST_SUMMARY_REPLACEMENT,
    },
    # Full CHIMERA evaluation loads native structures from complex_features.
    # The prepared MLS-Bench data keeps these files at
    # ${CHIMERA_DATA_ROOT}/complex_features, while the upstream helper expects
    # ${CHIMERA_DATA_ROOT}/processed/complex_features. Fall back to the actual
    # prepared-data location so DiffAb's saved predictions can be evaluated.
    {
        "op": "replace",
        "file": "chimera-bench/baselines/chimera_utils.py",
        "start_line": 448,
        "end_line": 449,
        "content": '''\
    root = Path(data_root) if data_root else get_data_root()
    feat_dir = root / "processed" / "complex_features"
    if not feat_dir.exists():
        alt_feat_dir = root / "complex_features"
        if alt_feat_dir.exists():
            feat_dir = alt_feat_dir
''',
    },
]

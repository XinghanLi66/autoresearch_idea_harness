"""Lagrangian Self-Distillation (LSD) baseline — official ``losses.py`` reference."""

import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parent / "baseline_reference_ops.py"
_spec = importlib.util.spec_from_file_location("baseline_reference_ops", _p)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

_BENCH_LSD = """# MLS-Bench: LSD (slurm_id=0)
export FLOWMAPS_BENCH_SLURM_ID=0
"""

OPS = _mod.OPS + [
    {
        "op": "replace",
        "file": "scripts/bench_env.sh",
        "start_line": 1,
        "end_line": 2,
        "content": _BENCH_LSD,
    },
]

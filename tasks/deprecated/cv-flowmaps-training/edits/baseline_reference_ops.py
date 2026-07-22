"""Shared rigorous replace op: official flow-maps loss terms (Boffi et al., NeurIPS 2025).

Used by ``lsd.edit.py``, ``psd_uniform.edit.py``, and ``psd_midpoint.edit.py`` (same reference code;
``scripts/bench_env.sh`` sets ``FLOWMAPS_BENCH_SLURM_ID`` for LSD vs PSD configs).

Line range ``36–178`` matches ``losses_template.py`` span after ``mid_edit``.
"""

from pathlib import Path

_FILE = "flow-maps/py/common/losses.py"
_CONTENT = (Path(__file__).resolve().parent / "paper_losses.py").read_text()

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 36,
        "end_line": 178,
        "content": _CONTENT,
    },
]

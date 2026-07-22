"""Mid-edit operations for the safe-exploration-fixed-budget task."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py",
        "content": _CUSTOM_PY,
    },
]

"""Mid-edit operations for the dex-retarget task.

Creates two files in the dex-retargeting workspace:
1. custom_optimizer.py — the agent's editable optimizer (from template)
2. evaluate_custom.py — the evaluation harness
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_EVAL_PATH = Path(__file__).parent / "evaluate_template.py"

_CUSTOM_PY = _TEMPLATE_PATH.read_text()
_EVAL_PY = _EVAL_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "dex-retargeting/src/dex_retargeting/custom_optimizer.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "dex-retargeting/evaluate_custom.py",
        "content": _EVAL_PY,
    },
]

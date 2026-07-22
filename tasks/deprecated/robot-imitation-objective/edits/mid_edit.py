"""Mid-edit operations for robot-imitation-objective task.

Applied to the CleanDiffuser workspace after pre_edit, before the agent starts.
Creates train_custom.py -- the agent's editable training file -- from custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "CleanDiffuser/train_custom.py",
        "content": _CUSTOM_PY,
    },
]

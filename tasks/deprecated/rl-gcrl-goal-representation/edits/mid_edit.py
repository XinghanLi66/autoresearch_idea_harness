"""Mid-edit operations for the rl-gcrl-goal-representation task.

Creates custom_train.py in the dual-goal-representations workspace
from the custom_template.py file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "dual-goal-representations/custom_train.py",
        "content": _CUSTOM_PY,
    },
]

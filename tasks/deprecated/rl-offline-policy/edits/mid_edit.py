"""Mid-edit operations for the rl-offline-policy task.

Applied to the fql workspace after pre_edit, before the agent starts.
Creates custom_train.py -- the agent's editable training file -- from custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "fql/custom_train.py",
        "content": _CUSTOM_PY,
    },
]

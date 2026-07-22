"""Mid-edit operations for the jepa-mask-strategy task.

Applied to the eb_jepa workspace after pre_edit, before the agent starts.
Creates custom_mask.py -- the agent's editable training file -- from
custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "eb_jepa/custom_mask.py",
        "content": _CUSTOM_PY,
    },
]

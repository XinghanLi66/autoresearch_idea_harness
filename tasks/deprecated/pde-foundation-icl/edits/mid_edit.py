"""Mid-edit operations for pde-foundation-icl.
Creates src/custom_model.py (editable model) and src/custom_train_eval.py (train+eval script).
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

_TRAIN_EVAL_PATH = Path(__file__).parent / "custom_train_eval.py"
_TRAIN_EVAL_PY = _TRAIN_EVAL_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "VICON/src/custom_model.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "VICON/src/custom_train_eval.py",
        "content": _TRAIN_EVAL_PY,
    },
]

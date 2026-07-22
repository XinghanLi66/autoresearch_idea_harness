"""Mid-edit operations for ai4bio-protein-function-prediction.
Creates DeepProtein/custom_protein.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "DeepProtein/custom_protein.py",
        "content": _CUSTOM_PY,
    },
]

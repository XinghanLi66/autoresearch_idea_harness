"""Mid-edit operations for mol3d-generation.
Creates Uni-3DAR/custom_mol3d.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "Uni-3DAR/custom_mol3d.py",
        "content": _CUSTOM_PY,
    },
]

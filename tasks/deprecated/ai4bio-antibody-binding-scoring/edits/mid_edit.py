"""Mid-edit operations for ai4bio-antibody-binding-scoring.
Creates AbBiBench/custom_abscore.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "AbBiBench/custom_abscore.py",
        "content": _CUSTOM_PY,
    },
]

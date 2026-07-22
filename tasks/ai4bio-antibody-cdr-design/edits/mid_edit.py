"""Mid-edit operations for ai4bio-antibody-cdr-design.
Creates chimera-bench/custom_cdr.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "chimera-bench/custom_cdr.py",
        "content": _CUSTOM_PY,
    },
]

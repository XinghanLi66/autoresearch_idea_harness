"""Mid-edit: create the editable vocoder template."""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

OPS = [
    {
        "op": "create",
        "file": "speechbrain/custom_vocoder.py",
        "content": _CONTENT,
    },
]

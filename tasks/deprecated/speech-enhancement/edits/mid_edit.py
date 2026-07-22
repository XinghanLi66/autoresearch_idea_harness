"""Mid-edit: create the editable speech enhancement template."""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

OPS = [
    {
        "op": "create",
        "file": "speechbrain/custom_speech_enhancement.py",
        "content": _CONTENT,
    },
]

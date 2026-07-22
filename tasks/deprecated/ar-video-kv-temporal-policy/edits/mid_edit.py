"""Mid-edit operations for ar-video-kv-temporal-policy."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "FAR/custom_video_eval.py",
        "content": _CUSTOM_PY,
    },
]

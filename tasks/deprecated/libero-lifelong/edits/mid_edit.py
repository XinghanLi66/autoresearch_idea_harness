"""Mid-edit operations for the libero-lifelong task.

Applied to the LIBERO workspace after pre_edit, before the agent starts.
Creates custom.py — the agent's editable lifelong learning algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "LIBERO/libero/lifelong/algos/custom.py",
        "content": _CUSTOM_PY,
    },
]

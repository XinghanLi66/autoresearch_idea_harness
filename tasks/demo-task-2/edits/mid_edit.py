"""Mid-edit operations for the demo-task-2 task.

Applied to the pytorch-examples workspace after pre_edit, before the agent starts.
Creates main_custom.py — the agent's editable training file — from custom_template.py.

Uses a separate filename (main_custom.py) to avoid conflicts with pre_edit
operations that target the original main.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "pytorch-examples/mnist/main_custom.py",
        "content": _CUSTOM_PY,
    },
]

"""Mid-edit operations for the marl-mixing-network task.

Applied to the epymarl workspace after pre_edit, before the agent starts.
Creates custom.py — the agent's editable mixing network — from custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "epymarl/src/modules/mixers/custom.py",
        "content": _CUSTOM_PY,
    },
]

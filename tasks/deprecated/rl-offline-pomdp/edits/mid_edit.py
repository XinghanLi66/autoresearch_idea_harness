"""Mid-edit operations for the rl-offline-pomdp task.

Applied to the katakomba workspace after pre_edit, before the agent starts.
Creates custom_nethack_pomdp.py — the agent's editable algorithm file — from custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "katakomba/algorithms/small_scale/custom_nethack_pomdp.py",
        "content": _CUSTOM_PY,
    },
]

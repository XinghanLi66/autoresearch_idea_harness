"""Mid-edit operations for the humanoid-ppo-extractor task.

Applied to the humanoid-bench workspace after pre_edit, before the agent starts.
Creates train_custom.py — the agent's editable training file — from custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "humanoid-bench/ppo/train_custom.py",
        "content": _CUSTOM_PY,
    },
]

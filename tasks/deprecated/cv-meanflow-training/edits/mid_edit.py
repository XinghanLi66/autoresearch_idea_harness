"""Mid-edit operations for cv-meanflow-training task.

Applied to the alphaflow-main workspace after pre_edit, before the agent starts.
Replaces lines 141-179 in custom_train.py with the custom template,
removing the baseline implementation so the agent must implement it themselves.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_TEMPLATE = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "alphaflow-main/custom_train.py",
        "content": _CUSTOM_TEMPLATE,
    },
]

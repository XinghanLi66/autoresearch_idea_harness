"""Mid-edit operations for the ml-feature-selection task.

Applied to the scikit-learn workspace after pre_edit, before the agent starts.
Creates custom_featsel.py — the agent's editable feature selection file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_featsel.py",
        "content": _CUSTOM_PY,
    },
]

"""Mid-edit operations for the rl-offline-discrete task.

Applied to the d3rlpy workspace after pre_edit, before the agent starts.
Creates custom_atari.py — the agent's editable algorithm file — from custom_template.py.
"""

from pathlib import Path

_DIR = Path(__file__).parent

_CUSTOM_PY = (_DIR / "custom_template.py").read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "d3rlpy/atari_offline/custom_atari.py",
        "content": _CUSTOM_PY,
    },
]

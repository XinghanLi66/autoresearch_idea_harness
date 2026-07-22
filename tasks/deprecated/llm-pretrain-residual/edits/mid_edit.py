"""Mid-edit: creates custom gpt.py from template, replacing the original nanochat gpt.py.

Applied to the nanochat workspace after pre_edit, before the agent starts.
Overwrites nanochat/nanochat/gpt.py with the editable template so the agent
can modify the residual stream design.
"""
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "nanochat/nanochat/gpt.py",
        "content": _CUSTOM_PY,
    },
]

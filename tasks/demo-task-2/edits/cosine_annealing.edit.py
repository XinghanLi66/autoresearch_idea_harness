"""CosineAnnealingLR baseline — rigorous codebase edit ops.

Changes from default StepLR:
  1. Import CosineAnnealingLR instead of StepLR (line 7)
  2. Replace scheduler creation (line 131)

No custom class needed — uses the imported scheduler directly.
Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "pytorch-examples/mnist/main_custom.py"

# ── 1. Replace scheduler creation (line 131) ─────────────────────────────────

_COSINE_SCHEDULER = """\
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
"""

# ── 2. Replace scheduler import (line 7) ─────────────────────────────────────

_COSINE_IMPORT = """\
from torch.optim.lr_scheduler import CosineAnnealingLR
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 131,
        "end_line": 131,
        "content": _COSINE_SCHEDULER,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _COSINE_IMPORT,
    },
]

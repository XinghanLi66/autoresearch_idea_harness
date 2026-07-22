"""Warmup + Cosine Decay baseline — rigorous codebase edit ops.

Custom-implemented scheduler (not imported from torch):
  1. Import math and _LRScheduler base class (line 7)
  2. Define WarmupCosineScheduler class in custom area (lines 9-11)
  3. Use custom scheduler (line 131)

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "pytorch-examples/mnist/main_custom.py"

# ── 1. Replace scheduler creation (line 131) ─────────────────────────────────

_WARMUP_CREATION = """\
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=2, total_epochs=args.epochs)
"""

# ── 2. Replace custom scheduler area (lines 9-11) ────────────────────────────

_WARMUP_CLASS = """\
class WarmupCosineScheduler(_LRScheduler):
    \"\"\"Linear warmup for warmup_epochs, then cosine decay to zero.\"\"\"
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]
"""

# ── 3. Replace scheduler import (line 7) ─────────────────────────────────────

_WARMUP_IMPORT = """\
import math
from torch.optim.lr_scheduler import _LRScheduler
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 131,
        "end_line": 131,
        "content": _WARMUP_CREATION,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 9,
        "end_line": 11,
        "content": _WARMUP_CLASS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _WARMUP_IMPORT,
    },
]

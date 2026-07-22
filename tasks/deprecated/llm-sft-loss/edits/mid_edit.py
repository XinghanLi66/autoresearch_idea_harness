"""Mid-edit operations for the llm-sft-loss task.

Applied to the LLaMA-Factory workspace after pre_edit, before the agent starts.
1. Creates custom_sft_loss.py — the agent's editable loss function.
2. Patches SFT trainer to import and use custom_loss_func.
3. Injects TRAIN_METRICS logging override into the trainer.

Operations are ordered bottom-to-top by line number for stable insertions.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── TRAIN_METRICS log override (inserted after line 148 of trainer.py) ──
# Overrides the log() method to emit TRAIN_METRICS lines to stdout.
_TRAIN_METRICS_OVERRIDE = '''\

    def log(self, logs, **kwargs):
        """Override to print TRAIN_METRICS for agent feedback."""
        super().log(logs, **kwargs)
        if self.state.global_step > 0:
            parts = [f"step={self.state.global_step}"]
            for k in ("loss", "learning_rate", "grad_norm", "epoch"):
                if k in logs:
                    parts.append(f"{k}={logs[k]}")
            print("TRAIN_METRICS: " + ", ".join(parts), flush=True)'''

# ── Import + set compute_loss_func (inserted after line 125 of trainer.py) ──
# This goes right after the existing use_asft_loss block.
_IMPORT_CUSTOM_LOSS = '''\
        else:
            from .custom_sft_loss import custom_loss_func
            self.compute_loss_func = custom_loss_func'''

OPS = [
    # 1. Create the editable custom loss file
    {
        "op": "create",
        "file": "LLaMA-Factory/src/llamafactory/train/sft/custom_sft_loss.py",
        "content": _CUSTOM_PY,
    },
    # 2. Inject TRAIN_METRICS log override (after line 148 — end of _get_train_sampler)
    {
        "op": "insert",
        "file": "LLaMA-Factory/src/llamafactory/train/sft/trainer.py",
        "after_line": 148,
        "content": _TRAIN_METRICS_OVERRIDE,
    },
    # 3. Inject custom loss import (after line 125 — end of use_asft_loss block)
    {
        "op": "insert",
        "file": "LLaMA-Factory/src/llamafactory/train/sft/trainer.py",
        "after_line": 125,
        "content": _IMPORT_CUSTOM_LOSS,
    },
]

"""Mid-edit: creates custom_ttt_eval.py from template.

The task uses real GPT-2 Medium weights (pre-downloaded from HuggingFace),
so no pretraining is needed. The script loads HF weights, converts them
to nanoGPT format, then runs TTT adaptation + evaluation.

- custom_ttt_eval.py: TTT adaptation + evaluation (editable TTTAdapter class).
  Loads GPT-2 Medium weights from /data/gpt2-medium and runs TTT before evaluation.
"""
from pathlib import Path

_TTT_EVAL_TEMPLATE = (Path(__file__).parent / "ttt_eval_template.py").read_text()

OPS = [
    {
        "op": "create",
        "file": "nanoGPT/custom_ttt_eval.py",
        "content": _TTT_EVAL_TEMPLATE,
    },
]

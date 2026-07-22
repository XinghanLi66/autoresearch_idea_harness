"""No averaging baseline (identity / final checkpoint).

The simplest baseline: no weight averaging at all. The final model
checkpoint is used directly for evaluation. This is the default
behavior of standard training.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_NO_AVERAGING = '''\
class WeightAverager:
    """No weight averaging — evaluate the final checkpoint as-is."""
    def __init__(self, model, max_iters):
        pass

    def update(self, model, step):
        pass

    def apply(self, model):
        pass

    def restore(self, model):
        pass
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 205,
        "end_line": 252,
        "content": _NO_AVERAGING,
    },
]

"""EMA with fixed decay baseline.

Exponential Moving Average with a constant decay factor (alpha=0.999).
Updates the averaged weights at every step using:
    avg_param = alpha * avg_param + (1 - alpha) * current_param

This is the standard EMA approach used in many deep learning frameworks
(e.g., PyTorch EMA, timm, fairseq).

Reference: Polyak & Juditsky, "Acceleration of Stochastic Approximation
by Averaging" (SIAM 1992)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_EMA_FIXED = '''\
class WeightAverager:
    """EMA with fixed decay alpha=0.999."""
    def __init__(self, model, max_iters):
        self.alpha = 0.999
        self.avg_state = {k: v.clone() for k, v in model.state_dict().items()}
        self.saved_state = None

    def update(self, model, step):
        current_state = model.state_dict()
        for k in self.avg_state:
            self.avg_state[k].mul_(self.alpha).add_(current_state[k], alpha=1 - self.alpha)

    def apply(self, model):
        self.saved_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.avg_state)

    def restore(self, model):
        if self.saved_state is None:
            return
        model.load_state_dict(self.saved_state)
        self.saved_state = None
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 205,
        "end_line": 252,
        "content": _EMA_FIXED,
    },
]

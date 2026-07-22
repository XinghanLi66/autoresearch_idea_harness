"""SWA (Stochastic Weight Averaging) baseline.

Uniformly averages model checkpoints from the last 20% of training.
This is the classic SWA approach: collect checkpoints at regular
intervals during the final phase and compute their uniform average.

Reference: Izmailov et al., "Averaging Weights Leads to Wider Optima
and Better Generalization" (UAI 2018)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_SWA = '''\
class WeightAverager:
    """SWA: uniform average of checkpoints from the last 20% of training."""
    def __init__(self, model, max_iters):
        self.avg_state = None
        self.n_averaged = 0
        self.start_step = int(max_iters * 0.8)
        self.saved_state = None

    def update(self, model, step):
        if step < self.start_step:
            return
        current_state = {k: v.clone() for k, v in model.state_dict().items()}
        if self.avg_state is None:
            self.avg_state = current_state
            self.n_averaged = 1
        else:
            self.n_averaged += 1
            for k in self.avg_state:
                self.avg_state[k] += (current_state[k] - self.avg_state[k]) / self.n_averaged

    def apply(self, model):
        if self.avg_state is None:
            return
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
        "content": _SWA,
    },
]

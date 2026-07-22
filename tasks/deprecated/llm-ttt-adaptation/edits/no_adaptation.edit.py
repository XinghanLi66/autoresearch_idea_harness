"""No-adaptation baseline (lower bound).

The model is used as-is after pretraining with no test-time adaptation.
This is the baseline that all TTT methods should improve upon.
"""

_FILE = "nanoGPT/custom_ttt_eval.py"

_NO_ADAPT = """\
class TTTAdapter:
    \"\"\"No-adaptation baseline: evaluate the pretrained model directly.\"\"\"

    def __init__(self):
        self.ttt_lr = 0.0
        self.ttt_steps = 0

    def setup(self, model, config):
        self.config = config

    def adapt_and_evaluate(self, model, eval_chunks, ctx):
        model.eval()
        total_loss = 0.0
        n_chunks = 0
        with torch.no_grad():
            for x, y in eval_chunks:
                with ctx:
                    _, loss = model(x, y)
                total_loss += loss.item()
                n_chunks += 1
        return total_loss / max(n_chunks, 1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 276,
        "end_line": 346,
        "content": _NO_ADAPT,
    },
]

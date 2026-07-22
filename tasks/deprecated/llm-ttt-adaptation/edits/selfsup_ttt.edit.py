"""Self-supervised TTT baseline (strong).

Adapts the model's layer norm parameters and biases (if any) using
self-supervised next-token prediction on the evaluation context prefix.
This is lightweight (very few parameters) but effective because layer norms
control the activation statistics that differ between train and eval distributions.

Reference: Sun et al., "Learning to (Learn at Test Time): RNNs with
           Expressive Hidden States" (2024)
           Gandelsman et al., "Test-Time Training with Self-Supervision for
           Generalization under Distribution Shifts" (2022)
"""

_FILE = "nanoGPT/custom_ttt_eval.py"

_SELFSUP_TTT = """\
class TTTAdapter:
    \"\"\"Self-supervised TTT adapter.

    Adapts layer norm parameters and the final LN + lm_head bias using
    self-supervised next-token prediction on the evaluation context.
    This is extremely parameter-efficient while still effective.
    \"\"\"

    def __init__(self):
        self.ttt_lr = 1e-3
        self.ttt_steps = 10
        self.adapt_fraction = 0.2  # Use 20% of eval chunks for adaptation

    def setup(self, model, config):
        self.config = config

    def _get_adaptable_params(self, model):
        \"\"\"Get layer norm weights and biases for adaptation.\"\"\"
        params = []
        for name, param in model.named_parameters():
            # Adapt LayerNorm weights and biases (small but impactful)
            if 'ln_' in name or 'LayerNorm' in name:
                param.requires_grad_(True)
                params.append(param)
            else:
                param.requires_grad_(False)
        return params

    def adapt_and_evaluate(self, model, eval_chunks, ctx):
        # Create a deep copy for adaptation
        adapted = copy.deepcopy(model)
        device = next(model.parameters()).device

        # Split chunks: first fraction for adaptation, all for evaluation
        n_adapt = max(1, int(len(eval_chunks) * self.adapt_fraction))
        adapt_chunks = eval_chunks[:n_adapt]

        # Get adaptable parameters (layer norms only)
        adapt_params = self._get_adaptable_params(adapted)
        n_params = sum(p.numel() for p in adapt_params)

        if adapt_params and self.ttt_steps > 0:
            optimizer = torch.optim.SGD(adapt_params, lr=self.ttt_lr, momentum=0.9)

            # Self-supervised adaptation: next-token prediction on context
            adapted.train()
            for step in range(self.ttt_steps):
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for x, y in adapt_chunks:
                    with ctx:
                        _, loss = adapted(x, y)
                    loss.backward()
                    total_loss += loss.item()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(adapt_params, 1.0)
                optimizer.step()

        # Evaluate on all chunks
        adapted.eval()
        total_loss = 0.0
        n_chunks = 0
        with torch.no_grad():
            for x, y in eval_chunks:
                with ctx:
                    _, loss = adapted(x, y)
                total_loss += loss.item()
                n_chunks += 1

        # Clean up
        del adapted
        torch.cuda.empty_cache()

        return total_loss / max(n_chunks, 1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 276,
        "end_line": 346,
        "content": _SELFSUP_TTT,
    },
]

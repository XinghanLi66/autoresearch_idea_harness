"""LoRA Test-Time Training baseline (medium).

Adds low-rank adaptation (LoRA) matrices to the attention projections,
then fine-tunes only these lightweight parameters on the evaluation context
using next-token prediction loss before final evaluation.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
           Sun et al., "Learning to (Learn at Test Time)" (2024)
"""

_FILE = "nanoGPT/custom_ttt_eval.py"

_LORA_TTT = """\
class LoRALinear(nn.Module):
    \"\"\"Low-rank adaptation wrapper for nn.Linear.\"\"\"
    def __init__(self, base_linear, rank=8, alpha=16.0):
        super().__init__()
        self.base = base_linear
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        self.scaling = alpha / rank
        # Freeze base weights
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


class TTTAdapter:
    \"\"\"LoRA-based test-time training adapter.

    Wraps attention projection layers with low-rank adapters, then fine-tunes
    only the LoRA parameters on eval context before final evaluation.
    \"\"\"

    def __init__(self):
        self.ttt_lr = 3e-4
        self.ttt_steps = 5
        self.lora_rank = 8
        self.lora_alpha = 16.0

    def setup(self, model, config):
        self.config = config

    def _add_lora(self, model):
        \"\"\"Add LoRA adapters to all attention Q/K/V and output projections.\"\"\"
        for block in model.transformer.h:
            attn = block.attn
            attn.c_attn = LoRALinear(attn.c_attn, rank=self.lora_rank, alpha=self.lora_alpha)
            attn.c_proj = LoRALinear(attn.c_proj, rank=self.lora_rank, alpha=self.lora_alpha)
        return model

    def _get_lora_params(self, model):
        \"\"\"Get only the LoRA parameters for optimization.\"\"\"
        params = []
        for block in model.transformer.h:
            attn = block.attn
            if isinstance(attn.c_attn, LoRALinear):
                params.extend([attn.c_attn.lora_A, attn.c_attn.lora_B])
            if isinstance(attn.c_proj, LoRALinear):
                params.extend([attn.c_proj.lora_A, attn.c_proj.lora_B])
        return params

    def adapt_and_evaluate(self, model, eval_chunks, ctx):
        # Create a deep copy and add LoRA layers
        adapted = copy.deepcopy(model)
        adapted = self._add_lora(adapted)
        adapted.to(next(model.parameters()).device)

        # Use first 20% of chunks for adaptation, evaluate on all
        n_adapt = max(1, len(eval_chunks) // 5)
        adapt_chunks = eval_chunks[:n_adapt]

        # Fine-tune LoRA parameters on adaptation context
        lora_params = self._get_lora_params(adapted)
        if lora_params and self.ttt_steps > 0:
            optimizer = torch.optim.AdamW(lora_params, lr=self.ttt_lr, weight_decay=0.0)
            adapted.train()
            for step in range(self.ttt_steps):
                total_step_loss = 0.0
                for x, y in adapt_chunks:
                    with ctx:
                        _, loss = adapted(x, y)
                    loss.backward()
                    total_step_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

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

        # Clean up to free memory
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
        "content": _LORA_TTT,
    },
]

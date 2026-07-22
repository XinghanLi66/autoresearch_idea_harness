"""Explicit BF16 weight casting baseline (basic).

Explicitly casts weights to BF16 before matmul during training,
storing master weights in FP32 for optimizer updates. This is a
simple precision strategy that ensures consistent BF16 computation.

Reference: Micikevicius et al., "Mixed Precision Training" (2018)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_BF16_CAST = """\
class CustomLinear(nn.Module):
    \"\"\"Linear layer with explicit BF16 weight casting during forward.\"\"\"
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Cast weight to BF16 explicitly for consistent precision
        w = self.weight.to(torch.bfloat16)
        b = self.bias.to(torch.bfloat16) if self.bias is not None else None
        return F.linear(x.to(torch.bfloat16), w, b)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 52,
        "content": _BF16_CAST,
    },
]

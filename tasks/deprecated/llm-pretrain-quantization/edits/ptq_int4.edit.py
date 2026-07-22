"""Post-Training Quantization (PTQ) int4 baseline.

No quantization-aware training — the model trains normally with full-precision
weights. At evaluation time, weights are quantized to int4 and dequantized
(symmetric per-channel). This is the simplest baseline: no training overhead
but maximum quantization degradation.

Reference: Jacob et al., "Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_PTQ_INT4 = """\
def fake_quantize_weight(weight, num_bits=4):
    \"\"\"PTQ baseline: no fake quantization during training (pass-through).\"\"\"
    return weight


def fake_quantize_activation(x, num_bits=4):
    \"\"\"PTQ baseline: no fake quantization of activations.\"\"\"
    return x


def quantize_dequantize_weight(weight, num_bits=4):
    \"\"\"Symmetric per-channel int4 quantize-dequantize roundtrip.\"\"\"
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    # Per-channel scale: one scale per output channel (dim 0 of weight)
    w_abs_max = weight.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = w_abs_max / qmax
    w_q = torch.clamp(torch.round(weight / scale), qmin, qmax)
    return w_q * scale


class QATLinear(nn.Module):
    \"\"\"Linear layer — no QAT, only post-training quantization at eval.\"\"\"
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
        self.num_bits = 4

    def forward(self, x):
        if self.training:
            # No fake quantization — standard forward
            w = self.weight
        else:
            w = quantize_dequantize_weight(self.weight, self.num_bits)
        return F.linear(x, w, self.bias)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 34,
        "end_line": 118,
        "content": _PTQ_INT4,
    },
]

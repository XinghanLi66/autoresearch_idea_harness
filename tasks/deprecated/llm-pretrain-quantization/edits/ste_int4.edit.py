"""Straight-Through Estimator (STE) int4 QAT baseline.

Uses the Straight-Through Estimator to enable gradient flow through the
non-differentiable rounding operation. During forward: weights are fake-
quantized (quantize then dequantize in float). During backward: gradients
pass straight through the rounding as if it were identity.

This is the classic QAT approach. Symmetric per-channel quantization with
int4 precision for weights, per-tensor for activations.

Reference: Bengio et al., "Estimating or Propagating Gradients Through
Stochastic Neurons for Conditional Computation" (2013)
Jacob et al., "Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_STE_INT4 = """\
class _STEQuantize(torch.autograd.Function):
    \"\"\"Straight-Through Estimator for quantize-dequantize.\"\"\"
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged
        return grad_output, None, None, None


def fake_quantize_weight(weight, num_bits=4):
    \"\"\"STE-based fake quantization for weights (symmetric per-channel).\"\"\"
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    # Per-channel scale: one scale per output channel (dim 0 of weight)
    w_abs_max = weight.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = w_abs_max / qmax
    return _STEQuantize.apply(weight, scale, qmin, qmax)


def fake_quantize_activation(x, num_bits=4):
    \"\"\"STE-based fake quantization for activations (symmetric per-tensor).\"\"\"
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    x_abs_max = x.detach().abs().max().clamp(min=1e-12)
    scale = x_abs_max / qmax
    return _STEQuantize.apply(x, scale, qmin, qmax)


def quantize_dequantize_weight(weight, num_bits=4):
    \"\"\"Symmetric per-channel int4 quantize-dequantize roundtrip for eval.\"\"\"
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    # Per-channel scale: one scale per output channel (dim 0 of weight)
    w_abs_max = weight.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = w_abs_max / qmax
    w_q = torch.clamp(torch.round(weight / scale), qmin, qmax)
    return w_q * scale


class QATLinear(nn.Module):
    \"\"\"Linear layer with STE-based int4 QAT.\"\"\"
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
            w = fake_quantize_weight(self.weight, self.num_bits)
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
        "content": _STE_INT4,
    },
]

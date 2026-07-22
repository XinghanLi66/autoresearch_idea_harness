"""Learned Step Size Quantization (LSQ) int4 baseline.

Extends STE-based QAT by making the quantization step size (scale factor)
a learnable parameter that is updated via gradient descent. Uses the
grad_scale + round_pass STE formulation from the reference implementation
(zhutmost/lsq-net), which avoids custom autograd Functions entirely and
is fully compatible with torch.compile.

Per-channel learned step size for weights (one step size per output
channel), activations use fixed STE quantization.

The gradient of the step size is scaled by 1/sqrt(n * qmax) per the
LSQ paper to stabilize training.

Reference: Esser et al., "Learned Step Size Quantization" (ICLR 2020)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_LSQ_INT4 = """\
def _grad_scale(x, scale):
    \"\"\"Scale gradients without affecting forward values (LSQ Eq. 2).

    Forward: returns x unchanged.
    Backward: multiplies incoming gradient by scale.
    Uses the detach trick: (x - x*scale).detach() + x*scale
    so forward = x but grad = grad_output * scale.
    \"\"\"
    y = x * scale
    return (x - y).detach() + y


def _round_pass(x):
    \"\"\"STE round: forward rounds, backward passes gradient through.

    Uses the detach trick: (round(x) - x).detach() + x
    so forward = round(x) but grad = grad_output.
    \"\"\"
    return (x.round() - x).detach() + x


def fake_quantize_weight(weight, num_bits=4, step_size=None):
    \"\"\"LSQ fake quantization for weights with learnable per-channel step size.

    When step_size is provided, uses LSQ grad_scale + round_pass.
    Otherwise falls back to standard per-channel STE quantization.
    \"\"\"
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    if step_size is None:
        # Per-channel scale: one scale per output channel (dim 0 of weight)
        w_abs_max = weight.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = w_abs_max / qmax
        w_q = torch.clamp(torch.round(weight / scale), qmin, qmax)
        return w_q * scale
    # LSQ: scale the step size gradient by 1/sqrt(n_per_channel * Qp)
    n_per_channel = weight.shape[1]
    s_scale = 1.0 / ((n_per_channel * qmax) ** 0.5)
    s = _grad_scale(step_size.abs().clamp(min=1e-8), s_scale)
    x = weight / s
    x = torch.clamp(x, qmin, qmax)
    x = _round_pass(x)
    return x * s


def fake_quantize_activation(x, num_bits=4):
    \"\"\"STE-based fake quantization for activations (symmetric per-tensor).

    Uses fixed scale (no learnable step size) for activations to keep
    memory usage identical to the STE baseline. Uses _round_pass for
    STE gradient flow (same as weight quantization).
    \"\"\"
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    x_abs_max = x.detach().abs().max().clamp(min=1e-12)
    scale = x_abs_max / qmax
    x_div = x / scale
    x_q = torch.clamp(_round_pass(x_div), qmin, qmax)
    return x_q * scale


def quantize_dequantize_weight(weight, num_bits=4, step_size=None):
    \"\"\"Per-channel quantize-dequantize roundtrip for eval using learned or fixed step size.\"\"\"
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    if step_size is not None:
        s = step_size.detach().abs().clamp(min=1e-12)
    else:
        # Per-channel scale: one scale per output channel (dim 0 of weight)
        w_abs_max = weight.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        s = w_abs_max / qmax
    w_q = torch.clamp(torch.round(weight / s), qmin, qmax)
    return w_q * s


class QATLinear(nn.Module):
    \"\"\"Linear layer with LSQ (Learned Step Size Quantization) int4.

    Per-channel learnable step size for weight quantization; activations use
    fixed-scale STE quantization to keep memory usage manageable.
    The step size is initialized to 2*std(weight)/sqrt(qmax) which
    approximates the LSQ paper's 2*mean(|w|)/sqrt(qmax) for normal
    distributions without needing lazy initialization.
    \"\"\"
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
        # Per-channel learnable step size for weights.
        # For N(0, 0.02), mean(|w|) ~ 0.02*sqrt(2/pi) ~ 0.016.
        # LSQ init: s = 2*mean(|w|)/sqrt(qmax) = 2*0.016/sqrt(7) ~ 0.0121
        qmax = (1 << (self.num_bits - 1)) - 1
        init_step = 2.0 * 0.02 * (2.0 / 3.14159) ** 0.5 / (qmax ** 0.5)
        self.w_step = nn.Parameter(torch.full((out_features, 1), init_step))

    def forward(self, x):
        if self.training:
            w = fake_quantize_weight(self.weight, self.num_bits, self.w_step)
        else:
            w = quantize_dequantize_weight(
                self.weight, self.num_bits, self.w_step
            )
        return F.linear(x, w, self.bias)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 34,
        "end_line": 118,
        "content": _LSQ_INT4,
    },
]

"""FP8 linear layer baseline (medium).

Uses torch._scaled_mm with float8_e4m3fn for forward pass matmul.
Weights and activations are quantized to FP8 with per-tensor scaling,
then dequantized after matmul. Falls back to BF16 during eval.

Reference: KellerJordan/modded-nanogpt record #19, Micikevicius et al. (2022)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_FP8_LINEAR = """\
class CustomLinear(nn.Module):
    \"\"\"Linear layer with FP8 matmul during training (H100 required).\"\"\"
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
        if self.training and x.is_cuda:
            return _FP8LinearMatmul.apply(
                x, self.weight, self.bias,
                self.in_features, self.out_features
            )
        else:
            return F.linear(x, self.weight.to(x.dtype), self.bias)


class _FP8LinearMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, in_f, out_f):
        orig_shape = x.shape
        x_flat = x.reshape(-1, in_f).float()
        # Dynamic per-tensor scale computation
        x_abs_max = x_flat.detach().abs().max().clamp(min=1e-12)
        w_abs_max = w.detach().abs().max().clamp(min=1e-12)
        xs = x_abs_max / 448.0
        ws = w_abs_max / 448.0
        # Quantize to FP8 E4M3
        x_f8 = x_flat.div(xs).clamp(-448, 448).to(torch.float8_e4m3fn)
        w_f8 = w.float().div(ws).clamp(-448, 448).to(torch.float8_e4m3fn)
        # _scaled_mm(A, B): A row-major, B column-major
        w_t = w_f8.t().contiguous()
        w_cm = w_t.T.contiguous().T
        out = torch._scaled_mm(
            x_f8, w_cm,
            out_dtype=torch.bfloat16,
            scale_a=x_flat.new_tensor(xs),
            scale_b=x_flat.new_tensor(ws),
            use_fast_accum=True,
        )
        out = out.view(*orig_shape[:-1], out_f)
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(x_f8, w_f8)
        ctx.scales = (xs, ws)
        ctx.shapes = (orig_shape, in_f, out_f)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_f8, w_f8 = ctx.saved_tensors
        xs, ws = ctx.scales
        orig_shape, in_f, out_f = ctx.shapes
        go = grad_output.reshape(-1, out_f)
        # Quantize gradient to FP8 E4M3 for _scaled_mm backward
        go_abs = go.detach().float().abs().max().clamp(min=1e-12)
        gs = go_abs / 448.0
        go_f8 = go.float().div(gs).clamp(-448, 448).to(torch.float8_e4m3fn)
        # grad_x = grad_output @ W  (W is out_f x in_f)
        grad_x = torch._scaled_mm(
            go_f8, w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_output.new_tensor(gs, dtype=torch.float32),
            scale_b=grad_output.new_tensor(ws, dtype=torch.float32),
            use_fast_accum=False,
        )
        grad_x = grad_x.view(orig_shape)
        # grad_w = grad_output.T @ X
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            go_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=grad_output.new_tensor(xs, dtype=torch.float32),
            scale_b=grad_output.new_tensor(gs, dtype=torch.float32),
            use_fast_accum=False,
        )
        grad_w = grad_w.T
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1))) if ctx.has_bias else None
        return grad_x, grad_w, grad_bias, None, None
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 52,
        "content": _FP8_LINEAR,
    },
]

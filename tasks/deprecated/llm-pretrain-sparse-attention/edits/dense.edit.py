"""Dense causal attention (reference upper bound).

This baseline is the no-sparsity oracle — it deliberately bypasses the
density-budget check by setting ``self.is_dense_oracle = True``. Use only as a
reference; agents are NOT permitted to set this flag.

Reference: standard multi-head causal attention (Vaswani et al., 2017).
"""

_FILE = "nanoGPT/custom_pretrain.py"

_DENSE_ATTENTION = """\
class SparseSelfAttention(nn.Module):
    \"\"\"Dense causal attention reference (oracle, density=1.0).\"\"\"
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        self.reported_density = 1.0
        self.is_dense_oracle = True   # reference upper bound, bypass budget
        self.use_pos_emb = True

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 34,
        "end_line": 72,
        "content": _DENSE_ATTENTION,
    },
]

"""Vanilla attention-only control.

Removes the TTT memory residual by returning zeros from TitansMemoryLayer.
This is a no-memory lower bound for ttt-memory and the pure-attention control
for ttt-memory-2048.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_REGION = """\
def init_memory_state(memory, dim):
    memory.memory_dim = dim
    memory.depth = 0
    memory.chunk_size = 64


def compute_keys_values(memory, x):
    return x, x


def compute_surprise(memory, fast_W, keys, values):
    return keys.new_zeros(())


def apply_update(memory, fast_W, fast_S, grads, gates=None):
    return fast_W, fast_S


class TitansMemoryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd
        init_memory_state(self, self.dim)

    def forward(self, x):
        return torch.zeros_like(x)

class CausalSelfAttention(nn.Module):
    '''Standard causal softmax attention (RoPE) with no memory residual.'''
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.memory = TitansMemoryLayer(config)
        self.use_pos_emb = False
        self._rope_base = 10000.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def _rope_cache(self, T, device, dtype):
        hd = self.head_dim
        inv = 1.0 / (self._rope_base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
        pos = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype).view(1, 1, T, hd), emb.sin().to(dtype).view(1, 1, T, hd)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x, cos, sin):
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = self._rope_cache(T, x.device, q.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        if self.flash:
            a = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.head_dim)
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            att = att.masked_fill(~mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            a = torch.einsum('bhqk,bhkd->bhqd', att, v)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(a))

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 35,
        "end_line": 274,
        "content": _REGION,
    },
]

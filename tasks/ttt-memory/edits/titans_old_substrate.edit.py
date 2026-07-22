"""Titans OLD substrate — pre-faithful-rewrite reference.

Mirrors the substrate used in commits prior to 3c11c1878 (April 2026):
- chunk_size = 64 (16 chunks per 1024 seq, 32 chunks per 2048 seq)
- full dim x dim fast weights (no multi-head split)
- shared-across-batch fast weights (single tensor with requires_grad)
- autograd.grad inner gradient
- write-then-read order
- fixed scalar SGD+momentum (lr=1.0, mu=0.9, decay=0.99)
- no q/k normalization
- compute_surprise uses F.mse_loss reduction='mean' (-> implicit grad scale 1/(BTC))

This is the implementation that produced niah_2k = 0.975 / 0.05 on the OLD
345M run before the "faithful rewrite" stack of changes broke training.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_REGION = """\
def init_memory_state(memory, dim):
    memory.memory_dim = dim
    memory.depth = 2
    memory.chunk_size = 64
    memory.learning_rate = 1.0
    memory.momentum_coef = 0.9
    memory.decay = 0.99
    std = 1.0 / math.sqrt(dim)
    W1 = torch.empty(dim, dim); torch.nn.init.normal_(W1, mean=0.0, std=std)
    W2 = torch.empty(dim, dim); torch.nn.init.normal_(W2, mean=0.0, std=std)
    memory.register_buffer('M1_init', W1)
    memory.register_buffer('M2_init', W2)


def compute_keys_values(memory, x):
    return memory.k_proj(x), memory.v_proj(x)


def compute_surprise(memory, fast_W, keys, values):
    pred = memory._apply_fast_mlp(fast_W, keys)
    return F.mse_loss(pred, values, reduction='mean')


def apply_update(memory, fast_W, fast_S, grads):
    lr, mu, alpha = memory.learning_rate, memory.momentum_coef, memory.decay
    new_W, new_S = [], []
    for W, S, g in zip(fast_W, fast_S, grads):
        S_new = mu * S - lr * g
        W_new = alpha * W + S_new
        new_W.append(W_new); new_S.append(S_new)
    return new_W, new_S


class TitansMemoryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        init_memory_state(self, self.dim)

    def _apply_fast_mlp(self, fast_W, x):
        h = x @ fast_W[0].transpose(-1, -2)
        if len(fast_W) >= 2:
            h = F.silu(h)
            h = h @ fast_W[1].transpose(-1, -2)
        return h

    def _init_fast(self):
        W = [self.M1_init.detach().clone()]
        if self.depth >= 2:
            W.append(self.M2_init.detach().clone())
        for w in W:
            w.requires_grad_(True)
        S = [torch.zeros_like(w) for w in W]
        return W, S

    @torch.compiler.disable
    def forward(self, x):
        B, T, C = x.shape
        fast_W, fast_S = self._init_fast()
        outs = []
        cs = self.chunk_size
        for s in range(0, T, cs):
            e = min(s + cs, T)
            chunk = x[:, s:e]
            with torch.enable_grad():
                keys, values = compute_keys_values(self, chunk.detach())
                surprise = compute_surprise(self, fast_W, keys, values)
                grads = torch.autograd.grad(
                    surprise, fast_W, create_graph=False, retain_graph=False)
            fast_W, fast_S = apply_update(self, fast_W, fast_S, grads)
            fast_W = [w.detach().requires_grad_(True) for w in fast_W]
            fast_S = [s_.detach() for s_ in fast_S]
            q = self.q_proj(chunk)
            outs.append(self._apply_fast_mlp(fast_W, q))
        y = torch.cat(outs, dim=1)
        return self.out_proj(y)


class CausalSelfAttention(nn.Module):
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
        a = self.resid_dropout(self.c_proj(a))
        m = self.memory(x)
        return a + m
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

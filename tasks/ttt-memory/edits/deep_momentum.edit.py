"""Deep Momentum Gradient Descent — Behrouz et al. 2025 Nested Learning
(arXiv:2512.24695) §4 Equation 51.

DMGD recurrence (Eq. 51, paper line 465 of Methods.tex):

    W_{i+1} = W_i + m_{i+1}
    m_{i+1} = α_{i+1} m_i  -  η_t · φ(∇L(W_i; x_i))

Key differences from Titans Eq. 13-14 (titans_default):
  (1) NO weight-decay / forget-gate on W: the memory just accumulates
      momentum (`W += m`), only `m` decays via α_{i+1}. This is faithful
      to Eq. 51 — the (1 − α_t) gate on W in Titans is dropped here.
  (2) A learnable feature map φ is applied to the gradient before it is
      added to the momentum. The paper describes φ as a "higher-order
      feature mapping (that may be learned through its internal
      objective)"; here we use an identity skip plus a small row-wise MLP
      residual whose weights are co-trained with the slow-projection params.
  (3) Only two data-dependent gates (α_{i+1}, η_t), parallel to the
      Titans setup. The projection P_i in Eq. 51 is taken as the identity
      (a defensible simplification — its only function is matching dim
      between φ-space and weight space, which our shape-preserving φ
      handles directly).

Memory architecture is depth=2 MLP, same as titans_default, so this is
a pure update-rule ablation against titans_default.

Eq. 51 specifies the optimizer recurrence but not a head decomposition. We use
the same 16 x 64 multi-head fast-weight factorization as Titans/TTT so the DMGD
state scales with head_dim^2 instead of width^2.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_REGION = """\
def init_memory_state(memory, dim):
    memory.memory_dim = dim
    memory.depth = 2
    memory.chunk_size = 256
    memory.head_dim = 64
    assert dim % memory.head_dim == 0
    memory.num_memory_heads = dim // memory.head_dim
    assert dim == memory.num_memory_heads * memory.head_dim
    std = 1.0 / math.sqrt(memory.head_dim)
    W1 = torch.empty(memory.num_memory_heads, memory.head_dim, memory.head_dim); torch.nn.init.normal_(W1, mean=0.0, std=std)
    W2 = torch.empty(memory.num_memory_heads, memory.head_dim, memory.head_dim); torch.nn.init.normal_(W2, mean=0.0, std=std)
    memory.register_buffer('M1_init', W1)
    memory.register_buffer('M2_init', W2)


def _linear(layer, x):
    return F.linear(x, layer.weight, layer.bias)


def _silu_grad(x):
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def compute_keys_values(memory, x):
    B, T, _ = x.shape
    H, D = memory.num_memory_heads, memory.head_dim
    k = F.normalize(F.silu(_linear(memory.k_proj, x).view(B, T, H, D)), dim=-1)
    v = F.silu(_linear(memory.v_proj, x).view(B, T, H, D))
    return k, v


def compute_surprise(memory, fast_W, keys, values):
    pred = memory._apply_fast_mlp(fast_W, keys)
    return (pred - values).pow(2).sum(dim=-1).mean(dim=1).sum()


def compute_fast_gradients(memory, fast_W, keys, values):
    scale = 2.0 / max(keys.shape[1], 1)
    h1_pre = torch.einsum('bthi,bhoi->btho', keys, fast_W[0])
    h1 = F.silu(h1_pre)
    pred = torch.einsum('bthi,bhoi->btho', h1, fast_W[1])
    err = pred - values
    grad_W2 = scale * torch.einsum('btho,bthi->bhoi', err, h1)
    grad_h1 = scale * torch.einsum('btho,bhoi->bthi', err, fast_W[1])
    grad_pre = grad_h1 * _silu_grad(h1_pre)
    grad_W1 = torch.einsum('btho,bthi->bhoi', grad_pre, keys)
    return [grad_W1, grad_W2]


def apply_update(memory, fast_W, fast_S, grads, gates):
    # NL DMGD / higher-order feature map update, factorized per head.
    alpha_t, eta_t = gates
    phi_a, phi_b = memory.phi_a, memory.phi_b
    new_W, new_S = [], []
    for W, S, g in zip(fast_W, fast_S, grads):
        # phi is row-wise within each head: (B, H, D, D) -> (B, H, D, D).
        h = F.linear(g, phi_a.weight, phi_a.bias)
        h = F.silu(h)
        phi_g = g + F.linear(h, phi_b.weight, phi_b.bias)
        S_new = alpha_t.unsqueeze(-1) * S - eta_t.unsqueeze(-1) * phi_g
        W_new = W + S_new                            # NO (1-alpha) decay on W
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
        self.alpha_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.eta_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.use_checkpoint = False
        init_memory_state(self, self.dim)
        # Learned higher-order feature map phi, applied per head row. Keeping it
        # at head_dim -> 2*head_dim -> head_dim matches the fast-weight factorization.
        self.phi_a = nn.Linear(self.head_dim, 2 * self.head_dim, bias=True)
        self.phi_b = nn.Linear(2 * self.head_dim, self.head_dim, bias=True)
        for p in (self.alpha_proj, self.eta_proj):
            nn.init.zeros_(p.weight)
        nn.init.constant_(self.alpha_proj.bias, 2.2)
        nn.init.constant_(self.eta_proj.bias,  -2.2)
        nn.init.normal_(self.phi_a.weight, std=0.01)
        nn.init.zeros_(self.phi_a.bias)
        nn.init.zeros_(self.phi_b.weight); nn.init.zeros_(self.phi_b.bias)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_memory_heads, self.head_dim)

    def _merge_heads(self, x):
        B, T, H, D = x.shape
        return x.contiguous().view(B, T, H * D)

    def _gate_heads(self, x):
        return x.view(x.shape[0], self.num_memory_heads, self.head_dim)

    def _apply_fast_mlp(self, fast_W, x):
        h = torch.einsum('bthi,bhoi->btho', x, fast_W[0])
        if len(fast_W) >= 2:
            h = F.silu(h)
            h = torch.einsum('bthi,bhoi->btho', h, fast_W[1])
        return h

    def _init_fast(self, batch_size, dtype):
        W = [self.M1_init.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()]
        if self.depth >= 2:
            W.append(self.M2_init.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous())
        if self.training:
            for w in W:
                w.requires_grad_(True)
        S = [torch.zeros_like(w) for w in W]
        return W, S

    def _compute_gates(self, chunk):
        chunk_mean = chunk.mean(dim=1)
        alpha_t = torch.sigmoid(_linear(self.alpha_proj, chunk_mean))
        eta_t = torch.sigmoid(_linear(self.eta_proj, chunk_mean))
        return self._gate_heads(alpha_t), self._gate_heads(eta_t)

    def _chunk_step(self, chunk, *state):
        depth = self.depth
        fast_W = list(state[:depth])
        fast_S = list(state[depth:])
        chunk_f = chunk

        gates = self._compute_gates(chunk_f)
        q = F.normalize(F.silu(self._split_heads(_linear(self.q_proj, chunk_f))), dim=-1)
        out = self._apply_fast_mlp(fast_W, q)
        keys, values = compute_keys_values(self, chunk_f)
        grads = compute_fast_gradients(self, fast_W, keys, values)
        fast_W, fast_S = apply_update(self, fast_W, fast_S, grads, gates)
        if not self.training:
            fast_W = [w.detach() for w in fast_W]
            fast_S = [s_.detach() for s_ in fast_S]
        return (out.to(chunk.dtype), *fast_W, *fast_S)

    @torch.compiler.disable
    def forward(self, x):
        B, T, C = x.shape
        fast_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        fast_W, fast_S = self._init_fast(B, fast_dtype)
        outs = []
        cs = self.chunk_size
        if self.training and self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
        for s in range(0, T, cs):
            e = min(s + cs, T)
            chunk = x[:, s:e]
            state = tuple(fast_W + fast_S)
            if self.training and self.use_checkpoint:
                result = checkpoint(self._chunk_step, chunk, *state, use_reentrant=False)
            else:
                result = self._chunk_step(chunk, *state)
            outs.append(result[0])
            state = result[1:]
            fast_W = list(state[:self.depth])
            fast_S = list(state[self.depth:])
        y = self._merge_heads(torch.cat(outs, dim=1))
        return self.out_proj(y)

class CausalSelfAttention(nn.Module):
    '''Standard causal softmax attention (RoPE) + parallel Titans memory residual.'''
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

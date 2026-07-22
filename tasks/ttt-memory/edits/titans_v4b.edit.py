"""Titans v4b - same as v4 but TBPTT every 8 chunks.

The 345M baseline trained at exactly 4 chunks/sample without any
stabilization. v4 with tbptt_chunks=4 reduces back to that working
condition, so it doesn't actually exercise the stabilizers. v4b sets
tbptt_chunks=8 — past the empirical NaN threshold — to test whether
the RMS caps + bounded gates + fp32 fast state genuinely enable
co-training at chunks/sample > 4.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_REGION = """\
def init_memory_state(memory, dim):
    memory.memory_dim = dim
    memory.depth = 2
    memory.chunk_size = 64
    memory.head_dim = 64
    assert dim % memory.head_dim == 0
    memory.num_memory_heads = dim // memory.head_dim
    assert dim == memory.num_memory_heads * memory.head_dim

    memory.norm_eps = 1e-6
    memory.tbptt_chunks = 8
    memory.alpha_min = 0.02
    memory.alpha_span = 0.18
    memory.eta_max = 0.25
    memory.theta_max = 0.02
    memory.grad_rms_cap = 1.0
    memory.fast_weight_rms_cap = 2.0 / math.sqrt(memory.head_dim)
    memory.fast_state_rms_cap = 0.5 / math.sqrt(memory.head_dim)

    std = 1.0 / math.sqrt(memory.head_dim)
    W1 = torch.empty(memory.num_memory_heads, memory.head_dim, memory.head_dim)
    W2 = torch.empty(memory.num_memory_heads, memory.head_dim, memory.head_dim)
    torch.nn.init.normal_(W1, mean=0.0, std=std)
    torch.nn.init.normal_(W2, mean=0.0, std=std)
    memory.register_buffer('M1_init', W1)
    memory.register_buffer('M2_init', W2)


def _linear(layer, x):
    return F.linear(x, layer.weight, layer.bias)


def _rms(x, dims):
    return x.float().pow(2).mean(dim=dims, keepdim=True).add(1e-8).sqrt()


def _clip_rms(x, max_rms, dims=(-1, -2)):
    rms = _rms(x, dims)
    scale = torch.clamp(max_rms / rms, max=1.0).to(dtype=x.dtype).detach()
    return x * scale


def _silu_grad(x):
    x_safe = x.float().clamp(min=-20.0, max=20.0)
    sig = torch.sigmoid(x_safe)
    return sig * (1.0 + x_safe * (1.0 - sig))


def compute_keys_values(memory, x, dtype=None):
    B, T, _ = x.shape
    H, D = memory.num_memory_heads, memory.head_dim
    k = _linear(memory.k_proj, x).view(B, T, H, D).float()
    v = _linear(memory.v_proj, x).view(B, T, H, D).float()
    k = F.normalize(F.silu(k), dim=-1, eps=memory.norm_eps)
    v = F.silu(v)
    if dtype is not None:
        k = k.to(dtype=dtype)
        v = v.to(dtype=dtype)
    return k, v


def compute_surprise(memory, fast_W, keys, values):
    pred = memory._apply_fast_mlp(fast_W, keys)
    return (pred - values).pow(2).sum(dim=-1).mean(dim=1).sum()


def compute_fast_gradients(memory, fast_W, keys, values):
    scale = 2.0 / max(keys.shape[1], 1)
    h1_pre = torch.einsum('bthi,bhoi->btho', keys, fast_W[0])
    if len(fast_W) == 1:
        err = h1_pre - values
        grad = scale * torch.einsum('btho,bthi->bhoi', err, keys)
        return [_clip_rms(grad, memory.grad_rms_cap)]

    h1 = F.silu(h1_pre.float()).to(dtype=fast_W[0].dtype)
    pred = torch.einsum('bthi,bhoi->btho', h1, fast_W[1])
    err = pred - values
    grad_W2 = scale * torch.einsum('btho,bthi->bhoi', err, h1)
    grad_h1 = scale * torch.einsum('btho,bhoi->bthi', err, fast_W[1])
    grad_pre = grad_h1 * _silu_grad(h1_pre).to(dtype=grad_h1.dtype)
    grad_W1 = torch.einsum('btho,bthi->bhoi', grad_pre, keys)
    return [_clip_rms(grad_W1, memory.grad_rms_cap),
            _clip_rms(grad_W2, memory.grad_rms_cap)]


def apply_update(memory, fast_W, fast_S, grads, gates):
    alpha_t, eta_t, theta_t = gates
    decay = (1.0 - alpha_t).unsqueeze(-1)
    eta = eta_t.unsqueeze(-1)
    theta = theta_t.unsqueeze(-1)
    new_W, new_S = [], []
    for W, S, g in zip(fast_W, fast_S, grads):
        update = _clip_rms(theta * g, memory.fast_state_rms_cap)
        S_new = _clip_rms(eta * S - update, memory.fast_state_rms_cap)
        W_new = _clip_rms(decay * W + S_new, memory.fast_weight_rms_cap)
        new_W.append(W_new)
        new_S.append(S_new)
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
        self.theta_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        for p in (self.alpha_proj, self.eta_proj, self.theta_proj):
            nn.init.zeros_(p.weight)
        nn.init.constant_(self.alpha_proj.bias, -4.6)
        nn.init.constant_(self.eta_proj.bias, -2.0)
        nn.init.constant_(self.theta_proj.bias, -2.0)
        self.use_checkpoint = False
        init_memory_state(self, self.dim)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_memory_heads, self.head_dim)

    def _merge_heads(self, x):
        B, T, H, D = x.shape
        return x.contiguous().view(B, T, H * D)

    def _gate_heads(self, x):
        return x.view(x.shape[0], self.num_memory_heads, self.head_dim)

    def _apply_fast_mlp(self, fast_W, x):
        x = x.to(dtype=fast_W[0].dtype)
        h = torch.einsum('bthi,bhoi->btho', x, fast_W[0])
        if len(fast_W) >= 2:
            h = F.silu(h.float()).to(dtype=fast_W[0].dtype)
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
        chunk_mean = chunk.mean(dim=1).float()
        alpha_raw = torch.sigmoid(_linear(self.alpha_proj, chunk_mean).float())
        eta_raw = torch.sigmoid(_linear(self.eta_proj, chunk_mean).float())
        theta_raw = torch.sigmoid(_linear(self.theta_proj, chunk_mean).float())
        alpha_t = self.alpha_min + self.alpha_span * alpha_raw
        eta_t = self.eta_max * eta_raw
        theta_t = self.theta_max * theta_raw
        return tuple(self._gate_heads(g) for g in (alpha_t, eta_t, theta_t))

    def _chunk_step(self, chunk, *state):
        depth = self.depth
        fast_W = list(state[:depth])
        fast_S = list(state[depth:])
        chunk_f = chunk.float()

        gates = self._compute_gates(chunk_f)
        q = _linear(self.q_proj, chunk_f).view(
            chunk.shape[0], chunk.shape[1], self.num_memory_heads, self.head_dim).float()
        q = F.normalize(F.silu(q), dim=-1, eps=self.norm_eps).to(dtype=fast_W[0].dtype)
        out = self._apply_fast_mlp(fast_W, q)
        keys, values = compute_keys_values(self, chunk_f, dtype=fast_W[0].dtype)
        grads = compute_fast_gradients(self, fast_W, keys, values)
        fast_W, fast_S = apply_update(self, fast_W, fast_S, grads, gates)
        if not self.training:
            fast_W = [w.detach() for w in fast_W]
            fast_S = [s_.detach() for s_ in fast_S]
        return (out.to(chunk.dtype), *fast_W, *fast_S)

    @torch.compiler.disable
    def forward(self, x):
        B, T, C = x.shape
        fast_W, fast_S = self._init_fast(B, torch.float32)
        outs = []
        cs = self.chunk_size
        for chunk_idx, s in enumerate(range(0, T, cs)):
            e = min(s + cs, T)
            chunk = x[:, s:e]
            state = tuple(fast_W + fast_S)
            result = self._chunk_step(chunk, *state)
            outs.append(result[0])
            state = result[1:]
            fast_W = list(state[:self.depth])
            fast_S = list(state[self.depth:])
            if self.training and self.tbptt_chunks > 0 and (chunk_idx + 1) % self.tbptt_chunks == 0 and e < T:
                fast_W = [w.detach().requires_grad_(True) for w in fast_W]
                fast_S = [s_.detach() for s_ in fast_S]
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

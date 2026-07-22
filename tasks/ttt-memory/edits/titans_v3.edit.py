"""Titans v3 — corrected substrate (multi-head + per-batch + closed-form
+ chunk_size=64 + tamed init gates).

Same paper-aligned structural choices as titans_default (multi-head fast
weights, per-batch independent fast W, closed-form inner gradient,
data-dependent gates), but with the following corrections vs the broken
titans_default:

  1. chunk_size: 256 -> 64.
     Restores 4x more memory updates per training sample. With train_ctx=2048,
     this is 32 chunks/sample (matches OLD substrate at the same length).

  2. theta_proj.bias: 0.0 -> -6.0  (sigmoid 0.5 -> 0.0025)
     OLD's effective inner step magnitude was ~2/D (with D=full embed dim
     after autograd-of-mean normalization). NEW closed-form scale 2/T (with
     no division by output dim) makes per-element writes ~D/T larger than
     OLD when theta is matched. To match OLD's ~1e-3 per-step magnitude:
     theta * (2/T) ~ 1e-3 -> theta ~ 1e-3 * T/2 = 1e-3 * 32 = 0.03; choose
     a slightly more conservative 0.0025 (bias=-6) since chunk count is 4x
     OLD's at train_ctx=2048.

  3. eta_proj.bias: 2.2 -> -2.0  (sigmoid 0.9 -> 0.12)
     With eta=0.9 the momentum compounds 1+0.9+0.81+... over a chunk
     sequence; combined with the too-aggressive theta this saturates
     fast_W. eta=0.12 keeps the inner update mostly direct-grad.

  4. alpha_proj.bias = -4.6 (sigmoid -> 0.01) is unchanged — slow forget is
     paper-aligned and OLD also used decay=0.99 (i.e. 1 - alpha = 0.99).

  5. Fast weights kept in FP32 inside the inner loop (regardless of
     autocast) — bf16 accumulation over 32 chunks compounds rounding noise
     into NaN. Outputs cast back to bf16 for outer computation.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_REGION = """\
def init_memory_state(memory, dim):
    memory.memory_dim = dim
    memory.depth = 2
    memory.chunk_size = 64                # was 256 (broken); now 64 -> 32 chunks at ctx=2048
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
    k = _linear(memory.k_proj, x).view(B, T, H, D)
    v = _linear(memory.v_proj, x).view(B, T, H, D)
    k = F.normalize(F.silu(k), dim=-1)
    v = F.silu(v)
    return k, v


def compute_surprise(memory, fast_W, keys, values):
    pred = memory._apply_fast_mlp(fast_W, keys)
    return (pred - values).pow(2).sum(dim=-1).mean(dim=1).sum()


def compute_fast_gradients(memory, fast_W, keys, values):
    # autograd.grad with create_graph=False to MATCH the OLD substrate's
    # gradient policy: slow params (k_proj, v_proj, alpha/eta/theta_proj)
    # do NOT receive gradients flowing through the recurrent inner-update
    # chain. They still get gradients through the parallel `out` path.
    # create_graph=True caused NaN — the second-order graph through 32
    # chunks x 12 layers is too deep to be stable.
    with torch.enable_grad():
        leaf_W = [w.detach().requires_grad_(True) for w in fast_W]
        pred = memory._apply_fast_mlp(leaf_W, keys)
        loss = (pred - values).pow(2).sum(dim=-1).mean(dim=1).sum()
        grads = torch.autograd.grad(loss, leaf_W,
                                    create_graph=False, retain_graph=False)
    return list(grads)


def apply_update(memory, fast_W, fast_S, grads, gates):
    alpha_t, eta_t, theta_t = gates
    new_W, new_S = [], []
    for W, S, g in zip(fast_W, fast_S, grads):
        S_new = eta_t.unsqueeze(-1) * S - theta_t.unsqueeze(-1) * g
        W_new = (1.0 - alpha_t.unsqueeze(-1)) * W + S_new
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
        self.theta_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        for p in (self.alpha_proj, self.eta_proj, self.theta_proj):
            nn.init.zeros_(p.weight)
        nn.init.constant_(self.alpha_proj.bias, -4.6)   # sigmoid -> 0.01 (slow forget; paper-aligned)
        nn.init.constant_(self.eta_proj.bias,   -2.0)   # sigmoid -> 0.12 (modest momentum, was 0.9 before)
        nn.init.constant_(self.theta_proj.bias, -6.0)   # sigmoid -> 0.0025 (much smaller; was 0.5 before)
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
        theta_t = torch.sigmoid(_linear(self.theta_proj, chunk_mean))
        return tuple(self._gate_heads(g) for g in (alpha_t, eta_t, theta_t))

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
        # Keep fast_W / fast_S in fp32 unconditionally — bf16 accumulation
        # over many chunks compounds rounding noise into NaN at training scale.
        fast_dtype = torch.float32
        fast_W, fast_S = self._init_fast(B, fast_dtype)
        outs = []
        cs = self.chunk_size
        for s in range(0, T, cs):
            e = min(s + cs, T)
            chunk = x[:, s:e]
            state = tuple(fast_W + fast_S)
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

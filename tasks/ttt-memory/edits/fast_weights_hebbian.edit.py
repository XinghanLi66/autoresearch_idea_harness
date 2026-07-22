"""Fast Weights with Hebbian Outer-Product — Ba et al. 2016
"Using Fast Weights to Attend to the Recent Past" (NeurIPS 2016, arXiv:1610.06258).

The fast-weight matrix A is updated by a pure Hebbian outer product
(no surprise gradient, no autograd-based meta-objective):

    A(t) = λ · A(t-1) + η · h(t) · h(t)^T            (Eq. 1, paper line 116)

Reading uses the iterative inner loop with layer normalization
(Eq. 5, paper line 139):

    h_0(t+1)     = q                                  (preliminary)
    h_{s+1}(t+1) = f(LN[q + A(t) · h_s(t+1)])         for s = 0..S-1
    h(t+1)       = h_S(t+1)

with paper defaults λ = 0.95, η = 0.5, S = 1, f = SiLU. The key/value
projection is autoassociative (h appears on both sides of the outer
product), matching Ba 2016's original formulation.

The chunk-wise implementation is the exact unroll of Eq. 1 over all tokens in
the chunk:

    A_new = λ^T A_old + η · Σ_t λ^(T-1-t) h_t h_t^T

where T is the chunk length. compute_fast_gradients returns the negated
discounted outer-product sum, and apply_update applies the λ^T bulk decay.
compute_surprise is retained only as a compatible diagnostic objective; the
inner-loop LN-stabilized read is used for the layer's output.

This baseline differs from the other three on the most fundamental axis
— its memory update is NOT derived from an outer-loop surprise loss; it
is direct associative storage. Hyperparameters λ, η are constants per
the paper, not learned.

Ba et al. discuss the space cost of one full fast matrix per minibatch
sequence and reformulate the read as stored hidden-vector comparisons. In this
benchmark's explicit-matrix implementation, we use the same idea per head:
each 64-d head stores its own autoassociative h h^T matrix, giving a
block-diagonal multi-head fast memory rather than a full 1024 x 1024 matrix.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_REGION = """\
def init_memory_state(memory, dim):
    memory.memory_dim = dim
    memory.depth = 1                      # Ba 2016 uses a single fast-weight matrix
    memory.chunk_size = 256
    memory.lambda_decay = 0.95
    memory.eta_lr = 0.5
    memory.inner_steps = 1
    memory.head_dim = 64
    assert dim % memory.head_dim == 0
    memory.num_memory_heads = dim // memory.head_dim
    assert dim == memory.num_memory_heads * memory.head_dim
    A0 = torch.zeros(memory.num_memory_heads, memory.head_dim, memory.head_dim)
    memory.register_buffer('M1_init', A0)


def _linear(layer, x):
    return F.linear(x, layer.weight, layer.bias)


def compute_keys_values(memory, x):
    # Ba 2016 stores h(t) h(t)^T. We apply that autoassociation independently
    # inside each head; k_proj and v_proj are tied in __init__ to represent one h.
    B, T, _ = x.shape
    H, D = memory.num_memory_heads, memory.head_dim
    h = _linear(memory.k_proj, x).view(B, T, H, D)
    return h, h


def compute_surprise(memory, fast_W, keys, values):
    # Compatible diagnostic objective; the update path uses closed-form Eq. 1.
    pred = memory._apply_fast_mlp(fast_W, keys)
    return -(pred * values).sum(dim=-1).sum(dim=1).sum()


def compute_fast_gradients(memory, fast_W, keys, values):
    # Ba 2016 Eq. 1 unrolled over a chunk:
    # A_new = lambda^T A + eta * sum_t lambda^(T-1-t) h_t h_t^T.
    T = keys.shape[1]
    lam = memory.lambda_decay
    powers = torch.arange(T - 1, -1, -1, device=keys.device, dtype=torch.float32)
    weights = torch.pow(torch.tensor(lam, device=keys.device, dtype=torch.float32), powers).to(values.dtype)
    return [-torch.einsum('t,btho,bthi->bhoi', weights, values, keys)]


def apply_update(memory, fast_W, fast_S, grads):
    # Ba 2016 Eq. 1: per-token lambda decay, collapsed over this chunk.
    lam = memory.lambda_decay
    eta = memory.eta_lr
    T = memory._last_chunk_len
    decay = lam ** T
    new_W = [decay * W - eta * g for W, g in zip(fast_W, grads)]
    return new_W, fast_S


class TitansMemoryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = self.k_proj                 # Ba 2016 is autoassociative, no W_K/W_V split.
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ln_inner = nn.LayerNorm(64)
        self.use_checkpoint = False
        init_memory_state(self, self.dim)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_memory_heads, self.head_dim)

    def _merge_heads(self, x):
        B, T, H, D = x.shape
        return x.contiguous().view(B, T, H * D)

    def _apply_fast_mlp(self, fast_W, x):
        return torch.einsum('bthi,bhoi->btho', x, fast_W[0])

    def _inner_loop_read(self, fast_W, q):
        # Ba 2016 Eq. 5: h_{s+1} = f(LN[q + A h_s]) starting from h_0 = q.
        A = fast_W[0]
        h = q
        for _ in range(self.inner_steps):
            z = q + torch.einsum('bthi,bhoi->btho', h, A)
            h = F.silu(F.layer_norm(
                z, (self.head_dim,), self.ln_inner.weight, self.ln_inner.bias))
        return h

    def _init_fast(self, batch_size, dtype):
        W = [self.M1_init.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()]
        if self.training:
            for w in W:
                w.requires_grad_(True)
        S = [torch.zeros_like(w) for w in W]
        return W, S

    def _chunk_step(self, chunk, *state):
        depth = self.depth
        fast_W = list(state[:depth])
        fast_S = list(state[depth:])
        chunk_f = chunk

        q = self._split_heads(_linear(self.q_proj, chunk_f))
        out = self._inner_loop_read(fast_W, q)
        keys, values = compute_keys_values(self, chunk_f)
        self._last_chunk_len = keys.shape[1]
        grads = compute_fast_gradients(self, fast_W, keys, values)
        fast_W, fast_S = apply_update(self, fast_W, fast_S, grads)
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

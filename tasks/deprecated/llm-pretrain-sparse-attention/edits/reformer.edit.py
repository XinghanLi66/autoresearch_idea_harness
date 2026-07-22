"""Reformer — LSH (locality-sensitive hashing) attention.

Reference: Kitaev, Kaiser, Levskaya, "Reformer: The Efficient Transformer",
ICLR 2020.

Each query is hashed via a frozen random rotation matrix R; queries and keys
falling into the same bucket attend to each other. We then sort tokens by
bucket, group adjacent same-bucket tokens into chunks of size CHUNK=64, and
let each query attend to keys in (a) its own chunk and (b) the previous
chunk (the standard Reformer "chunk + previous-chunk" trick).

For causal LM we additionally mask out keys with position > query position.

Hyperparameters at T=1024, n_hashes=1, CHUNK=64:
  - Per query: at most 2 * CHUNK = 128 keys (own + previous chunk).
  - Causal-pair density: sum_t min(128,t+1) / (T*(T+1)/2) ~= 0.234.
  - Sequences are padded to a CHUNK multiple and trimmed back after attention.

Determinism: the random rotation R is sampled from the MLS-Bench SEED at
init and registered as a non-persistent buffer so it is stable across
forward passes.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_REFORMER_ATTENTION = """\
class SparseSelfAttention(nn.Module):
    \"\"\"LSH attention (Reformer): hash-bucket + chunked local attention.\"\"\"
    N_HASHES = 1
    N_BUCKETS = 16   # number of LSH buckets
    CHUNK = 64       # tokens per chunk after bucket-sort

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
        T = config.block_size
        D_head = config.n_embd // config.n_head
        # Random rotation R: (D_head, n_buckets // 2). argmax(concat(qR,-qR))
        # gives one of n_buckets buckets. Frozen, deterministic.
        gen = torch.Generator(device='cpu').manual_seed(int(os.environ.get('SEED', '42')))
        R = torch.randn(D_head, self.N_BUCKETS // 2, generator=gen)
        self.register_buffer('rot', R, persistent=False)
        # Density: each query attends to ~2*CHUNK keys (own + previous chunk).
        # Conservatively use 2*CHUNK / T as causal-pair density.
        # Average causal context = (T+1)/2; per-query attended ~ 2*CHUNK
        # (capped by causal). Actual causal density:
        #   sum_t min(2*CHUNK, t+1) / (T*(T+1)/2)
        Cw = 2 * self.CHUNK
        attended = sum(min(Cw, t + 1) for t in range(T))
        causal_pairs = T * (T + 1) / 2
        self.reported_density = float(attended / causal_pairs)
        self.is_dense_oracle = False
        self.use_pos_emb = True

    def _hash_buckets(self, x):
        \"\"\"x: (B,H,T,D) -> bucket ids: (B,H,T) in [0, N_BUCKETS).\"\"\"
        # Project: (B,H,T,N_BUCKETS//2)
        proj = torch.einsum('bhtd,dn->bhtn', x, self.rot.to(x.dtype))
        # Concatenate (proj, -proj) -> (B,H,T,N_BUCKETS), pick argmax
        cat = torch.cat([proj, -proj], dim=-1)
        return cat.argmax(dim=-1)   # (B,H,T)

    def forward(self, x):
        B, T_orig, C = x.size()
        H = self.n_head
        D = C // H
        Cw = self.CHUNK
        if T_orig % Cw != 0 or T_orig < Cw:
            T = ((T_orig + Cw - 1) // Cw) * Cw
            pad = T - T_orig
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        else:
            T = T_orig
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)   # (B,H,T,D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Use shared QK hashing: hash q (Reformer ties Q and K; we hash q here)
        # so queries and keys in the same bucket are routed together.
        buckets = self._hash_buckets(q)   # (B,H,T)
        # Stable sort key: bucket * T + position so within-bucket the original
        # order (and therefore causality) is preserved on a per-bucket basis.
        pos = torch.arange(T, device=x.device).view(1, 1, T).expand(B, H, T)
        sort_key = buckets * T + pos
        sorted_idx = sort_key.argsort(dim=-1)              # (B,H,T) — perm
        inv_idx = sorted_idx.argsort(dim=-1)               # inverse perm

        # Gather q,k,v in sorted order
        gather_d = sorted_idx.unsqueeze(-1).expand(B, H, T, D)
        q_s = q.gather(2, gather_d)
        k_s = k.gather(2, gather_d)
        v_s = v.gather(2, gather_d)
        pos_s = pos.gather(2, sorted_idx)                  # (B,H,T) — original position of each sorted token

        # Reshape into chunks of size Cw
        n_chunks = T // Cw
        q_c = q_s.view(B, H, n_chunks, Cw, D)
        k_c = k_s.view(B, H, n_chunks, Cw, D)
        v_c = v_s.view(B, H, n_chunks, Cw, D)
        pos_c = pos_s.view(B, H, n_chunks, Cw)

        # Concatenate previous chunk: keys = [prev_chunk, current_chunk]
        # For chunk 0, prev = current (no leak; causal mask will handle it).
        k_prev = torch.roll(k_c, shifts=1, dims=2)
        v_prev = torch.roll(v_c, shifts=1, dims=2)
        pos_prev = torch.roll(pos_c, shifts=1, dims=2)
        # First chunk's "prev" is itself rolled around — mask it out via causal
        # mask (positions there are larger than current chunk's positions, so
        # the position-based causal mask will suppress them).
        k_full = torch.cat([k_prev, k_c], dim=3)        # (B,H,n_chunks,2*Cw,D)
        v_full = torch.cat([v_prev, v_c], dim=3)
        pos_full = torch.cat([pos_prev, pos_c], dim=3)  # (B,H,n_chunks,2*Cw)

        # Position-based causal mask: query at (chunk i, slot j) sees key at
        # (chunk i, slot k) iff pos_c[i,j] >= pos_full[i,k].
        causal = pos_c.unsqueeze(-1) >= pos_full.unsqueeze(-2)  # (B,H,n_chunks,Cw,2*Cw)
        attn_mask = torch.where(causal,
                                torch.zeros((), dtype=q.dtype, device=x.device),
                                torch.full((), float('-inf'), dtype=q.dtype, device=x.device))

        # SDPA over the (chunk) dim flattened into batch
        # Flatten (B*H*n_chunks) batches: q (Cw,D), kv (2*Cw,D)
        Bf = B * H * n_chunks
        q_f = q_c.reshape(Bf, Cw, D)
        k_f = k_full.reshape(Bf, 2 * Cw, D)
        v_f = v_full.reshape(Bf, 2 * Cw, D)
        m_f = attn_mask.reshape(Bf, Cw, 2 * Cw)
        # SDPA expects (B, H, Lq, D); we treat Bf as batch with 1 'head'.
        y_f = torch.nn.functional.scaled_dot_product_attention(
            q_f.unsqueeze(1), k_f.unsqueeze(1), v_f.unsqueeze(1),
            attn_mask=m_f.unsqueeze(1),
            dropout_p=self.dropout if self.training else 0, is_causal=False)
        y_s = y_f.squeeze(1).view(B, H, T, D)            # sorted-order outputs

        # Un-sort back to original token order
        y = torch.empty_like(y_s)
        y.scatter_(2, sorted_idx.unsqueeze(-1).expand(B, H, T, D), y_s)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y[:, :T_orig, :]
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 34,
        "end_line": 72,
        "content": _REFORMER_ATTENTION,
    },
]

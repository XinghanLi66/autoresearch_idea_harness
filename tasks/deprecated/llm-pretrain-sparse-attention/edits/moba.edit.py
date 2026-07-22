"""MoBA — Mixture of Block Attention.

Reference: Lu et al., "MoBA: Mixture of Block Attention for Long-Context LLMs"
(Moonshot AI, 2025).

Per-token routing: for each query token t, score every block j by q_t . mean(K_j),
keep the top-K causal blocks (always including the diagonal block t belongs to),
and let t attend only to keys inside those blocks. This is *finer* than NSA's
selected branch (which routes per query block).

Hyperparameters at T=1024, Bk=64 (N=16):
  - TOP_K = 2 blocks per token  (incl. diagonal). Per-token, the query attends
    to at most 2 * Bk = 128 keys (capped by causality). Causal-pair density:
        sum_t min(128, t+1) / (T*(T+1)/2)
        = (128*129/2 + (T-128)*128) / (T*(T+1)/2)
        = 122944 / 524800 ~= 0.234  -> within the 0.25 budget.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_MOBA_ATTENTION = """\
class SparseSelfAttention(nn.Module):
    \"\"\"MoBA: per-token block routing — top-K causal blocks per query token.\"\"\"
    BLOCK = 64
    TOP_K = 2   # per-token block budget (incl. diagonal block)

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
        Bk = self.BLOCK
        N = T // Bk
        K_top = self.TOP_K
        # Reported density: per-token, attend to up to TOP_K blocks
        # (each block has Bk keys), capped by causal availability.
        # For token t (0-indexed), full causal = t+1 keys.
        # MoBA gives at most min(K_top * Bk, t+1) keys (causal-safe within blocks).
        # Density vs full causal:
        attended = 0
        causal_pairs = 0
        for t in range(T):
            attended += min(K_top * Bk, t + 1)
            causal_pairs += t + 1
        self.reported_density = float(attended / causal_pairs)
        self.is_dense_oracle = False
        self.use_pos_emb = True

    def forward(self, x):
        B, T_orig, C = x.size()
        Bk = self.BLOCK
        H = self.n_head
        D = C // H
        K_top = self.TOP_K
        if T_orig % Bk != 0 or T_orig < Bk:
            T = ((T_orig + Bk - 1) // Bk) * Bk
            pad = T - T_orig
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        else:
            T = T_orig
        N = T // Bk
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)   # (B,H,T,D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        scale = 1.0 / math.sqrt(D)

        # Block-mean keys: (B,H,N,D)
        k_blocks = k.view(B, H, N, Bk, D).mean(dim=3)
        # Per-token routing scores: (B,H,T,N) = q . mean(K_j)
        scores = torch.einsum('bhtd,bhnd->bhtn', q, k_blocks) * scale

        # Block-level causal mask: token t can pick block j only if j*Bk <= t
        # (i.e., the first key of block j is <= t).
        t_idx = torch.arange(T, device=x.device)
        n_idx = torch.arange(N, device=x.device)
        # j is causally available iff j*Bk <= t  <=>  j <= t // Bk
        causal_qb = (n_idx[None, :] * Bk) <= t_idx[:, None]   # (T,N)
        scores = scores.masked_fill(~causal_qb.view(1, 1, T, N), float('-inf'))

        # Force-include the diagonal block (the one t belongs to)
        diag_blk = (t_idx // Bk)        # (T,)
        diag_oh = torch.zeros(T, N, dtype=torch.bool, device=x.device)
        diag_oh[t_idx, diag_blk] = True
        # Mask diagonal block out of the candidate pool, then top-K (K-1) on the rest
        scores_other = scores.masked_fill(diag_oh.view(1, 1, T, N), float('-inf'))
        kk = min(K_top - 1, N - 1) if K_top > 1 else 0
        if kk > 0:
            topk_idx = scores_other.topk(kk, dim=-1).indices   # (B,H,T,kk)
        else:
            topk_idx = torch.empty(B, H, T, 0, dtype=torch.long, device=x.device)
        diag_idx = diag_blk.view(1, 1, T, 1).expand(B, H, T, 1)
        sel = torch.cat([topk_idx, diag_idx], dim=-1)          # (B,H,T,K_top)
        sel_mask = torch.zeros(B, H, T, N, dtype=torch.bool, device=x.device)
        sel_mask.scatter_(-1, sel, True)
        sel_mask = sel_mask & causal_qb.view(1, 1, T, N)

        # Expand to per-token-key mask: (B,H,T,T)
        # token t can attend to key u iff sel_mask[t, block(u)] AND u <= t
        k_tok_blk = (t_idx // Bk)
        block_allow = sel_mask[:, :, :, k_tok_blk]              # (B,H,T,T)
        causal_tok = (t_idx[:, None] >= t_idx[None, :])          # (T,T)
        allowed = block_allow & causal_tok.view(1, 1, T, T)
        attn_mask = torch.where(allowed,
                                torch.zeros((), dtype=q.dtype, device=x.device),
                                torch.full((), float('-inf'), dtype=q.dtype, device=x.device))
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0, is_causal=False)
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
        "content": _MOBA_ATTENTION,
    },
]

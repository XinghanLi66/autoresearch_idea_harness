"""Full NSA (Native Sparse Attention) — 3 branches with learned gating.

Reference: Yuan et al., "Native Sparse Attention" (DeepSeek 2025).

Three attention branches combined via per-token learned gates:

  1. Compressed branch  — each query attends to ALL block-mean-pooled K/V
     (cheap global summary). Density ~ N/T = 16/1024 = 0.016 at T=1024,
     Bk=64.
  2. Selected branch    — top-K causal blocks chosen via the compressed
     attention scores; we use TOP_K=1 (diagonal-only) so the total density
     stays under the 0.25 budget. This branch is computed with a single
     batched per-block SDPA — no full (B,H,T,T) mask is materialized.
  3. Sliding branch     — local window W=48 for fine-grained nearby tokens.
     Computed with a per-query-block gather over a (Bk + W) key window.

A small MLP (`nn.Linear(n_embd, 3)`) produces per-token logits which a
softmax converts into mixing weights (g_c, g_s, g_w) over the 3 branch
outputs. Reported density is the SUM of the three branch densities (the
runtime touches all of them).

Density estimate at T=1024, Bk=64 (N=16), TOP_K=1, W=48:
  - compressed:  N/T                   ~= 0.016
  - selected:    sum_i min(K,i+1)/Cp   ~= 0.118
  - sliding:     sum min(t+1,W)/Cp     ~= 0.092 (W=48)
  TOTAL                                ~= 0.226 (under 0.25 budget)

Memory note: a previous version materialized a full (B,H,T,T) bf16 mask
for the selected branch (~2 GB / layer × 24 layers = 48 GB), causing OOM
on H100 at BSZ=64. The current implementation eliminates that mask with
gather-based block-local SDPA, and additionally wraps the 3-branch core
in torch.utils.checkpoint to halve activation memory.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_NSA_FULL_ATTENTION = """\
class SparseSelfAttention(nn.Module):
    \"\"\"Full NSA with 3 branches (compressed + selected + sliding) + learned gating.\"\"\"
    BLOCK = 64
    TOP_K = 1          # diagonal-only — keeps total density 0.226 < 0.25 budget
    WINDOW = 48        # sliding-window size

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Per-token gate logits over the 3 branches (compressed, selected, sliding)
        self.gate = nn.Linear(config.n_embd, 3, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        T = config.block_size
        Bk = self.BLOCK
        N = T // Bk
        W = self.WINDOW
        # Compressed branch density
        d_comp = float(N) / float(T)
        # Selected-branch density (block-pair counted, causal)
        attended_pairs = 0
        causal_pairs_blk = 0
        for i in range(N):
            attended_pairs += min(self.TOP_K, i + 1)
            causal_pairs_blk += i + 1
        d_sel = float(attended_pairs / causal_pairs_blk)
        # Sliding-branch density
        causal_pairs_tok = T * (T + 1) / 2
        win_pairs = sum(min(t + 1, W) for t in range(T))
        d_win = float(win_pairs / causal_pairs_tok)
        self.reported_density = d_comp + d_sel + d_win
        self.is_dense_oracle = False
        self.use_pos_emb = True

    def _three_branch_forward(self, x):
        B, T_orig, C = x.size()
        Bk = self.BLOCK
        H = self.n_head
        D = C // H
        K_top = self.TOP_K
        W = self.WINDOW
        # Pad T up to a multiple of Bk for block-level operations. Required at
        # eval time (e.g. lm-eval-harness uses T as low as 16, well below Bk=64).
        # Padded tokens contribute zero attention via causal masking; we slice
        # the output back to T_orig before returning.
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
        dp = self.dropout if self.training else 0.0

        # ── Branch 1: compressed (block-mean K/V) ────────────────────────
        # K/V have shape (B,H,N,D) — N=16 keys total, so attention is cheap.
        k_blocks = k.view(B, H, N, Bk, D).mean(dim=3)
        v_blocks = v.view(B, H, N, Bk, D).mean(dim=3)
        # Token-query -> key-block causal mask: q's block index STRICTLY > key-block.
        # Using >= would leak: query at pos t in block b_t would attend to the
        # mean of block b_t, which is computed from tokens [b_t*Bk .. (b_t+1)*Bk-1]
        # — i.e. positions ≥ t+1 are already pooled into the mean. Use STRICT
        # past-only blocks here; the within-block diagonal is covered by the
        # selected branch (which force-includes the diagonal block) and recent
        # locals are covered by the sliding branch.
        q_tok_blk = torch.arange(T, device=x.device) // Bk             # (T,)
        blk_idx = torch.arange(N, device=x.device)
        causal_qb_bool = q_tok_blk[:, None] > blk_idx[None, :]         # (T,N) bool
        # bool mask -> SDPA accepts bool (True = attend). (1,1,T,N) is tiny.
        comp_mask = causal_qb_bool.view(1, 1, T, N)
        o_c = torch.nn.functional.scaled_dot_product_attention(
            q, k_blocks, v_blocks, attn_mask=comp_mask,
            dropout_p=dp, is_causal=False)                              # (B,H,T,D)
        # First Bk tokens (block 0) have no strictly-earlier blocks in the
        # compressed view, so SDPA returns NaN for those rows. Zero them so
        # the gated sum doesn't propagate NaN — those tokens rely on the
        # selected/sliding branches.
        o_c = torch.nan_to_num(o_c, nan=0.0)

        # ── Branch 2: selected (top-K causal blocks via compressed scores) ──
        # TOP_K=1 is a special case: each query block attends exactly to
        # keys/values from its diagonal block. We compute this with a single
        # batched matmul over per-block tiles — NO (B,H,T,T) mask is built.
        if K_top == 1:
            # Reshape to per-block tiles: (B,H,N,Bk,D)
            qb = q.view(B, H, N, Bk, D)
            kb = k.view(B, H, N, Bk, D)
            vb = v.view(B, H, N, Bk, D)
            # Per-block causal SDPA: each (Bk,Bk) tile, causal within block.
            # Flatten leading dims so we can call SDPA in one shot.
            qb_flat = qb.reshape(B * H * N, Bk, D)
            kb_flat = kb.reshape(B * H * N, Bk, D)
            vb_flat = vb.reshape(B * H * N, Bk, D)
            # Add a singleton head dim for SDPA: (BHN,1,Bk,D)
            o_s_flat = torch.nn.functional.scaled_dot_product_attention(
                qb_flat.unsqueeze(1), kb_flat.unsqueeze(1), vb_flat.unsqueeze(1),
                attn_mask=None, dropout_p=dp, is_causal=True).squeeze(1)
            o_s = o_s_flat.view(B, H, N, Bk, D).reshape(B, H, T, D)
        else:
            # General TOP_K>1: gather K_top blocks of K/V per query block.
            # We compute selection scores from block-mean queries vs block-mean keys.
            q_blocks = q.view(B, H, N, Bk, D).mean(dim=3)              # (B,H,N,D)
            scores = torch.einsum('bhnd,bhmd->bhnm', q_blocks, k_blocks) * scale
            idx_n = torch.arange(N, device=x.device)
            causal_blk = idx_n[:, None] >= idx_n[None, :]              # (N,N)
            scores = scores.masked_fill(~causal_blk, float('-inf'))
            diag_eye = torch.eye(N, dtype=torch.bool, device=x.device)
            scores_other = scores.masked_fill(diag_eye, float('-inf'))
            kk = min(K_top - 1, N - 1)
            topk_idx = scores_other.topk(kk, dim=-1).indices           # (B,H,N,kk)
            diag = idx_n.view(1, 1, N, 1).expand(B, H, N, 1)
            sel = torch.cat([topk_idx, diag], dim=-1)                  # (B,H,N,K_top)
            # Gather K_top blocks of K/V for each (b,h,i): output (B,H,N,K_top,Bk,D)
            kb_full = k.view(B, H, N, Bk, D)
            vb_full = v.view(B, H, N, Bk, D)
            sel_e = sel.view(B, H, N, K_top, 1, 1).expand(B, H, N, K_top, Bk, D)
            k_sel = torch.gather(
                kb_full.unsqueeze(2).expand(B, H, N, N, Bk, D), 3, sel_e)
            v_sel = torch.gather(
                vb_full.unsqueeze(2).expand(B, H, N, N, Bk, D), 3, sel_e)
            # Flatten K_top * Bk as the key dim: (B,H,N,K_top*Bk,D)
            k_sel = k_sel.reshape(B, H, N, K_top * Bk, D)
            v_sel = v_sel.reshape(B, H, N, K_top * Bk, D)
            # Reshape queries per block: (B,H,N,Bk,D)
            qb = q.view(B, H, N, Bk, D)
            # Build a small per-tile causal mask of shape (N,K_top,Bk,Bk) bool.
            # For the selected key-blocks that equal the query-block (diagonal),
            # apply within-block causality; for strictly-earlier blocks, all keys
            # are allowed. sel < i -> all allowed; sel == i -> within-block causal.
            i_idx = idx_n.view(N, 1)
            # sel has gradient-free indices; pull to CPU-free bool tile mask.
            same = (sel == i_idx.view(1, 1, N, 1))                      # (B,H,N,K_top)
            # Build (B,H,N,Bk, K_top*Bk) attn mask
            tri_causal = torch.tril(torch.ones(Bk, Bk, dtype=torch.bool, device=x.device))
            full_block = torch.ones(Bk, Bk, dtype=torch.bool, device=x.device)
            # For each (b,h,n,k): pick tri_causal if same else full_block.
            same_e = same.view(B, H, N, K_top, 1, 1)
            tile_mask = torch.where(same_e, tri_causal, full_block)     # (B,H,N,K_top,Bk,Bk)
            tile_mask = tile_mask.permute(0, 1, 2, 4, 3, 5).reshape(B, H, N, Bk, K_top * Bk)
            # Flatten leading: (B*H*N, 1, Bk, K_top*Bk)
            qb_f = qb.reshape(B * H * N, Bk, D).unsqueeze(1)
            k_f = k_sel.reshape(B * H * N, K_top * Bk, D).unsqueeze(1)
            v_f = v_sel.reshape(B * H * N, K_top * Bk, D).unsqueeze(1)
            m_f = tile_mask.reshape(B * H * N, 1, Bk, K_top * Bk)
            o_s_f = torch.nn.functional.scaled_dot_product_attention(
                qb_f, k_f, v_f, attn_mask=m_f, dropout_p=dp, is_causal=False).squeeze(1)
            o_s = o_s_f.view(B, H, N, Bk, D).reshape(B, H, T, D)

        # ── Branch 3: sliding window (per-query-block gather, no full TxT mask) ──
        # For query block i, keys are tokens [i*Bk - (W-1), i*Bk + Bk - 1]
        # capped at [0, T-1]. Pad on the left with zeros so we can gather a
        # uniform window length L = Bk + W - 1.
        L = Bk + W - 1
        # Build a left-pad of (W-1) zero tokens for keys/values
        pad = W - 1
        if pad > 0:
            k_pad = torch.nn.functional.pad(k, (0, 0, pad, 0))         # (B,H,T+pad,D)
            v_pad = torch.nn.functional.pad(v, (0, 0, pad, 0))
        else:
            k_pad = k
            v_pad = v
        # Gather tiles: for each query block i in [0..N-1], rows [i*Bk : i*Bk+L]
        # of k_pad (since k_pad is shifted right by `pad`, this corresponds to
        # original token range [i*Bk-(W-1) : i*Bk+Bk-1]).
        # Build via reshape using as_strided-friendly indexing.
        starts = torch.arange(N, device=x.device) * Bk                  # (N,)
        offs = torch.arange(L, device=x.device)                         # (L,)
        gather_idx = (starts[:, None] + offs[None, :])                  # (N,L)
        # Expand for gather: (B,H,N,L,D)
        gather_idx_e = gather_idx.view(1, 1, N, L, 1).expand(B, H, N, L, D)
        # k_pad has shape (B,H,T+pad,D); we need to index dim 2 with (N,L) per block.
        kp = k_pad.unsqueeze(2).expand(B, H, N, k_pad.size(2), D)
        vp = v_pad.unsqueeze(2).expand(B, H, N, v_pad.size(2), D)
        k_win = torch.gather(kp, 3, gather_idx_e)                       # (B,H,N,L,D)
        v_win = torch.gather(vp, 3, gather_idx_e)
        qb = q.view(B, H, N, Bk, D)
        # Build per-tile sliding mask of shape (Bk, L): query position p in [0..Bk-1]
        # may attend to key window position l in [0..L-1] iff
        #   (l - (W-1)) <= p AND (l - (W-1)) >= p - (W-1)
        # i.e. p+1-W <= l - (W-1) - (-) ... simpler: original key offset is
        # l - (W-1) relative to the start of the query block; query offset p.
        # Allowed: (key_off <= p) AND (key_off > p - W).
        p_idx = torch.arange(Bk, device=x.device).view(Bk, 1)           # (Bk,1)
        key_off = torch.arange(L, device=x.device).view(1, L) - (W - 1)  # (1,L)
        win_allowed = (key_off <= p_idx) & (key_off > p_idx - W)        # (Bk,L)
        # Also mask out the left-pad positions for the FIRST query block
        # (those padded keys correspond to negative absolute positions).
        # We do this per-block by additionally requiring absolute key index
        # (gather_idx - pad) >= 0  -> i.e. gather_idx >= pad. Compute per (N, L):
        abs_valid = (gather_idx - pad) >= 0                              # (N,L) bool
        # Combine: per (n, p, l): win_allowed[p,l] AND abs_valid[n,l]
        combined = win_allowed.view(1, Bk, L) & abs_valid.view(N, 1, L)  # (N,Bk,L)
        m_w = combined.view(1, 1, N, Bk, L).expand(B, H, N, Bk, L)
        # Flatten leading dims for SDPA
        qb_f = qb.reshape(B * H * N, Bk, D).unsqueeze(1)
        kw_f = k_win.reshape(B * H * N, L, D).unsqueeze(1)
        vw_f = v_win.reshape(B * H * N, L, D).unsqueeze(1)
        mw_f = m_w.reshape(B * H * N, 1, Bk, L)
        o_w_f = torch.nn.functional.scaled_dot_product_attention(
            qb_f, kw_f, vw_f, attn_mask=mw_f, dropout_p=dp, is_causal=False).squeeze(1)
        o_w = o_w_f.view(B, H, N, Bk, D).reshape(B, H, T, D)

        # ── Gating: per-token softmax over (g_c, g_s, g_w) ───────────────
        g_logits = self.gate(x)                       # (B,T,3)
        g = torch.softmax(g_logits, dim=-1)           # (B,T,3)
        g_c = g[:, :, 0].unsqueeze(1).unsqueeze(-1)   # (B,1,T,1)
        g_s = g[:, :, 1].unsqueeze(1).unsqueeze(-1)
        g_w = g[:, :, 2].unsqueeze(1).unsqueeze(-1)
        y = g_c * o_c + g_s * o_s + g_w * o_w         # (B,H,T,D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        # Slice padded output back to the original sequence length.
        return y[:, :T_orig, :]

    @torch.compiler.disable
    def forward(self, x):
        # Wrap the heavy 3-branch path in gradient checkpointing during training
        # to roughly halve activation memory (the three attention outputs and
        # gather tiles are the dominant tensors). At inference we run directly.
        # @torch.compiler.disable: dynamo cannot trace the gating-MLP +
        # gather + gradient-checkpoint combo (BackendCompilerFailed:
        # RuntimeError: val). Disabling on this method only lets the rest
        # of the GPT model continue to compile (~2× speedup vs no-compile).
        # Same pattern as gla in llm-pretrain-linear-attention.
        if self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint as _ckpt
            return _ckpt(self._three_branch_forward, x, use_reentrant=False)
        return self._three_branch_forward(x)
"""

OPS = [
    # Compile stays ON. The SparseSelfAttention.forward is decorated with
    # @torch.compiler.disable so dynamo skips it (the gather/checkpoint hot
    # path otherwise crashes with BackendCompilerFailed); the rest of the
    # GPT model still compiles, recovering ~2× throughput vs full eager.
    # Same trick as gla in llm-pretrain-linear-attention.
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 34,
        "end_line": 72,
        "content": _NSA_FULL_ATTENTION,
    },
]

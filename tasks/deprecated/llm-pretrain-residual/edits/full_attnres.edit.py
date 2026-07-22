"""Full Attention Residuals baseline (AttnRes).

Replaces fixed residual accumulation (resid_lambdas * x + x0_lambdas * x0)
with softmax attention over all preceding layer outputs. Each layer gets a
learned pseudo-query vector w_l that attends over RMSNorm'd keys from all
previous layers, enabling content-aware depth-wise selection.

Reference: "Attention Residuals" (Kimi Team, arXiv:2603.15031, 2026)

Key formula:
  h_l = sum_i alpha_{i->l} * v_i
  alpha_{i->l} = softmax(w_l^T RMSNorm(k_i))
  where v_0 = h_1 (embedding), v_i = f_i(h_i) for i >= 1

Changes:
  - has_ve: unchanged (keep value embeddings)
  - CausalSelfAttention: unchanged (keep VE gate)
  - GPT.__init__: Replace resid/x0 lambdas with per-layer pseudo-query vectors
  - GPT.init_weights: Initialize pseudo-queries to zero (uniform attention at init)
  - GPT.forward: Full AttnRes loop attending over all previous layer outputs
  - GPT.estimate_flops: Update excluded param count
  - GPT.num_scaling_params: Update param groups
  - GPT.setup_optimizer: Update param groups for pseudo-queries
"""

_FILE = "nanochat/nanochat/gpt.py"

# ── 1. GPT.forward: Full AttnRes loop (lines 413-417) ───────────────────────
_FORWARD_LOOP = """\
        x0 = x  # save initial normalized embedding
        # Full AttnRes: v_0 = embedding, v_i = f_i(h_i) = block transformation (no residual)
        sources = [x0]  # source representations to attend over
        for i, block in enumerate(self.transformer.h):
            # Compute attention weights: alpha_{j->i} = softmax(w_i^T RMSNorm(v_j))
            stacked = torch.stack(sources, dim=0)  # (num_sources, B, T, D)
            keys_normed = F.rms_norm(stacked, (stacked.size(-1),))  # RMSNorm prevents magnitude bias
            logits = torch.einsum('d, n b t d -> n b t', self.attnres_queries[i], keys_normed)
            weights = logits.softmax(dim=0)  # softmax over source dimension
            x = torch.einsum('n b t, n b t d -> b t d', weights, stacked)  # h_l = weighted sum of sources
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            block_out = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            # Store the transformation f_l(h_l) = block_out - x (block has internal residuals)
            sources.append(block_out - x)
            x = block_out  # final x after last block is passed to output layer
"""

# ── 2. GPT.setup_optimizer: replace scalar groups with query groups (lines 358-382)
_OPTIMIZER = """\
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        attnres_params = [self.attnres_queries]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(attnres_params)

        # Scale the LR for the AdamW parameters by 1/sqrt(dmodel) (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, attnres queries)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=attnres_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
"""

# ── 3. GPT.num_scaling_params: update for attnres queries (lines 329-356)
_NUM_SCALING_PARAMS = """\
    def num_scaling_params(self):
        \"\"\"
        Return detailed parameter counts for scaling law analysis.
        \"\"\"
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.attnres_queries.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }
"""

# ── 4. GPT.estimate_flops: update excluded params (lines 314-318)
_ESTIMATE_FLOPS = """\
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings, value embeds, and attnres queries
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.attnres_queries.numel())
"""

# ── 5. GPT.init_weights: initialize attnres queries to zero (lines 227-238)
_INIT_WEIGHTS = """\
        # AttnRes pseudo-queries: zero init => uniform attention at start of training
        # (all sources get equal weight, reducing to simple averaging initially)
        self.attnres_queries.zero_()

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
"""

# ── 6. GPT.__init__: replace lambdas with attnres queries (lines 176-185)
_INIT = """\
        # Full Attention Residuals: per-layer pseudo-query vectors for depth-wise attention
        # Each layer l gets a learned query w_l in R^d that attends over all previous
        # layer outputs via softmax(w_l^T RMSNorm(v_j)) weighting.
        # Shape: (n_layer, n_embd) — one d-dimensional query per layer
        self.attnres_queries = nn.Parameter(torch.zeros(config.n_layer, config.n_embd))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 413,
        "end_line": 417,
        "content": _FORWARD_LOOP,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 358,
        "end_line": 382,
        "content": _OPTIMIZER,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 329,
        "end_line": 356,
        "content": _NUM_SCALING_PARAMS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 314,
        "end_line": 318,
        "content": _ESTIMATE_FLOPS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 227,
        "end_line": 238,
        "content": _INIT_WEIGHTS,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 176,
        "end_line": 185,
        "content": _INIT,
    },
]

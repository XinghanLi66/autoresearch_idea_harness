"""Hyper-Connections (HC) baseline.

Maintains m parallel residual streams updated via learned transition matrices.
Each layer mixes the streams into a single input via alpha, computes the
transformation, and distributes it back across streams via beta. The streams
evolve through a learned transition matrix A_l per layer.

Reference: Zhu et al., "Hyper-Connections" (arXiv:2409.19606, 2025)
Also discussed in Table 5 of "Attention Residuals" (arXiv:2603.15031):
  H_l = H_{l-1} A_l + f_{l-1}(H_{l-1} alpha_{l-1}) beta^T_{l-1}
where H in R^{d x m}, A_l in R^{m x m}, alpha/beta in R^m.

Changes:
  - GPT.__init__: Replace resid/x0 lambdas with HC parameters (A, alpha, beta per layer)
  - GPT.init_weights: Initialize HC params (A=identity, alpha=[1,0,...], beta=[1,0,...])
  - GPT.forward: Multi-stream HC loop
  - GPT.estimate_flops: Update excluded param count
  - GPT.num_scaling_params: Update param groups
  - GPT.setup_optimizer: Update param groups for HC parameters
"""

_FILE = "nanochat/nanochat/gpt.py"

# ── 1. GPT.forward: HC multi-stream loop (lines 413-417) ────────────────────
_FORWARD_LOOP = """\
        x0 = x  # save initial normalized embedding
        # Hyper-Connections: maintain m parallel streams H in (B, T, D, m)
        m = self.hc_num_streams
        # Initialize streams: stream 0 = x0, rest = zeros
        H = torch.zeros(*x.shape, m, device=x.device, dtype=x.dtype)  # (B, T, D, m)
        H[..., 0] = x0
        for i, block in enumerate(self.transformer.h):
            # Mix streams into single input: x = H @ alpha
            alpha = self.hc_alpha[i]  # (m,)
            x_input = (H * alpha).sum(dim=-1)  # (B, T, D)
            ve = self.value_embeds[str(i)](idx).to(x_input.dtype) if str(i) in self.value_embeds else None
            block_out = block(x_input, ve, cos_sin, self.window_sizes[i], kv_cache)
            # Transformation: f = block_out - x_input (block has internal residuals)
            f = block_out - x_input  # (B, T, D)
            # Distribute output across streams: f @ beta^T -> (B, T, D, m)
            beta = self.hc_beta[i]  # (m,)
            f_distributed = f.unsqueeze(-1) * beta  # (B, T, D, m)
            # Update streams: H = H @ A + f_distributed
            A = self.hc_A[i]  # (m, m)
            H = torch.einsum('b t d i, i j -> b t d j', H, A) + f_distributed
        # Final output: mix streams for output layer
        x = (H * self.hc_alpha_out).sum(dim=-1)  # (B, T, D)
"""

# ── 2. GPT.setup_optimizer: HC param groups (lines 358-382) ─────────────────
_OPTIMIZER = """\
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        hc_params = [self.hc_A, self.hc_alpha, self.hc_beta, self.hc_alpha_out]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(hc_params)

        # Scale the LR for the AdamW parameters by 1/sqrt(dmodel) (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, HC params)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=hc_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
"""

# ── 3. GPT.num_scaling_params: update for HC params (lines 329-356) ─────────
_NUM_SCALING_PARAMS = """\
    def num_scaling_params(self):
        \"\"\"
        Return detailed parameter counts for scaling law analysis.
        \"\"\"
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.hc_A.numel() + self.hc_alpha.numel() + self.hc_beta.numel() + self.hc_alpha_out.numel()
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

# ── 4. GPT.estimate_flops: update excluded params (lines 314-318) ───────────
_ESTIMATE_FLOPS = """\
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings, value embeds, and HC scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        hc_numel = self.hc_A.numel() + self.hc_alpha.numel() + self.hc_beta.numel() + self.hc_alpha_out.numel()
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel + hc_numel)
"""

# ── 5. GPT.init_weights: initialize HC params (lines 227-238) ───────────────
_INIT_WEIGHTS = """\
        # HC params: A=identity (neutral transition), alpha/beta = one-hot on stream 0
        m = self.hc_num_streams
        with torch.no_grad():
            for i in range(self.config.n_layer):
                self.hc_A.data[i] = torch.eye(m)  # identity transition
            self.hc_alpha.fill_(0.0)
            self.hc_alpha.data[:, 0] = 1.0  # use stream 0 as input
            self.hc_beta.fill_(0.0)
            self.hc_beta.data[:, 0] = 1.0  # write output to stream 0
            self.hc_alpha_out.fill_(0.0)
            self.hc_alpha_out.data[0] = 1.0  # read stream 0 for output

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
"""

# ── 6. GPT.__init__: HC parameters (lines 176-185) ──────────────────────────
_INIT = """\
        # Hyper-Connections: m parallel streams with learned transition and mixing
        # H_l = H_{l-1} A_l + f(H_{l-1} alpha) beta^T
        self.hc_num_streams = 4  # m=4 streams (paper uses m=4 in Table 1)
        m = self.hc_num_streams
        self.hc_A = nn.Parameter(torch.zeros(config.n_layer, m, m))  # transition matrices
        self.hc_alpha = nn.Parameter(torch.zeros(config.n_layer, m))  # input mixing
        self.hc_beta = nn.Parameter(torch.zeros(config.n_layer, m))  # output distribution
        self.hc_alpha_out = nn.Parameter(torch.zeros(m))  # final output mixing
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

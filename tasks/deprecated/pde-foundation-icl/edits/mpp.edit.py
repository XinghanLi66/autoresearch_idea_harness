"""MPP baseline — Multiple Physics Pretraining.

Reference: McCabe et al., "Multiple Physics Pretraining for Physical Surrogate
           Models", NeurIPS 2023 AI4Science Workshop. arXiv:2310.02994

Architecture: Standard ViT that processes each frame independently, then uses
mean-pooled demonstration representations to condition the query prediction
via cross-attention. Key differences from ICON:
  - No causal in-context mask (each frame processed independently)
  - Cross-attention conditioning instead of joint sequence processing
  - Learnable spatial positional encoding only (no function/pair identity encoding)
  - Demonstrations are aggregated via mean pooling before conditioning
"""

_FILE = "VICON/src/custom_model.py"

_CONTENT = """\
class CrossAttentionBlock(nn.Module):
    \"\"\"Cross-attention block for conditioning query on demonstration context.\"\"\"

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, context):
        # Cross-attention: query attends to context
        h = self.norm1(query)
        h = query + self.cross_attn(h, context, context)[0]
        h = h + self.ffn(self.norm2(h))
        return h


class CustomModel(nn.Module):
    \"\"\"MPP: Multiple Physics Pretraining.

    Processes each frame independently through a ViT encoder, then conditions
    each pair's prediction on mean-pooled demonstration representations via
    cross-attention layers. No causal mask is used.
    \"\"\"

    def __init__(self, cfg):
        super(CustomModel, self).__init__()
        self.cfg = cfg
        assert cfg["patch_num_in"] == cfg["patch_num_out"]

        p = cfg["patch_num_in"]
        d = cfg["transformer"]["dim_token"]
        c = cfg["transformer"]["dim_channel"]
        pr = cfg["patch_resolution"]

        # Linear projection: flatten each patch and project to token dim
        self.pre_proj = nn.Linear(c * pr ** 2, d)
        # Output projection: token dim back to patch pixel space
        self.post_proj = nn.Linear(d, c * pr ** 2)

        # Spatial positional encoding only (no function/pair identity)
        self.patch_pos_encoding = nn.Parameter(torch.randn(p * p, d))

        # Per-frame ViT encoder (processes each frame independently)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["transformer"]["nhead"],
            dim_feedforward=cfg["transformer"]["dim_feedforward"],
            dropout=cfg["transformer"]["dropout"],
            activation="gelu",
            batch_first=True,
        )
        n_enc = max(cfg["transformer"]["num_layers"] // 2, 4)
        self.frame_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_enc,
        )

        # Cross-attention decoder: query tokens attend to aggregated demo context
        n_dec = cfg["transformer"]["num_layers"] - n_enc
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                d, cfg["transformer"]["nhead"],
                cfg["transformer"]["dim_feedforward"],
                cfg["transformer"]["dropout"],
            )
            for _ in range(n_dec)
        ])

        # Self-attention decoder layers (interleaved with cross-attention)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg["transformer"]["nhead"],
                dim_feedforward=cfg["transformer"]["dim_feedforward"],
                dropout=cfg["transformer"]["dropout"],
                activation="gelu",
                batch_first=True,
            )
            for _ in range(n_dec)
        ])

    def forward(self, x_tuple):
        init, end, c_mask = x_tuple
        # Stack init and end: (B, pairs, 2, C, H, W)
        x = torch.cat((init[:, :, None, :, :, :], end[:, :, None, :, :, :]), dim=2)

        p = self.cfg["patch_num_in"]
        d = self.cfg["transformer"]["dim_token"]
        bs, pairs, _, c, h, w = x.shape

        # Patchify all frames
        feature = x.view(-1, *x.shape[-3:])  # (B*pairs*2, C, H, W)
        c_img, ph, pw = feature.shape[-3:]
        h_patch = ph // p
        w_patch = pw // p
        feature = patchify(feature, patch_num=p)  # (B*pairs*2, P*P, C*ph*pw)

        # Project to token dimension
        feature = self.pre_proj(feature)  # (B*pairs*2, P*P, d)

        # Add spatial positional encoding
        feature = feature + self.patch_pos_encoding  # broadcast over batch

        # Encode each frame independently (no inter-frame attention)
        feature = self.frame_encoder(feature)  # (B*pairs*2, P*P, d)

        # Reshape to (B, pairs, 2, P*P, d)
        feature = feature.view(bs, pairs, 2, p * p, d)

        # Build demonstration context: concatenate init and end, mean over demo pairs
        demo_init = feature[:, :-1, 0, :, :]   # (B, pairs-1, P*P, d)
        demo_end = feature[:, :-1, 1, :, :]    # (B, pairs-1, P*P, d)
        demo_ctx = torch.cat([demo_init, demo_end], dim=2)  # (B, pairs-1, 2*P*P, d)
        demo_ctx = demo_ctx.mean(dim=1)  # (B, 2*P*P, d)

        # Predict all pairs: each pair's init tokens conditioned on demo context
        all_init = feature[:, :, 0, :, :]  # (B, pairs, P*P, d)
        all_init_flat = all_init.reshape(bs * pairs, p * p, d)

        # Expand demo context for all pairs
        demo_ctx_expanded = demo_ctx.unsqueeze(1).expand(-1, pairs, -1, -1)
        demo_ctx_flat = demo_ctx_expanded.reshape(bs * pairs, -1, d)

        # Decode: interleave self-attention and cross-attention
        out = all_init_flat
        for self_attn, cross_attn in zip(self.self_attn_layers, self.cross_attn_layers):
            out = self_attn(out)
            out = cross_attn(out, demo_ctx_flat)

        # Project back to pixel space and depatchify
        out = self.post_proj(out)  # (B*pairs, P*P, C*ph*pw)
        out = depatchify(out, patch_num=p, c=c_img, h=h_patch, w=w_patch)
        out = out.view(bs, pairs, *out.shape[-3:])

        return out
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 89,
        "end_line": 205,
        "content": _CONTENT,
    },
]

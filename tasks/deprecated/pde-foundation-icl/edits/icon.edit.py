"""ICON baseline — In-Context Operator Network.

Reference: vendor/external_packages/VICON/src/models.py
Paper: Yang et al., "In-Context Operator Learning with Data Prompts for
       Differential Equation Problems", PNAS 2023.
       Cao et al., "VICON: Vision In-Context Operator Networks for Multi-Physics
       Fluid Dynamics Prediction", TMLR 2026. arXiv:2411.16063

Architecture: Patch-based vision transformer with:
  - Dual positional encoding (patch spatial + function/pair identity)
  - Alternating block lower-triangular causal attention mask
  - Linear pre/post projections for patch tokenization

This is the core ICON method adapted to vision (2D PDE fields). The causal
in-context mask enforces that the query cannot see its own answer, enabling
few-shot operator learning from demonstration pairs.
"""

_FILE = "VICON/src/custom_model.py"

_CONTENT = """\
class CustomModel(nn.Module):
    \"\"\"ICON: In-Context Operator Network (vision variant).

    Patch-based vision transformer with causal in-context attention mask.
    Uses dual positional encoding (spatial patches + function/pair identity).
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

        # Patch positional encoding: learnable embedding for each spatial patch position
        self.patch_pos_encoding = nn.Parameter(
            torch.randn(p * p, d)
        )

        # Function positional encoding: learnable embedding for each demo pair slot
        # demo_num * 2 because each pair has (init, end)
        self.func_pos_encoding = nn.Parameter(
            torch.randn(cfg["demo_num"] * 2, d)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["transformer"]["nhead"],
            dim_feedforward=cfg["transformer"]["dim_feedforward"],
            dropout=cfg["transformer"]["dropout"],
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg["transformer"]["num_layers"],
        )

        # Build alternating block causal mask
        mask = (
            1 - build_alternating_block_lowtri_mask(
                cfg["demo_num"], p * p, p * p
            )
        ).bool()
        self.register_buffer("mask", mask)

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

        # Add patch positional encoding (spatial position within each frame)
        if self.cfg.get("use_patch_pos_encoding", True):
            feature = feature + self.patch_pos_encoding  # broadcast over batch

        feature = feature.view(bs, -1, p * p, d)  # (B, pairs*2, P*P, d)

        # Add function positional encoding (which demo pair and init/end role)
        if self.cfg.get("use_func_pos_encoding", True):
            func_pe = self.func_pos_encoding.view(1, -1, 1, d)  # (1, demo_num*2, 1, d)
            func_pe = func_pe[:, :pairs * 2, :, :]  # trim to actual pairs
            feature = feature + func_pe

        # Flatten for transformer: (B, pairs*2*P*P, d)
        feature = feature.view(bs, -1, d)

        # Apply causal mask (trim to actual sequence length)
        seq_len = pairs * 2 * p * p
        mask = self.mask[:seq_len, :seq_len]
        feature = self.transformer(feature, mask=mask)

        # Reshape and extract input-side predictions
        feature = feature.view(bs, pairs, 2, p * p, d)
        feature = feature[:, :, 0, :, :]  # (B, pairs, P*P, d)

        # Project back to pixel space and depatchify
        feature = self.post_proj(feature)
        feature = feature.view(bs * pairs, *feature.shape[-2:])
        feature = depatchify(feature, patch_num=p, c=c_img, h=h_patch, w=w_patch)
        feature = feature.view(bs, pairs, *feature.shape[-3:])

        return feature
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

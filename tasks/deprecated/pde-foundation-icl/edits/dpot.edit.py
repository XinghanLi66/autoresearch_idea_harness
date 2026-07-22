"""DPOT baseline — Data-driven PDE Operator Transformer.

Reference: Hao et al., "DPOT: Auto-Regressive Denoising Operator Transformer
           for Large-Scale PDE Pre-Training", ICML 2024. arXiv:2403.03542

Architecture: Transformer with Fourier-based positional encodings and full
(non-causal) attention. Key differences from ICON:
  - 2D sinusoidal positional encoding (Fourier features) instead of learnable
  - Full bidirectional attention (no causal in-context mask)
  - Learnable function/pair identity encoding to distinguish demonstrations
  - All tokens from all pairs processed jointly in a single sequence

DPOT was originally designed for single-PDE auto-regressive prediction. Here
it is adapted to the in-context learning setting by processing all demonstration
and query frames jointly with full attention.
"""

_FILE = "VICON/src/custom_model.py"

_CONTENT = """\
class FourierPositionalEncoding2D(nn.Module):
    \"\"\"2D sinusoidal positional encoding using Fourier features.

    Generates fixed sinusoidal embeddings for a 2D grid of patch positions,
    following the Fourier feature approach used in DPOT.
    \"\"\"

    def __init__(self, d_model, patch_num):
        super().__init__()
        p = patch_num
        pe = torch.zeros(p * p, d_model)
        position_y = torch.arange(0, p).unsqueeze(1).float()  # (p, 1)
        position_x = torch.arange(0, p).unsqueeze(1).float()  # (p, 1)

        # Use half the dimensions for each spatial axis
        d_half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half)
        )

        # Y-axis encoding
        pe_y = torch.zeros(p, d_half)
        pe_y[:, 0::2] = torch.sin(position_y * div_term[:d_half // 2])
        pe_y[:, 1::2] = torch.cos(position_y * div_term[:d_half // 2])

        # X-axis encoding
        pe_x = torch.zeros(p, d_half)
        pe_x[:, 0::2] = torch.sin(position_x * div_term[:d_half // 2])
        pe_x[:, 1::2] = torch.cos(position_x * div_term[:d_half // 2])

        # Combine: for each (y, x) position, concatenate y-encoding and x-encoding
        for i in range(p):
            for j in range(p):
                pe[i * p + j, :d_half] = pe_y[i]
                pe[i * p + j, d_half:] = pe_x[j]

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, P*P, d)

    def forward(self, x):
        # x: (B, P*P, d) or broadcastable
        return x + self.pe


class CustomModel(nn.Module):
    \"\"\"DPOT: Data-driven PDE Operator Transformer.

    Transformer with 2D Fourier positional encoding and full attention.
    All demonstration and query frames are processed jointly without
    causal masking. Uses learnable pair identity encoding.
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

        # 2D Fourier positional encoding (fixed sinusoidal)
        self.fourier_pos_enc = FourierPositionalEncoding2D(d, p)

        # Learnable function/pair identity encoding
        self.func_pos_encoding = nn.Parameter(
            torch.randn(cfg["demo_num"] * 2, d)
        )

        # Transformer encoder with full attention (no causal mask)
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

        # Add 2D Fourier positional encoding (spatial position within each frame)
        feature = self.fourier_pos_enc(feature)

        feature = feature.view(bs, -1, p * p, d)  # (B, pairs*2, P*P, d)

        # Add function/pair identity encoding
        if self.cfg.get("use_func_pos_encoding", True):
            func_pe = self.func_pos_encoding.view(1, -1, 1, d)
            func_pe = func_pe[:, :pairs * 2, :, :]
            feature = feature + func_pe

        # Flatten for transformer: (B, pairs*2*P*P, d)
        feature = feature.view(bs, -1, d)

        # Full attention — no causal mask (DPOT uses bidirectional attention)
        feature = self.transformer(feature)

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

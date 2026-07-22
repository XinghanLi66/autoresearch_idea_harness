"""Adaptive LayerNorm (AdaLN) conditioning baseline.

Inspired by DiT (Peebles & Xie, 2023), the observation embedding modulates
LayerNorm parameters. The combined observation + timestep embedding produces
per-layer (shift, scale, gate) triplets:
    h = gate * (scale * LayerNorm(h) + shift)

This zero-initializes the gate to make conditioning start as identity,
following the AdaLN-Zero strategy from DiT.
"""

_FILE = "CleanDiffuser/custom_conditioning.py"

_ADALN = """\
class ConditioningModule(nn.Module):
    \"\"\"Adaptive LayerNorm conditioning (AdaLN-Zero).

    Combines observation and timestep embeddings to produce per-layer
    LayerNorm modulation parameters (shift, scale, gate). Follows the
    AdaLN-Zero strategy from DiT: gate is initialized to zero so that
    conditioning starts as identity.
    \"\"\"

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Encode observation
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Per-layer AdaLN: combined embedding -> (shift, scale, gate)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim, elementwise_affine=False)
            for _ in range(4)
        ])
        self.adaln_modulations = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
            for _ in range(4)
        ])
        # Zero-initialize the modulation output layers (AdaLN-Zero)
        for mod in self.adaln_modulations:
            nn.init.constant_(mod[-1].weight, 0)
            nn.init.constant_(mod[-1].bias, 0)

    def encode_condition(self, obs):
        return self.obs_encoder(obs)

    def condition_features(self, h, t_emb, cond, layer_idx):
        # Combine obs and timestep embeddings
        combined = cond + t_emb
        # Produce shift, scale, gate
        modulation = self.adaln_modulations[layer_idx](combined)
        shift, scale, gate = modulation.chunk(3, dim=-1)
        # AdaLN: normalize, scale+shift, gate
        h_norm = self.layer_norms[layer_idx](h)
        h = gate * (scale * h_norm + shift)
        return h
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 70,
        "end_line": 131,
        "content": _ADALN,
    },
]

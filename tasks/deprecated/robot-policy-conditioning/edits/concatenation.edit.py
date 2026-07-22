"""Concatenation conditioning baseline -- rigorous codebase edit ops.

Simplest conditioning: project the observation and concatenate it with the
noisy action at input. The base network then processes the combined
representation. Conditioning at intermediate layers simply passes features
through unchanged (all conditioning happens at input via the encode step,
which modifies the initial projection).

Since we can only modify condition_features and encode_condition, we implement
this by concatenating obs embedding into each layer's features via a
per-layer projection of [h; cond] -> h.
"""

_FILE = "CleanDiffuser/custom_conditioning.py"

_CONCATENATION = """\
class ConditioningModule(nn.Module):
    \"\"\"Concatenation conditioning: project [features; obs_emb] at each layer.

    At each layer, the observation embedding is concatenated with the
    intermediate features and projected back to hidden_dim. This is the
    simplest form of conditioning -- essentially expanding the input space
    with observation information at every layer.
    \"\"\"

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Encode observation to embedding
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Per-layer concat projection: [h; cond] -> h
        self.concat_projs = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(4)
        ])

    def encode_condition(self, obs):
        return self.obs_encoder(obs)

    def condition_features(self, h, t_emb, cond, layer_idx):
        combined = torch.cat([h, cond], dim=-1)
        return self.concat_projs[layer_idx](combined)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 70,
        "end_line": 131,
        "content": _CONCATENATION,
    },
]

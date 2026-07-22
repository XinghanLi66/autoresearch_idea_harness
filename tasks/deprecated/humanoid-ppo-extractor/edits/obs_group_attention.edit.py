"""Observation Group Attention baseline — rigorous codebase edit ops.

Partitions the observation vector into equal-sized groups, encodes
each group with a dedicated MLP, then fuses via multi-head
self-attention. Inspired by structured observation processing in
HumanoidBench (Sferrazza et al., 2024) and DexPBT.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "humanoid-bench/ppo/train_custom.py"

# ── 1. Replace feature extractor class (lines 20-38) ────────────────────

_OBS_GROUP_ATTN_CLASS = """\
class CustomFeatureExtractor(BaseFeaturesExtractor):
    \"\"\"Observation group attention: split obs into groups, encode, attend, fuse.\"\"\"
    def __init__(self, observation_space, features_dim=128, n_groups=8, d_group=32, n_heads=4):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.n_groups = n_groups
        # Compute group sizes (last group absorbs remainder)
        base_size = n_input // n_groups
        self.group_sizes = [base_size] * (n_groups - 1) + [n_input - base_size * (n_groups - 1)]
        # Per-group encoders
        self.group_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(gs, d_group), nn.ReLU()) for gs in self.group_sizes
        ])
        # Self-attention over group tokens
        self.attn = nn.MultiheadAttention(embed_dim=d_group, num_heads=n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_group)
        # Output projection
        self.output_proj = nn.Linear(n_groups * d_group, features_dim)

    def forward(self, observations):
        # Split obs into groups
        groups = torch.split(observations, self.group_sizes, dim=-1)
        # Encode each group: (batch, n_groups, d_group)
        tokens = torch.stack([enc(g) for enc, g in zip(self.group_encoders, groups)], dim=1)
        # Self-attention
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)
        # Flatten and project
        fused = tokens.reshape(tokens.size(0), -1)
        return self.output_proj(fused)

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 38,
        "content": _OBS_GROUP_ATTN_CLASS,
    },
]

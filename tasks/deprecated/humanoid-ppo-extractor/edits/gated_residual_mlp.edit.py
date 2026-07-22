"""Gated Residual MLP baseline — rigorous codebase edit ops.

Uses GRU-style gating in residual blocks for adaptive information
flow, inspired by Highway Networks and gated architectures in
recent RL feature extractors.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "humanoid-bench/ppo/train_custom.py"

# ── 1. Replace feature extractor class (lines 20-38) ────────────────────

_GATED_RESIDUAL_CLASS = """\
class _GatedResBlock(nn.Module):
    \"\"\"Residual block with sigmoid gating for adaptive skip connections.\"\"\"
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.gate_fc = nn.Linear(dim, dim)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        gate = torch.sigmoid(self.gate_fc(x))
        return gate * h + (1 - gate) * x


class CustomFeatureExtractor(BaseFeaturesExtractor):
    \"\"\"Gated residual MLP with adaptive skip connections.\"\"\"
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.input_proj = nn.Linear(n_input, 256)
        self.block1 = _GatedResBlock(256)
        self.block2 = _GatedResBlock(256)
        self.output_proj = nn.Linear(256, features_dim)

    def forward(self, observations):
        x = torch.relu(self.input_proj(observations))
        x = self.block1(x)
        x = self.block2(x)
        return self.output_proj(x)

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 38,
        "content": _GATED_RESIDUAL_CLASS,
    },
]

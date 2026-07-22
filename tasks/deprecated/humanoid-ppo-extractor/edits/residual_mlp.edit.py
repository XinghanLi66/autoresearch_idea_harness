"""Residual MLP baseline — rigorous codebase edit ops.

Replaces the default 2-layer MLP with a residual MLP that uses
skip connections for better gradient flow.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "humanoid-bench/ppo/train_custom.py"

# ── 1. Replace feature extractor class (lines 20-38) ────────────────────

_RESIDUAL_MLP_CLASS = """\
class CustomFeatureExtractor(BaseFeaturesExtractor):
    \"\"\"Residual MLP feature extractor with skip connections.\"\"\"
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.input_proj = nn.Linear(n_input, 256)
        self.block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.output_proj = nn.Linear(256, features_dim)

    def forward(self, observations):
        x = torch.relu(self.input_proj(observations))
        x = torch.relu(x + self.block1(x))
        x = torch.relu(x + self.block2(x))
        return self.output_proj(x)

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 38,
        "content": _RESIDUAL_MLP_CLASS,
    },
]

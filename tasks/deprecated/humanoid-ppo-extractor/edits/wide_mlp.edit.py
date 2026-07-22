"""Wide MLP baseline — rigorous codebase edit ops.

Replaces the default 2-layer MLP (64 units) with a 3-layer wide MLP
(256 units per hidden layer, 256 output features).

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "humanoid-bench/ppo/train_custom.py"

# ── 1. Replace feature extractor class (lines 20-38) ────────────────────

_WIDE_MLP_CLASS = """\
class CustomFeatureExtractor(BaseFeaturesExtractor):
    \"\"\"Wide MLP feature extractor with 256-unit hidden layers.\"\"\"
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations):
        return self.net(observations)

"""

# Ordered bottom-to-top so line numbers remain stable across ops.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 20,
        "end_line": 38,
        "content": _WIDE_MLP_CLASS,
    },
]

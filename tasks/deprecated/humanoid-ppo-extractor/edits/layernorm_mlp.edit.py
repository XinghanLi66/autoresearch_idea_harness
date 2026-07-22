"""LayerNorm MLP baseline — rigorous codebase edit ops.

Replaces the default 2-layer MLP with a 3-layer MLP that uses
LayerNorm after each hidden layer for training stability.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "humanoid-bench/ppo/train_custom.py"

# ── 1. Replace feature extractor class (lines 20-38) ────────────────────

_LAYERNORM_MLP_CLASS = """\
class CustomFeatureExtractor(BaseFeaturesExtractor):
    \"\"\"MLP feature extractor with LayerNorm for stable training.\"\"\"
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, features_dim),
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
        "content": _LAYERNORM_MLP_CLASS,
    },
]

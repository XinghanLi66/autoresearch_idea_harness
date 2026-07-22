"""Spectral Norm MLP baseline — rigorous codebase edit ops.

Applies spectral normalization to all linear layers to prevent
plasticity loss during long training runs (Lyle et al., 2023).
Uses ELU activations common in recent locomotion work.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "humanoid-bench/ppo/train_custom.py"

# ── 1. Replace feature extractor class (lines 20-38) ────────────────────

_SPECTRAL_NORM_CLASS = """\
class CustomFeatureExtractor(BaseFeaturesExtractor):
    \"\"\"MLP with spectral normalization to maintain plasticity.\"\"\"
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(n_input, 256)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(256, 256)),
            nn.ELU(),
            nn.utils.spectral_norm(nn.Linear(256, features_dim)),
            nn.ELU(),
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
        "content": _SPECTRAL_NORM_CLASS,
    },
]

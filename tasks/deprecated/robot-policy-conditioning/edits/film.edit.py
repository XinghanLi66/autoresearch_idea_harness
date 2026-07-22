"""FiLM (Feature-wise Linear Modulation) conditioning baseline.

The observation embedding produces per-layer scale (gamma) and shift (beta)
parameters via learned linear projections. At each layer:
    h = gamma_i(obs) * h + beta_i(obs)

This is the conditioning mechanism used in ChiUNet1d (cond_predict_scale=True).
It allows the observation to multiplicatively gate and additively shift features,
providing more expressive conditioning than simple addition.
"""

_FILE = "CleanDiffuser/custom_conditioning.py"

_FILM = """\
class ConditioningModule(nn.Module):
    \"\"\"FiLM conditioning: per-layer scale and shift from observation.

    The observation embedding is used to produce per-layer affine
    transformation parameters (gamma, beta). Features are modulated as:
        h = gamma * h + beta
    This follows Feature-wise Linear Modulation (Perez et al., 2018).
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
        # Per-layer FiLM generators: obs_emb -> (gamma, beta)
        self.film_generators = nn.ModuleList([
            nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim * 2))
            for _ in range(4)
        ])

    def encode_condition(self, obs):
        return self.obs_encoder(obs)

    def condition_features(self, h, t_emb, cond, layer_idx):
        film_params = self.film_generators[layer_idx](cond)
        gamma, beta = film_params.chunk(2, dim=-1)
        # FiLM: scale and shift
        return gamma * h + beta
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 70,
        "end_line": 131,
        "content": _FILM,
    },
]

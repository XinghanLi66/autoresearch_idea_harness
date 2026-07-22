"""Cross-attention conditioning baseline.

Action features attend to observation features via multi-head attention.
At each layer, the intermediate features h serve as queries, while the
observation embedding serves as keys and values. This allows each feature
dimension to selectively attend to relevant parts of the observation.
"""

_FILE = "CleanDiffuser/custom_conditioning.py"

_CROSS_ATTENTION = """\
class ConditioningModule(nn.Module):
    \"\"\"Cross-attention conditioning: features attend to observation tokens.

    The observation is encoded into a sequence of tokens (via chunking the
    embedding into multiple tokens). At each layer, multi-head cross-attention
    is applied where action features are queries and obs tokens are keys/values.
    \"\"\"

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.n_tokens = 4  # number of observation tokens
        self.n_heads = 4

        # Encode observation into multiple tokens
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim * self.n_tokens),
            nn.GELU(),
            nn.Linear(hidden_dim * self.n_tokens, hidden_dim * self.n_tokens),
        )

        # Per-layer cross-attention
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, self.n_heads, batch_first=True)
            for _ in range(4)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(4)
        ])

    def encode_condition(self, obs):
        # (B, obs_dim) -> (B, n_tokens, hidden_dim)
        tokens = self.obs_encoder(obs)
        return tokens.view(obs.shape[0], self.n_tokens, self.hidden_dim)

    def condition_features(self, h, t_emb, cond, layer_idx):
        # h: (B, hidden_dim) -> (B, 1, hidden_dim) for attention
        h_seq = h.unsqueeze(1)
        # Cross-attention: query=h, key/value=obs_tokens
        attn_out, _ = self.cross_attns[layer_idx](h_seq, cond, cond)
        # Residual + LayerNorm
        h = self.layer_norms[layer_idx](h + attn_out.squeeze(1))
        return h
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 70,
        "end_line": 131,
        "content": _CROSS_ATTENTION,
    },
]

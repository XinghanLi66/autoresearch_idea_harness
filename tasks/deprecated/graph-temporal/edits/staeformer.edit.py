"""STAEformer baseline -- Spatio-temporal adaptive embedding with spatial attention.

STAEformer (CIKM'23): Spatio-Temporal Adaptive Embedding Makes Vanilla
Transformer SOTA for Traffic Forecasting. The spatial component uses
adaptive node embeddings combined with multi-head attention to capture
spatial dependencies without relying on a predefined graph structure.

Reference: Liu et al., "Spatio-Temporal Adaptive Embedding Makes Vanilla
Transformer SOTA for Traffic Forecasting", CIKM 2023.
"""

_FILE = "BasicTS/custom_graph_model.py"

_CONTENT = """\
class SpatialLayer(nn.Module):
    \"\"\"Spatial adaptive embedding + multi-head attention (STAEformer).

    Uses learnable spatial node embeddings to compute multi-head attention
    among all nodes. Combines graph structure bias with adaptive attention.
    This captures both the fixed spatial topology and dynamic spatial patterns.
    \"\"\"

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 num_heads: int = 4, embed_dim: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.embed_dim = embed_dim
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # Learnable spatial embeddings (lazily initialized on first forward)
        self.spatial_embed = None

        # QKV projections incorporating spatial embeddings
        self.q_proj = nn.Linear(in_dim + embed_dim, out_dim)
        self.k_proj = nn.Linear(in_dim + embed_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        if self.spatial_embed is None:
            self.spatial_embed = nn.Parameter(
                torch.randn(N, self.embed_dim, device=x.device) * 0.02)

        # Augment features with spatial embeddings
        se = self.spatial_embed.unsqueeze(0).expand(B, -1, -1)  # [B, N, E]
        x_aug = torch.cat([x, se], dim=-1)  # [B, N, D+E]

        # Multi-head attention
        Q = self.q_proj(x_aug).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_aug).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores with optional graph structure bias
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Add graph bias: log(adj + eps) to encourage attention along edges
        graph_bias = torch.log(adj.clamp(min=1e-6)).unsqueeze(0).unsqueeze(0)
        attn = attn + 0.1 * graph_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, -1)
        out = self.out_proj(out)
        out = self.norm(out)
        return self.dropout(out)
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 94, "end_line": 127, "content": _CONTENT},
]

"""ASTGCN baseline -- Spatial attention graph convolution.

ASTGCN (AAAI'19): Attention-Based Spatial-Temporal Graph Convolutional Networks.
The spatial attention mechanism computes dynamic attention scores between
node pairs, weighting the adjacency for each sample in the batch.

Reference: Guo et al., "Attention Based Spatial-Temporal Graph Convolutional
Networks for Traffic Flow Forecasting", AAAI 2019.
"""

_FILE = "BasicTS/custom_graph_model.py"

_CONTENT = """\
class SpatialLayer(nn.Module):
    \"\"\"Spatial attention graph convolution (ASTGCN).

    Computes dynamic spatial attention: S = V_s * sigmoid(X * W_1 * W_2 * X^T + b_s).
    Combines attention with the adjacency matrix for message passing.
    \"\"\"

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(in_dim, in_dim))
        self.W2 = nn.Parameter(torch.empty(in_dim, in_dim))
        self.V_s = nn.Parameter(torch.empty(in_dim, in_dim))
        self.b_s = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.V_s)

        # Chebyshev-like graph conv with attention-weighted adjacency
        self.weight = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Spatial attention: S = softmax(sigmoid(x @ W1) @ W2 @ x^T)
        # x: [B, N, D]
        lhs = torch.sigmoid(x @ self.W1)     # [B, N, D]
        rhs = (x @ self.W2).transpose(1, 2)  # [B, D, N]
        S = (lhs @ self.V_s) @ rhs           # [B, N, N]
        S = S + self.b_s

        # Combine with fixed adjacency
        S_attn = F.softmax(S * adj.unsqueeze(0), dim=-1)  # mask by adjacency

        # Message passing with attention
        h = torch.matmul(S_attn, x)    # [B, N, D]
        out = self.weight(h)
        return self.dropout(F.relu(out))
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 94, "end_line": 127, "content": _CONTENT},
]

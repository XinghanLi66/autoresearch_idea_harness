"""DCRNN baseline -- Diffusion convolution.

DCRNN (ICLR'18): Diffusion Convolutional Recurrent Neural Network.
Models traffic flow as a diffusion process on a directed graph.
Uses K-step random walk diffusion with forward and backward transition
matrices: sum_k (theta_k * (D_o^{-1} A)^k + theta_k' * (D_i^{-1} A^T)^k) * X.

Reference: Li et al., "Diffusion Convolutional Recurrent Neural Network:
Data-Driven Traffic Forecasting", ICLR 2018.
"""

_FILE = "BasicTS/custom_graph_model.py"

_CONTENT = """\
class SpatialLayer(nn.Module):
    \"\"\"Bidirectional diffusion convolution (DCRNN).

    Performs K-step random walk diffusion using forward (D_o^{-1}A)
    and backward (D_i^{-1}A^T) transition matrices.
    \"\"\"

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, K: int = 3):
        super().__init__()
        self.K = K
        # Forward and backward diffusion weights
        self.weight_fwd = nn.Linear(in_dim * K, out_dim)
        self.weight_bwd = nn.Linear(in_dim * K, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

    def _diffusion_matrix(self, adj, direction="forward"):
        \"\"\"Compute row-normalized transition matrix.\"\"\"
        if direction == "backward":
            adj = adj.t()
        deg = adj.sum(dim=1).clamp(min=1e-8)
        return adj / deg.unsqueeze(1)

    def _k_hop_diffusion(self, x, trans, K):
        \"\"\"Compute [x, T*x, T^2*x, ..., T^{K-1}*x] and concatenate.\"\"\"
        supports = [x]
        h = x
        for _ in range(K - 1):
            h = torch.matmul(trans.unsqueeze(0), h)  # [B, N, D]
            supports.append(h)
        return torch.cat(supports, dim=-1)  # [B, N, K*D]

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Forward diffusion
        T_fwd = self._diffusion_matrix(adj, "forward")
        h_fwd = self._k_hop_diffusion(x, T_fwd, self.K)
        out_fwd = self.weight_fwd(h_fwd)

        # Backward diffusion
        T_bwd = self._diffusion_matrix(adj, "backward")
        h_bwd = self._k_hop_diffusion(x, T_bwd, self.K)
        out_bwd = self.weight_bwd(h_bwd)

        out = out_fwd + out_bwd + self.bias
        return self.dropout(F.relu(out))
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 94, "end_line": 127, "content": _CONTENT},
]

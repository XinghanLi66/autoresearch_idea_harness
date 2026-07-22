"""GWNet baseline -- Adaptive diffusion convolution.

Graph WaveNet (IJCAI'19): Combines diffusion convolution on the fixed
adjacency with an adaptive (learnable) adjacency matrix. The adaptive
graph is parameterized as softmax(E1 * E2^T) where E1, E2 are learnable
node embeddings.

Reference: Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph
Modeling", IJCAI 2019.
"""

_FILE = "BasicTS/custom_graph_model.py"

_CONTENT = """\
class SpatialLayer(nn.Module):
    \"\"\"Adaptive diffusion convolution (Graph WaveNet).

    Combines K-hop diffusion on both fixed and adaptive adjacency.
    Adaptive adjacency: softmax(E1 @ E2^T) with learnable node embeddings.
    \"\"\"

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 K: int = 3, embed_dim: int = 10):
        super().__init__()
        self.K = K
        self.embed_dim = embed_dim
        # Adaptive graph parameters (lazily initialized on first forward)
        self.E1 = None
        self.E2 = None

        # Diffusion weights: 2 directions * (fixed + adaptive)
        total_in = in_dim * K * 2  # fwd + bwd, each K hops
        self.fc = nn.Linear(total_in, out_dim)
        self.dropout = nn.Dropout(dropout)

    def _init_adaptive_params(self, num_nodes, device):
        self.E1 = nn.Parameter(torch.randn(num_nodes, self.embed_dim, device=device) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, self.embed_dim, device=device) * 0.1)

    def _k_hop(self, x, trans, K):
        supports = [x]
        h = x
        for _ in range(K - 1):
            h = torch.matmul(trans.unsqueeze(0), h)
            supports.append(h)
        return torch.cat(supports, dim=-1)

    def _row_normalize(self, adj):
        deg = adj.sum(dim=1).clamp(min=1e-8)
        return adj / deg.unsqueeze(1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if self.E1 is None:
            self._init_adaptive_params(adj.size(0), adj.device)
        # Adaptive adjacency
        adp = F.softmax(F.relu(self.E1 @ self.E2.t()), dim=1)

        # Combined adjacency: fixed + adaptive
        combined = adj + adp

        T_fwd = self._row_normalize(combined)
        T_bwd = self._row_normalize(combined.t())

        h_fwd = self._k_hop(x, T_fwd, self.K)
        h_bwd = self._k_hop(x, T_bwd, self.K)

        h = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, N, 2*K*D]
        out = self.fc(h)
        return self.dropout(F.relu(out))
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 94, "end_line": 127, "content": _CONTENT},
]

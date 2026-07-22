"""MTGNN baseline -- Mix-hop propagation with graph learning.

MTGNN (KDD'20): Connecting the Dots: Multivariate Time Series Forecasting
with Graph Neural Networks. Uses a graph learning layer that constructs
an adaptive adjacency matrix, combined with mix-hop propagation that
aggregates information across multiple neighborhood sizes.

Reference: Wu et al., "Connecting the Dots: Multivariate Time Series
Forecasting with Graph Neural Networks", KDD 2020.
"""

_FILE = "BasicTS/custom_graph_model.py"

_CONTENT = """\
class SpatialLayer(nn.Module):
    \"\"\"Mix-hop propagation with adaptive graph learning (MTGNN).

    Graph learning: uni-directional node embeddings with top-k sparsification.
    Mix-hop propagation: concatenates features from 0-hop to K-hop neighbors,
    then projects back. Combines fixed and learned graphs.
    \"\"\"

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0,
                 K: int = 3, embed_dim: int = 10,
                 top_k: int = 20, alpha: float = 0.3):
        super().__init__()
        self.K = K
        self.alpha = alpha  # balance fixed vs adaptive graph
        self.top_k = top_k
        self.embed_dim = embed_dim

        # Graph learning parameters (lazily initialized on first forward)
        self.E1 = None
        self.E2 = None

        # Mix-hop propagation: projects concatenated multi-hop features
        self.fc_in = nn.Linear(in_dim, out_dim)
        self.fc_out = nn.Linear(out_dim * (K + 1), out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def _init_adaptive_params(self, num_nodes, device):
        self.E1 = nn.Parameter(torch.randn(num_nodes, self.embed_dim, device=device) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, self.embed_dim, device=device) * 0.1)

    def _learn_graph(self):
        \"\"\"Uni-directional graph learning with top-k sparsification.\"\"\"
        # Compute pairwise similarity
        sim = torch.mm(self.E1, self.E2.t())
        sim = F.relu(sim)

        # Top-k sparsification per row
        N = sim.size(0)
        k = min(self.top_k, N)
        topk_vals, topk_idx = sim.topk(k, dim=1)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, topk_idx, 1.0)
        sim = sim * mask

        # Row normalization
        deg = sim.sum(dim=1).clamp(min=1e-8)
        return sim / deg.unsqueeze(1)

    def _mix_hop(self, x, adj):
        \"\"\"Mix-hop propagation: collect features from 0 to K hops.\"\"\"
        hops = [x]
        h = x
        for _ in range(self.K):
            h = torch.matmul(adj.unsqueeze(0), h)
            hops.append(h)
        return torch.cat(hops, dim=-1)  # [B, N, (K+1)*D]

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if self.E1 is None:
            self._init_adaptive_params(adj.size(0), adj.device)

        x = self.fc_in(x)

        # Adaptive graph
        adp = self._learn_graph()

        # Row normalize fixed graph
        deg = adj.sum(dim=1).clamp(min=1e-8)
        adj_norm = adj / deg.unsqueeze(1)

        # Combined adjacency
        combined = self.alpha * adj_norm + (1 - self.alpha) * adp

        # Mix-hop propagation
        h = self._mix_hop(x, combined)  # [B, N, (K+1)*out_dim]
        out = self.fc_out(h)
        out = self.norm(out)
        return self.dropout(F.relu(out))
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 94, "end_line": 127, "content": _CONTENT},
]

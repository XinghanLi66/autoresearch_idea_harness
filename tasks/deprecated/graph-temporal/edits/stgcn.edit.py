"""STGCN baseline -- Chebyshev graph convolution.

STGCN (IJCAI'18): Spatio-Temporal Graph Convolutional Networks.
Uses Chebyshev polynomial approximation for spectral graph convolution.
K-order Chebyshev polynomials of the scaled Laplacian approximate
localized graph filters.

Reference: Yan et al., "Spatio-Temporal Graph Convolutional Networks:
A Deep Learning Framework for Traffic Forecasting", IJCAI 2018.
"""

_FILE = "BasicTS/custom_graph_model.py"

_CONTENT = """\
class SpatialLayer(nn.Module):
    \"\"\"Chebyshev spectral graph convolution (STGCN).

    Approximates graph convolution using K-order Chebyshev polynomials
    of the scaled graph Laplacian: T_0(L)=I, T_1(L)=L, T_k(L)=2*L*T_{k-1}-T_{k-2}.
    \"\"\"

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, K: int = 3):
        super().__init__()
        self.K = K
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, out_dim))
            for _ in range(K)
        ])
        self.bias = nn.Parameter(torch.zeros(out_dim))
        for w in self.weights:
            nn.init.xavier_uniform_(w)
        self.dropout = nn.Dropout(dropout)

    def _scaled_laplacian(self, adj):
        \"\"\"Compute scaled Laplacian: L_scaled = 2*L/lambda_max - I.\"\"\"
        N = adj.size(0)
        deg = adj.sum(dim=1).clamp(min=1e-8)
        deg_inv_sqrt = deg.pow(-0.5)
        L = torch.eye(N, device=adj.device) - deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        # Approximate lambda_max ~ 2 for normalized Laplacian
        L_scaled = L - torch.eye(N, device=adj.device)
        return L_scaled

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        L = self._scaled_laplacian(adj)

        # Chebyshev polynomials: T_0(L)*x, T_1(L)*x, ..., T_{K-1}(L)*x
        T_prev = x                           # T_0 = I
        out = torch.matmul(T_prev, self.weights[0])

        if self.K > 1:
            T_curr = torch.matmul(L.unsqueeze(0), x)  # T_1 = L
            out = out + torch.matmul(T_curr, self.weights[1])

            for k in range(2, self.K):
                T_next = 2 * torch.matmul(L.unsqueeze(0), T_curr) - T_prev
                out = out + torch.matmul(T_next, self.weights[k])
                T_prev, T_curr = T_curr, T_next

        out = out + self.bias
        return self.dropout(F.relu(out))
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 94, "end_line": 127, "content": _CONTENT},
]

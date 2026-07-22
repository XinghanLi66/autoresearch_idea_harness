# Graph Signal Propagation: Spectral/Spatial Graph Filters

## Research Question
Design a novel graph signal propagation filter for node feature aggregation in graph neural networks. The filter should effectively handle both homophilic and heterophilic graphs.

## Background
Graph Neural Networks propagate node features through graph structure using graph filters. The choice of filter is critical: simple low-pass filters (like GCN's first-order approximation) work well on homophilic graphs (where connected nodes share labels) but fail on heterophilic graphs (where connected nodes often differ). Modern spectral methods learn polynomial filters in various bases:

- **Monomial basis** (GPRGNN): h(A) = sum_k gamma_k A^k -- simple but can be numerically unstable
- **Bernstein basis** (BernNet): non-negative, excellent controllability, but O(K^2) complexity
- **Chebyshev interpolation** (ChebNetII): avoids Runge phenomenon, O(K) complexity
- **Jacobi polynomials** (JacobiConv): orthogonal, fast convergence, generalizes Chebyshev/Legendre

Key design axes include: polynomial basis choice, coefficient initialization and constraints, normalization (GCN vs Laplacian), and interaction with the MLP encoder.

## Task
Modify the `CustomProp` (propagation layer) and `CustomFilter` (full model) classes in `custom_filter.py`. The propagation layer defines how node features are filtered across the graph; the model wraps it with an MLP encoder and output head.

## Interface
```python
class CustomProp(MessagePassing):
    def __init__(self, K, alpha=0.1, **kwargs):
        # K: polynomial order, alpha: teleport probability
        ...
    def forward(self, x, edge_index, edge_weight=None):
        # x: [num_nodes, channels], edge_index: [2, num_edges]
        # Returns: filtered features [num_nodes, channels]
        ...

class CustomFilter(nn.Module):
    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        ...
    def forward(self, data):
        # data: PyG Data object with data.x, data.edge_index
        # Returns: log_softmax predictions [num_nodes, num_classes]
        ...
```

## Available Utilities
- `gcn_norm(edge_index)` -- GCN normalization D^{-1/2}AD^{-1/2}
- `get_laplacian(edge_index, normalization='sym')` -- symmetric normalized Laplacian L = I - D^{-1/2}AD^{-1/2}
- `add_self_loops(edge_index, edge_weight, fill_value)` -- add self-loops
- `self.propagate(edge_index, x=x, norm=norm)` -- single-step message passing
- `cheby(i, x)` -- evaluate Chebyshev polynomial T_i(x)
- `comb(n, k)` -- binomial coefficient (from scipy)
- Constants: `K`, `ALPHA`, `HIDDEN`, `DROPOUT`, `DPRATE`

## Evaluation
Evaluated on 4 benchmark datasets spanning homophilic and heterophilic graphs:
- **Cora** (2,708 nodes, 7 classes, homophilic) -- citation network
- **CiteSeer** (3,327 nodes, 6 classes, homophilic) -- citation network
- **Texas** (183 nodes, 5 classes, heterophilic) -- WebKB webpage network
- **Cornell** (183 nodes, 5 classes, heterophilic) -- WebKB webpage network

Each dataset runs 10 random splits (60/20/20 train/val/test) with early stopping.
Metric: mean test accuracy over 10 runs (higher is better).


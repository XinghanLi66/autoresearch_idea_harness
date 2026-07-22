# Graph Neural Network: Node Classification Message Passing

## Research Question
Design a novel message passing mechanism for graph neural networks that improves node classification performance across citation network benchmarks.

## Background
Graph Neural Networks (GNNs) learn node representations by iteratively aggregating information from neighboring nodes through message passing. The core design choices are:
- **Message construction**: how to compute messages from source to target nodes (e.g., linear transform, attention-weighted, edge-conditioned)
- **Aggregation**: how to combine incoming messages (e.g., sum, mean, max, attention-weighted)
- **Update**: how to integrate aggregated messages with the node's own representation (e.g., residual, gated, concatenation)

Classic approaches include GCN (symmetric normalization), GAT (attention-based weighting), and GraphSAGE (mean aggregation with self/neighbor separation). Recent advances like Graph Transformers (GPS) combine local message passing with global self-attention, while methods like NAGphormer use multi-hop tokenization with Transformer encoders.

## Task
Modify the `CustomMessagePassingLayer` class and `CustomGNN` model in `custom_nodecls.py` to implement a novel message passing mechanism. Your implementation must work within PyTorch Geometric's `MessagePassing` framework.

## Model Interface
```python
class CustomMessagePassingLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Define learnable parameters and layers
        ...

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # x: [num_nodes, in_channels], edge_index: [2, num_edges]
        # Returns: [num_nodes, out_channels]
        ...

    def message(self, x_j: Tensor, ...) -> Tensor:
        # Define per-edge message computation
        ...

class CustomGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        ...

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # Returns: [num_nodes, out_channels] (logits)
        ...
```

## Available PyG Utilities
- `MessagePassing` base class: `self.propagate(edge_index, ...)` orchestrates message/aggregate/update
- `add_self_loops(edge_index)`: add self-loop edges
- `degree(index, num_nodes)`: compute node degrees
- `softmax(src, index)`: sparse softmax over edges
- Convolution layers for reference: `GCNConv`, `GATConv`, `SAGEConv` (imported but read-only)

## Evaluation
Trained and evaluated on three citation network benchmarks (semi-supervised node classification with standard Planetoid splits):
- **Cora**: 2,708 nodes, 5,429 edges, 7 classes, 1,433 features
- **CiteSeer**: 3,327 nodes, 4,732 edges, 6 classes, 3,703 features
- **PubMed**: 19,717 nodes, 44,338 edges, 3 classes, 500 features

Metrics: test accuracy and macro F1 score (higher is better). Training: 200 epochs with early stopping (patience=50), Adam optimizer, lr=0.01, weight_decay=5e-4.


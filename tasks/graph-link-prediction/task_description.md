# Task: Graph Link Prediction

## Research Question
Design a novel link prediction method for graphs. The goal is to learn an encoder that maps nodes to embeddings and a decoder that scores candidate edges, such that the model accurately predicts missing or future links across diverse graph types.

## Background
Link prediction is a fundamental graph learning task: given a partially observed graph, predict which unobserved edges are likely to exist. It has applications in social network analysis (friend recommendation), citation networks (paper recommendation), knowledge graph completion, and biological interaction prediction.

Classical approaches include:
- **GCN + dot product**: GCN encodes nodes; dot product of embeddings scores edges. Simple but often competitive.
- **VGAE (Variational Graph Auto-Encoder)**: Probabilistic encoder with KL regularization for robust link prediction.
- **Node2Vec**: Random-walk-based embedding with biased walks capturing structural roles and communities.

Recent SOTA methods explore richer structural information:
- **SEAL**: Extracts k-hop enclosing subgraphs per edge, uses DRNL labeling trick + GNN for edge classification.
- **Neo-GNN**: Learns neighborhood overlap features from adjacency matrix to augment GNN predictions.
- **BUDDY**: Subgraph sketching with MinHash/HyperLogLog features for scalable structural information.

## What to Implement
Implement the `LinkPredictor` class in `custom_linkpred.py`. You must implement:
1. `__init__(self, in_channels, hidden_channels, num_layers, dropout)`: Set up your model.
2. `encode(self, x, edge_index) -> Tensor [N, hidden_channels]`: Encode nodes into embeddings.
3. `decode(self, z_src, z_dst) -> Tensor [num_edges]`: Score candidate edges from source/dest embeddings.
4. `forward(self, x, edge_index, edge_label_index) -> Tensor [num_edges]`: Full forward pass.

## Input Format
- `x`: Node features `[N, in_channels]` — feature dimension varies by dataset.
- `edge_index`: Training graph edges `[2, E_train]` in COO format (undirected).
- `edge_label_index`: Candidate edges to score `[2, num_candidates]`.

## Available PyG Modules (pre-installed)
You can use any PyTorch Geometric module: `GCNConv`, `SAGEConv`, `GATConv`, `GINConv`, `GraphConv`, `MessagePassing`, global pooling, etc. Also available: `torch_geometric.utils` (negative_sampling, to_undirected, degree, etc.), `torch_geometric.nn`, and `torch_geometric.transforms`.

## Evaluation
The model is tested on 3 benchmark datasets:

### Citation Networks (metric: AUC, MRR, Hits@20 — higher is better):
- **Cora**: 2,708 nodes, 10,556 edges, 1,433 features, 7 classes. 85/5/10 link split.
- **CiteSeer**: 3,327 nodes, 9,104 edges, 3,703 features, 6 classes. 85/5/10 link split.

### Collaboration Network (metric: Hits@50, MRR — higher is better):
- **ogbl-collab**: 235,868 nodes, 1,285,465 edges, 128 features. Official OGB split.

## Editable Region
Lines 122-196 of `custom_linkpred.py` are editable (between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers). You may define the `LinkPredictor` class and any helper functions/classes within this region. The class must conform to the interface above.

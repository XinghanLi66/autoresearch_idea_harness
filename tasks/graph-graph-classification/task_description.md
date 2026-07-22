# Graph-Level Readout/Pooling for Graph Classification

## Research Question
Design a novel graph-level readout (pooling) mechanism that effectively aggregates node representations into a graph-level embedding for graph classification, improving accuracy and generalization across diverse molecular and biological graph datasets.

## Background
Graph classification requires mapping a variable-size graph to a fixed-size vector for downstream prediction. The standard approach uses simple permutation-invariant operations (sum, mean, max) over node embeddings, but these discard structural information and treat all nodes equally. Recent advances explore:

- **Sum/Mean/Max Readout**: Simple aggregation; GIN (Xu et al., 2019) shows sum readout is most expressive among basic operations.
- **SortPooling** (Zhang et al., 2018): Sorts nodes by structural role via WL colors and applies 1D convolution, capturing ordering information.
- **Set2Set** (Vinyals et al., 2016): LSTM-based attention over node set, learning to attend to relevant nodes.
- **SAGPool** (Lee et al., 2019): Self-attention scores for hierarchical node selection, retaining top-k informative nodes.
- **DiffPool** (Ying et al., 2018): Differentiable soft cluster assignment for hierarchical graph coarsening.
- **GMT** (Baek et al., 2021): Graph Multiset Transformer using multi-head attention for structure-aware readout.

There is substantial room to improve graph readout by combining attention mechanisms, multi-scale aggregation, structural encodings, or learned pooling strategies.

## What You Can Modify
The `GraphReadout` class (lines 41-80) in `custom_graph_cls.py`. This class receives node embeddings from a fixed GIN backbone and must produce graph-level embeddings.

You can modify:
- The aggregation function (sum, mean, max, attention, learned weights)
- Whether to use hierarchical coarsening (cluster, pool, repeat)
- How to combine multi-layer GNN outputs (jumping knowledge, concatenation, attention)
- Self-attention or cross-attention mechanisms over nodes
- Structural encoding or positional information in readout
- Any combination of the above

Constraints:
- Input: `x` [N_total, hidden_dim], `edge_index` [2, E_total], `batch` [N_total], `layer_outputs` list of [N_total, hidden_dim]
- Output: [B, output_dim] tensor (set `self.output_dim` in `__init__`)
- Must handle variable graph sizes within a batch
- Must be permutation equivariant/invariant as appropriate
- All imports available: `torch`, `torch.nn`, `torch.nn.functional`, `torch_geometric.nn`, `torch_geometric.utils`

## Evaluation
- **Metrics**: Test accuracy (%, higher is better) and macro F1 (%, higher is better), averaged over 10-fold stratified cross-validation
- **Datasets**:
  - MUTAG (188 graphs, 2 classes, molecular mutagenicity)
  - PROTEINS (1113 graphs, 2 classes, protein enzyme classification)
  - NCI1 (4110 graphs, 2 classes, chemical compound activity)
- **GNN backbone**: 5-layer GIN (hidden_dim=64), fixed
- **Training**: Adam (lr=0.01), cosine annealing, 350 epochs per fold
- **Evaluation**: 10-fold stratified cross-validation, report mean accuracy and macro F1


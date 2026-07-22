# Task: Protein Structure Representation Learning

## Research Question
Design a novel geometric GNN encoder for learning protein structure representations from 3D alpha-carbon coordinates. The encoder must capture both local geometric patterns (bond angles, dihedral angles) and global structural motifs to produce informative per-residue and per-protein embeddings.

## Background
Protein function is determined by 3D structure. Geometric GNNs that operate on protein structure graphs (nodes = residues at alpha-carbon positions, edges = spatial/sequential neighbors) have emerged as powerful tools for learning protein representations. Key challenges include:
- **Geometric awareness**: The encoder should leverage 3D spatial information (distances, angles, orientations) beyond simple adjacency.
- **Equivariance/invariance**: Representations should be invariant to rigid body transformations (rotations, translations) of the protein.
- **Multi-scale structure**: Proteins exhibit hierarchical structure (secondary structure elements, domains, global fold) that the encoder should capture.

Existing approaches include:
- **SchNet**: Uses continuous-filter convolutions with Gaussian radial basis function distance expansion. Invariant by design.
- **EGNN**: E(n) equivariant message passing that jointly updates node features and coordinates.
- **GearNet**: Geometry-Aware Relational Graph Neural Network with multiple edge types (sequential, spatial, k-nearest) and relational convolutions.

## What to Implement
Implement the `ProteinEncoder` class and any helper modules in `custom_protein_encoder.py`. You must implement:
1. `__init__(self, ...)`: Set up the encoder architecture. The input node features have dimension `SCALAR_NODE_DIM=28` (20-dim amino acid one-hot + 2-dim positional encoding + 6-dim pseudo-dihedral features).
2. `forward(self, pos, node_feat, batch) -> (node_emb, graph_emb)`: Encode the protein graph.
   - `pos`: (N, 3) alpha-carbon coordinates
   - `node_feat`: (N, 28) scalar node features (computed by the fixed `compute_node_features` function)
   - `batch`: (N,) batch assignment indices
   - Returns: `node_emb` (N, out_dim) per-node embeddings, `graph_emb` (B, out_dim) per-graph embeddings

## Evaluation
The encoder is evaluated on three protein function/structure prediction benchmarks:

### EC Number Prediction (384-class, multiclass)
- Predicts enzyme commission number from protein structure
- Metric: **accuracy** (top-1)

### GO Biological Process (1943-class, multilabel)
- Predicts Gene Ontology biological process annotations
- Metric: **f1_max** (maximum F1 across thresholds)

### Fold Classification (1195-class, multiclass)
- Predicts protein fold from the SCOPe/CATH hierarchy
- Metric: **accuracy** (top-1)

Higher is better for all metrics.

## Editable Region
Lines 107-234 of `custom_protein_encoder.py` (the section between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers). You may define any helper classes, layers, or functions within this region. The region must contain a `ProteinEncoder` class with the interface described above.

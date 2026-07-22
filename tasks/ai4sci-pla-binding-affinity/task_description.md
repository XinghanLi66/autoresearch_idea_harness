# Task: Protein-Ligand Binding Affinity Prediction

## Research Question
Design a GNN architecture that effectively models protein-ligand interactions to predict binding affinity (-logKd/Ki) from 3D structural data. The goal is to learn representations that capture both intra-molecular structure and inter-molecular interactions between ligand and protein pocket.

## Background
Predicting the binding affinity between a drug molecule (ligand) and its target protein is a central task in structure-based drug design. Given a protein-ligand complex represented as a heterogeneous graph, the model must predict the binding strength (-logKd/Ki). Key challenges include:
- **Heterogeneous interactions**: The complex contains two types of molecules (ligand and pocket) with distinct chemistry, connected by non-covalent inter-molecular edges.
- **Geometric features**: Edge features encode rich 3D geometric information (angles, triangle areas, distances between neighboring atoms).
- **Bidirectional modeling**: Inter-molecular interactions can be modeled from ligand→pocket and pocket→ligand perspectives, potentially yielding different insights.

Existing approaches include:
- **EHIGN**: Heterogeneous graph convolution (CIG intra + NIG inter) with dual prediction (ligand→pocket and pocket→ligand) and bias correction.
- **GIGN**: Geometric interaction features with distance-based inter-molecular edges.
- **SchNet**: Continuous-filter convolution with RBF distance expansion on heterogeneous graphs.
- **EGNN**: E(n)-equivariant message passing treating distances as scalar edge features.

## What to Implement
Implement the `AffinityModel` class in `custom_pla.py`. You must implement:
1. `__init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim)`: Set up your model architecture.
2. `forward(self, batch: PLABatch) -> Tensor`: Return predictions of shape `[B]`.

## Batch Format (PLABatch)
```python
@dataclass
class PLABatch:
    # Ligand graph
    lig_x: Tensor              # [total_lig_atoms, 35] atom features
    lig_edge_index: Tensor     # [2, total_lig_edges] COO format
    lig_edge_attr: Tensor      # [total_lig_edges, 17] bond + geometric features
    lig_batch: Tensor          # [total_lig_atoms] graph assignment (0..B-1)

    # Pocket graph
    poc_x: Tensor              # [total_poc_atoms, 35] atom features
    poc_edge_index: Tensor     # [2, total_poc_edges] COO format
    poc_edge_attr: Tensor      # [total_poc_edges, 17] bond + geometric features
    poc_batch: Tensor          # [total_poc_atoms] graph assignment (0..B-1)

    # Inter-molecular edges (ligand -> pocket)
    l2p_edge_index: Tensor     # [2, total_l2p_edges] (src=ligand, dst=pocket)
    l2p_edge_attr: Tensor      # [total_l2p_edges, 11] geometric features

    # Inter-molecular edges (pocket -> ligand)
    p2l_edge_index: Tensor     # [2, total_p2l_edges] (src=pocket, dst=ligand)
    p2l_edge_attr: Tensor      # [total_p2l_edges, 11] geometric features

    # Metadata
    num_lig_atoms: List[int]   # per-complex ligand atom counts
    num_poc_atoms: List[int]   # per-complex pocket atom counts
    inter_batch: Tensor        # [total_l2p_edges] graph assignment for inter edges

    # Target
    labels: Tensor             # [B] binding affinity (-logKd/Ki)
```

## Atom Features (35 dimensions)
One-hot encodings of: element (C/N/O/S/F/P/Cl/Br/I/Unknown = 10), degree (0-6 = 7), implicit valence (0-6 = 7), hybridization (SP/SP2/SP3/SP3D/SP3D2 = 5), aromatic (1), total Hs (0-4 = 5).

## Intra-molecular Edge Features (17 dimensions)
Bond type (4) + conjugated (1) + in_ring (1) + geometric features (11): angle statistics (max/sum/mean), triangle area statistics (max/sum/mean), neighbor distance statistics (max/sum/mean), pairwise distances (L1, L2).

## Inter-molecular Edge Features (11 dimensions)
Geometric features only (same 11-dim encoding as intra-molecular geometric features): computed between ligand-pocket atom pairs within 5Å distance threshold.

## Evaluation
The model is trained on PDBbind and tested on three held-out benchmarks:
- **PDBbind 2013 core set** (107 complexes): CASF-2013 benchmark
- **PDBbind 2016 core set** (285 complexes): CASF-2016 benchmark
- **PDBbind 2019 holdout** (4366 complexes): Temporal split

Metrics: **RMSE** (lower is better), **Rp** / Pearson correlation (higher is better).

### Note on Baseline Ordering vs Original Papers
The baselines (EHIGN / GIGN / SchNet / EGNN) are paper-faithful re-implementations
on this task's data pipeline (PDBbind **v2020** general+refined → temporal/CASF
splits, with intra/inter graph features regenerated from raw PDB/SDF). The
original EHIGN and GIGN papers train on PDBbind v2016/v2019 with their own
preprocessing, so absolute numbers and the relative ordering between baselines
on this leaderboard may differ from the published numbers. In particular,
GIGN can match or slightly outperform EHIGN on the 2013 core set under our
3-seed evaluation (small ΔRMSE within ~0.05 RMSE / ~0.015 Rp std), while the
EHIGN paper reports EHIGN > GIGN. This is expected: it reflects (a) the
different train set (v2020) and (b) 3-seed noise on a 107-complex test set.
The baseline implementations are intentionally NOT tuned to recover the
published ordering; they are kept faithful to the published methods.

## Editable Region
Lines 101-191 of `custom_pla.py` are editable (between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers). You may define helper classes, layers, or functions within this region. The region must contain an `AffinityModel` class with the specified interface.

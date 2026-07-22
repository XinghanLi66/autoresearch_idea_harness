# Graph Generation Model Design

## Research Question
Design a novel generative model architecture for unconditional graph generation that produces realistic graph structures matching the statistical properties of a training distribution.

## Background
Graph generation is a fundamental problem in machine learning with applications in drug discovery, social network modeling, and materials science. The goal is to learn the distribution of a set of graphs and generate new graphs that are statistically indistinguishable from the training data.

Existing approaches span several paradigms:
- **Autoregressive**: GraphRNN (You et al., 2018) generates graphs node-by-node with RNNs, GRAN (Liao et al., 2019) uses graph attention for one-shot block generation
- **VAE-based**: GraphVAE (Simonovsky & Komodakis, 2018) encodes graphs into latent space and decodes adjacency matrices
- **Flow-based**: MoFlow (Zang & Wang, 2020) uses normalizing flows for invertible graph generation
- **Diffusion/Score-based**: GDSS (Jo et al., 2022) applies score-based SDEs to graph generation, DiGress (Vignac et al., 2023) uses discrete denoising diffusion

Evaluation uses Maximum Mean Discrepancy (MMD) between graph statistics (degree distribution, clustering coefficients, orbit counts) of generated and reference graphs. Lower MMD indicates the generated graphs better match the training distribution.

## What You Can Modify
The `GraphGenerator` class (lines 446-590) in `custom_graphgen.py`. This class must implement:

1. `__init__(self, max_nodes, **kwargs)`: Initialize model parameters and optimizer
2. `train_step(self, adj, node_counts) -> dict`: Perform one training step on a batch of adjacency matrices. Must return a dict containing at least `'loss'` (float).
3. `sample(self, n_samples, device) -> (adj, node_counts)`: Generate graphs. Returns:
   - `adj`: Tensor [n_samples, max_nodes, max_nodes] — binary symmetric adjacency matrices (no self-loops)
   - `node_counts`: Tensor [n_samples] — number of nodes per graph (minimum 2)

The input adjacency matrices are binary, symmetric, zero-diagonal, and padded to `max_nodes`.

You may define helper classes/functions within the editable region. The model's optimizer should be created inside `__init__` and updated inside `train_step`.

Available imports (in the FIXED section): `torch`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `numpy`, `math`.

## Evaluation
- **Metrics** (all lower is better):
  - `mmd_degree`: MMD of degree distributions
  - `mmd_clustering`: MMD of clustering coefficient distributions
  - `mmd_orbit`: MMD of 4-orbit count distributions
  - `mmd_avg`: Average of the three MMD metrics
- **Datasets**:
  - `community_small`: 100 synthetic 2-community graphs (12-20 nodes)
  - `ego_small`: 200 ego graphs from Citeseer (4-18 nodes)
  - `enzymes`: 587 protein structure graphs from BRENDA (10-125 nodes)
- **Training**: 500 epochs, batch size 32, single GPU (note: reduced from the 3000 epochs used in some published setups to fit the per-task compute budget; all baselines are trained with the same schedule for a fair comparison)
- **Seeds**: Multiple seeds for statistical reliability


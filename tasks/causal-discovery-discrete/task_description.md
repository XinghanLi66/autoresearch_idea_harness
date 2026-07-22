# Causal Discovery on Real-World Bayesian Network Datasets (bnlearn)

## Research Question
Design a causal discovery algorithm that recovers the **CPDAG** (Completed Partially Directed Acyclic Graph) from purely observational discrete data sampled from real-world Bayesian networks in the bnlearn repository.

## Background
The bnlearn repository (https://www.bnlearn.com/bnrepository/) hosts a collection of well-known Bayesian network benchmarks from diverse domains (medicine, biology, meteorology, insurance, agriculture, IT). Each network has a known ground-truth DAG with discrete variables and conditional probability tables. Given observational samples from these networks, the task is to recover the causal structure.

Under the faithfulness assumption, observational data can identify the Markov Equivalence Class (MEC) of the true DAG, represented by a CPDAG. The challenge lies in handling discrete data with varying cardinalities, network sizes (5–76 nodes), and edge densities.

## Task
Implement a causal discovery algorithm in `bench/custom_algorithm.py`. Your `run_causal_discovery(X)` function receives integer-encoded discrete observational data and must return the estimated CPDAG as a `causallearn.graph.GeneralGraph.GeneralGraph` object.

## Interface
```python
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
```

## Evaluation Scenarios

### Small Networks (<20 nodes)
| Label | Network | Nodes | Edges | Samples | Domain |
|-------|---------|-------|-------|---------|--------|
| Cancer | Cancer | 5 | 4 | 500 | Medical |
| Earthquake | Earthquake | 5 | 4 | 500 | Seismology |
| Survey | Survey | 6 | 6 | 500 | Social science |
| Asia | Asia | 8 | 8 | 1000 | Medical (lung diseases) |
| Sachs | Sachs | 11 | 17 | 1000 | Biology (protein signaling) |

### Medium Networks (20–50 nodes)
| Label | Network | Nodes | Edges | Samples | Domain |
|-------|---------|-------|-------|---------|--------|
| Child | Child | 20 | 25 | 2000 | Medical |
| Insurance | Insurance | 27 | 52 | 5000 | Automotive insurance |
| Water | Water | 32 | 66 | 5000 | Water treatment |
| Mildew | Mildew | 35 | 46 | 5000 | Agriculture (crop disease) |
| Alarm | Alarm | 37 | 46 | 5000 | Medical monitoring |
| Barley | Barley | 48 | 84 | 10000 | Agriculture |

### Large Networks (50–100 nodes)
| Label | Network | Nodes | Edges | Samples | Domain |
|-------|---------|-------|-------|---------|--------|
| Hailfinder | Hailfinder | 56 | 66 | 10000 | Meteorology |
| Hepar2 | Hepar2 | 70 | 123 | 10000 | Medical (liver disorders) |
| Win95pts | Win95pts | 76 | 112 | 10000 | IT (Windows troubleshooting) |

## Metrics
Metrics are computed between estimated CPDAG and ground-truth CPDAG (converted from the true DAG via `dag2cpdag`):
- **SHD** (Structural Hamming Distance): total edge errors (lower is better)
- **Adjacency Precision / Recall**: skeleton recovery quality
- **Arrow Precision / Recall**: edge orientation accuracy

## Baselines
- `pc`: Peter-Clark algorithm with chi-squared CI test (constraint-based)
- `ges`: Greedy Equivalence Search with BDeu score (score-based)
- `grasp`: Greedy Relaxations of the Sparsest Permutation with BDeu (permutation-based, SOTA)
- `boss`: Best Order Score Search with BDeu (permutation-based, SOTA)
- `hc`: Hill-Climbing search with BDeu score (score-based)

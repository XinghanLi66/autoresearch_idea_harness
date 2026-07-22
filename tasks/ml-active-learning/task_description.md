# Active Learning: Query Strategy Design

## Research Question
Design a novel pool-based active learning query strategy that outperforms existing methods (uncertainty sampling, entropy sampling, BADGE, BAIT, BALD) across diverse tabular classification datasets.

## Background
Active learning aims to minimize labeling cost by intelligently selecting which unlabeled samples to query for labels. In pool-based active learning, a query strategy selects batches of samples from an unlabeled pool to be labeled by an oracle, then the model is retrained. The goal is to achieve the highest possible accuracy with the fewest labeled samples.

Classic approaches include:
- **Uncertainty Sampling**: Select samples where the model is least confident (lowest max predicted probability)
- **Entropy Sampling**: Select samples with highest predictive entropy
- **Query By Committee**: Select samples with maximal disagreement among an ensemble

Modern approaches incorporate diversity and information-theoretic principles:
- **BADGE** (Ash et al., ICLR 2020): Uses gradient embeddings with k-means++ for diverse, uncertain batch selection
- **BAIT** (Ash et al., NeurIPS 2021): Optimizes Fisher information to select maximally informative batches
- **BALD** (Houlsby et al., 2011): Uses MC Dropout to estimate mutual information between predictions and parameters

## Task
Modify the `CustomSampling` class in `badge/query_strategies/custom_sampling.py` to implement a novel query strategy. The strategy must implement the `query(n)` method that returns `n` indices from the unlabeled pool.

## Interface
```python
class CustomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super().__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n) -> np.ndarray:
        # Must return n indices into self.X of unlabeled samples to label
        ...
```

**Available from the Strategy base class:**
- `self.X`: pool features (numpy array, shape `[n_pool, n_features]`)
- `self.Y`: pool labels (torch LongTensor, shape `[n_pool]`)
- `self.idxs_lb`: boolean mask of labeled samples
- `self.n_pool`: total pool size
- `self.predict_prob(X, Y)`: softmax probabilities `[len(X), n_classes]`
- `self.predict_prob_dropout_split(X, Y, n_drop)`: MC dropout probs `[n_drop, len(X), n_classes]`
- `self.get_embedding(X, Y)`: penultimate-layer embeddings `[len(X), emb_dim]`
- `self.get_grad_embedding(X, Y)`: gradient embeddings `[len(X), emb_dim * n_classes]`
- `self.get_exp_grad_embedding(X, Y)`: expected Fisher embeddings `[len(X), n_classes, emb_dim]`

## Evaluation
- **Datasets**: 3 OpenML tabular classification datasets (letter recognition, spambase, splice)
- **Protocol**: 20 rounds of batch active learning, evaluated after each round
- **Metrics**:
  - `accuracy`: Test accuracy at the end of 20 AL rounds (fixed label budget)
  - `auc`: Area under the learning curve (accuracy vs. number of labeled samples), measuring sample efficiency across all rounds
- Higher is better for both metrics.


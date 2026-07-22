**Core idea:**  
Introduce a hybrid objective combining a local term (kNN graph) and a global term (a spectral manifold prior) into a single optimization over 2D coordinates, with a soft alignment of the embedding to a learned linear subspace.

**Non-trivial crux:**  
Balancing the two terms with a single hyperparameter risks overfitting; instead, derive the subspace from PCA of the original data and then optimize only the nonlinear residual, decoupling the global structure from the local cost — this avoids a full nonlinear optimization on the entire space.

---

**Reasoning process:**

Start from a clear diagnosis of the failure mode. Existing nonlinear methods either sacrifice global structure for local neighborhoods (t-SNE, UMAP) or vice versa (Laplacian Eigenmaps, Isomap). The score metrics confirm this: t-SNE and UMAP have high continuity but low trustworthiness and low global kNN accuracy; PCA is fast but cannot represent nonlinear manifolds. So the crux is not computing a better neighborhood graph — it is that no single loss captures both local and global structure, and the nonlinear embedding is being optimized over the full 2D space, which is computationally fragile.

So the fix must be *structural*, not just tuning a parameter. A proven route is to separate a global component from a local residual. In linear PCA, you project onto the top principal components and the residual is noise — this is a form of the same principle. Generalizing it: let the global structure be a linear subspace (PCA), and let the local deviations be a nonlinear residual. That reduces the nonlinear optimization to a small residual, which should be easier and more stable than a full nonlinear search.

How to make this precise? Express the 2D embedding as:  
$$
Z = P + \epsilon
$$  
where $ P $ is the PCA projection onto the top two components (global structure) and $ \epsilon $ is a small residual to be optimized. Then the local cost is a kNN graph (as in UMAP) but applied to $ Z $. The global term is the PCA subspace itself — no explicit penalty is needed because the residual is already constrained to be small.

Why this works better:  
1. The PCA subspace already preserves the global variance and large-scale relationships, so the nonlinear residual only needs to fix local structure — this is a *low-uncertainty* optimization problem.  
2. The kNN graph enforces local neighborhood continuity while the subspace keeps the embedding globally aligned — the two complement rather than conflict.  
3. The residual's small scale means the optimizer is less likely to get stuck in local minima or blow up.

How to implement:  
1. Compute a PCA projection to 2D as the initial embedding $ P $.  
2. Define the nonlinear residual $ \epsilon $ as a small perturbation from this initial embedding (initialize it close to zero).  
3. Optimize only $ \epsilon $, with a kNN graph-based local loss (e.g., a UMAP-style symmetric KL divergence between distances in the graph and Euclidean distances in the embedding), using a small learning rate and early stopping.  
4. The final embedding is $ Z = P + \epsilon $.  

This is a *single* optimization stage, not two separate stages, which keeps the algorithm simple and fast. The PCA step is fast; the nonlinear residual is cheap because it is small and the graph is sparse. For reproducibility, seed both the PCA and the optimizer. The same objective can be extended to more than two dimensions by keeping more PCA components and optimizing the residual in higher dimensions.

The decision rule for the hyperparameter is to choose the smallest number of PCA components that capture most of the variance (e.g., 2–5 components) — the residual is then only a fine adjustment, and the PCA already handles the large-scale structure.
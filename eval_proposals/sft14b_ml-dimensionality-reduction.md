**Mocking:** World-class AI researcher

**Mocking:** I am a researcher with a strong preference for geometric intuitions and an aversion to heuristic tuning.

Let me start from what I actually want to preserve. PCA is linear, so it can only find global linear directions - it can't bend to follow a curved manifold. t-SNE and UMAP are nonlinear but they are *local* in spirit: they care a lot about preserving immediate neighborhood structure and less about the larger picture. My goal is *both* local and global structure. That's the tension I need to resolve, not a parameter to tune.

So what breaks if I naively do both? If I build a graph with both local and global links and then run a single embedding, the local links dominate because they are more numerous and dense. Global relationships get drowned out. I need a mechanism that *separately* represents local and global structure before merging them.

Here is the leap: split representation into two parallel graphs, each encoding one type of structure, and then combine them in a *weighted* embedding. This is not an obvious move. I could try a single graph with both local and global links, but that would mix the scales and the dense local links would overwhelm the sparse global ones. Separating them is the decision, and the crux is controlling the balance between them.

How do I make the two graphs? For the *local* graph, a k-nearest-neighbor graph (k=5-10) captures the neighborhood structure. For the *global* graph, I want a scale-invariant measure of whether two points are similar globally. A distance-based similarity like rbf-kernel similarity (Gaussian kernel) is scale-invariant and smooths the global structure, so that's my global similarity function.

Now, how do I embed both representations? I'll map each graph separately into its own low-dimensional space using spectral embedding, which naturally handles the graph structure. Then I merge the two low-dimensional embeddings with a weighted combination - the key is to let the *weight* of the global embedding be a parameter rather than a fixed value. This is the decision I'll test systematically: sweep the global weight from 0 to 1 and choose the value that maximizes the metrics.

Why is this non-trivial? It's not just a matter of picking a higher weight. The global component may be sparse and unstable at extreme weights. I need to validate the weight choice on the specific datasets rather than assume a universal value.

I'll structure the implementation as a clear pipeline: compute local and global graphs, embed each, then combine. The combination is the only place where the decision matters; the rest is standard and well-understood.

**Core idea:** Split data structure into two parallel graphs (local kNN and global rbf-similarity), embed each with spectral embedding, then merge them in a weighted combination to preserve both local neighborhoods and global relationships.
**Non-trivial crux:** The two graphs must be treated separately (not merged into one graph) and the balance between them is controlled by a tunable weight that must be validated per dataset rather than set arbitrarily.
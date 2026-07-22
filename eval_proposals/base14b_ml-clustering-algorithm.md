Core idea: Introduce a **density-aware adaptive distance metric** that scales Euclidean distances by the inverse square root of local density estimates, combined with a **hybrid spectral-HDBSCAN approach** that first constructs a graph using this metric and then applies hierarchical density-based clustering on the graph's embedding.

Non-trivial crux: Balancing the density scaling factor to prevent over-smoothing in dense regions while preserving separation in sparse regions, achieved by using kNN-based local density estimates and a dynamic thresholding mechanism during graph construction.

**Step-by-Step Reasoning:**
1. **Local Density Estimation:** Compute local density for each point using kNN distances (e.g., 5th nearest neighbor) to capture spatial density variations.
2. **Adaptive Distance Metric:** Define a custom distance $ d'(x,y) = d_{\text{Euclidean}}(x,y) \cdot \frac{1}{\sqrt{\rho(x)\rho(y)}} $, where $ \rho $ is the local density. This scales distances in dense regions (low kNN distance) to be smaller, promoting clustering, and larger in sparse regions (high kNN distance), preventing false merges.
3. **Graph Construction:** Build a kNN graph using $ d' $, then compute the graph Laplacian to project data into a lower-dimensional space where density-aware structure is emphasized.
4. **Hybrid Clustering:** Apply HDBSCAN on the Laplacian embedding, leveraging its parameter-free density-based clustering while benefiting from the adaptive metric's structural refinement.

This approach addresses non-convexity via spectral embedding, adapts to density variations through the metric, and avoids parameter sensitivity by combining HDBSCAN's hierarchy with the graph's density-aware geometry.
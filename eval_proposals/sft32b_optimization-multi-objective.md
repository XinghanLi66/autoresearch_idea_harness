**Core idea:** Replace reference-vector-based selection entirely with a dynamic local-density estimation that adaptively identifies and rewards underpopulated regions of the objective space.

**Non-trivial crux:** The density must be estimated in objective space while accounting for the actual distribution of solutions in that space — computing it naively on the raw objective values will fail when objectives are on different scales or have skewed geometry.

**Reasoning:** The field has long recognized that reference points and crowding distance are scale-sensitive and static, which limits convergence and diversity. I'll borrow from density-estimation ideas in multi-objective literature but make it self-adaptive rather than parameter-driven. Here's the mechanism:

1. **Density as a proxy for underrepresentation:** Regions with fewer nearby solutions are underrepresented, so we should favor selection there. This is the same intuition as crowding distance, but I'll generalize it.

2. **Locality matters:** Density should be computed relative to the distribution of solutions in the objective space. I'll use a k-nearest-neighbor-type neighborhood — but instead of fixed k, I'll dynamically find the smallest number of neighbors whose objectives span the full objective range. This avoids being overwhelmed by outliers and adapts to the problem's geometry.

3. **Scale invariance:** Before computing any distances, I'll normalize the objectives to a unit hypercube. This is essential because raw distances are dominated by objectives with larger ranges, which would bias the selection.

4. **Distance metric:** Use a modified Manhattan distance weighted by the inverse of each objective's range. This makes objectives contribute proportionally to their actual spread rather than their scale.

5. **Selection pressure:** The final selection is a tournament where one candidate is chosen from a sparsely populated region and one from the rest. The sparser region candidate wins with a probability biased toward the difference in their density estimates.

How I keep myself honest: I'll benchmark against NSGA-II and NSGA-III on ZDT1, DTLZ2, and a real multi-objective engineering problem (e.g., wing design). The claim only holds if convergence and spread improve; if the adaptive density doesn't clearly beat the static methods, I'll fall back to a simpler, more interpretable baseline.

**Implementation sketch:** In the `survive` method, for each individual compute its density as the smallest k such that the k-nearest neighbors span the full normalized objective range, then use this as a selection bias in a sparsity-favoring tournament. Normalize objectives before any distance computation.
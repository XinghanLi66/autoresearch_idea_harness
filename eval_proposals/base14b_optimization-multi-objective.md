Core idea: Dynamically generate reference vectors from the current population's objective space using clustering, then use these adaptive vectors to guide survival selection with a hybrid metric combining dominance rank and cluster-based diversity.  
Non-trivial crux: Efficiently updating reference vectors via population clustering without introducing bias or computational overhead while maintaining convergence and diversity.  

**Mechanism Breakdown:**  
1. **Dynamic Reference Vectors:** After each generation, cluster the population's objective values (e.g., via k-means) to identify current Pareto front regions. These cluster centers act as adaptive reference points.  
2. **Hybrid Survival Metric:** For individuals in the same dominance front, compute a combined fitness metric that weights:  
   - **Dominance rank** (to ensure convergence).  
   - **Distance to nearest cluster center** (to maintain diversity by penalizing crowding in overrepresented regions).  
3. **Adaptive Clustering:** The number of clusters is proportional to the population size and problem dimensionality, ensuring coverage of the objective space while avoiding overfitting.  

This approach self-organizes reference points based on the population's evolution, outperforming fixed reference vectors (NSGA-III) and decomposition methods (MOEA/D) by adaptively responding to the problem's geometry.
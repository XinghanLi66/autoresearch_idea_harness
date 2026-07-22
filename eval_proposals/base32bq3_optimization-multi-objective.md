Core idea:  
Introduce **angle-penalized crowding distance** in parent selection, combining dominance rank with a dynamic geometric diversity metric that prioritizes solutions contributing to wider angular spreads in the objective space, especially in non-convex or high-dimensional Pareto front regions.  

Non-trivial crux:  
Designing a computationally efficient method to calculate adaptive angular distances between solutions in the objective space while avoiding bias toward any particular front shape, ensuring it complements non-dominated sorting without requiring reference vectors or prior knowledge of the problem geometry.
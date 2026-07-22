# Multi-Objective Optimization: Custom Evolutionary Strategy Design

## Research Question
Design a novel multi-objective evolutionary algorithm (MOEA) strategy that achieves better convergence, diversity, and spread on standard benchmark problems than classic approaches like NSGA-II, MOEA/D, and SPEA2.

## Background
Multi-objective optimization aims to find a set of Pareto-optimal solutions that represent the best trade-offs among conflicting objectives. Evolutionary algorithms are the dominant approach, differing primarily in three components:

- **Parent selection**: How to choose individuals for mating (e.g., tournament with crowding distance, reference-vector-based).
- **Variation**: How to produce offspring via crossover and mutation operators.
- **Environmental selection (survival)**: How to prune the combined parent+offspring pool back to population size (e.g., non-dominated sorting + crowding, decomposition into subproblems, indicator-based selection).

Classic algorithms include:
- **NSGA-II** (Deb et al., 2002): Non-dominated sorting + crowding distance for diversity.
- **MOEA/D** (Zhang & Li, 2007): Decomposes the problem into scalar subproblems using weight vectors.
- **SPEA2** (Zitzler et al., 2001): Strength-based fitness with density estimation via k-nearest neighbors.

State-of-the-art methods include:
- **NSGA-III** (Deb & Jain, 2014): Reference-point-based selection for many-objective problems.
- **RVEA** (Cheng et al., 2016): Angle-penalized distance with adaptive reference vectors.
- **AGE-MOEA** (Panichella, 2019): Adaptive geometry estimation for survival selection.

There is active research into strategies that combine ideas across these paradigms, adapt to problem geometry, or use novel diversity maintenance mechanisms.

## Task
Implement a custom multi-objective evolutionary strategy by modifying the `CustomMOEA` class in `deap/custom_moea.py`. You should implement the `select`, `vary`, `survive`, and optionally `on_generation` methods. The algorithm must work for both 2-objective and 3-objective problems.

## Interface
```python
class CustomMOEA:
    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        """Initialize the MOEA with problem parameters."""

    def select(self, population: list, k: int) -> list:
        """Select k parents from the population for mating.
        Returns: list of k selected individuals."""

    def vary(self, parents: list) -> list:
        """Apply crossover and mutation to produce offspring.
        Returns: list of offspring (fitness invalidated)."""

    def survive(self, population: list, offspring: list) -> list:
        """Environmental selection: choose pop_size individuals from combined pool.
        Returns: list of pop_size individuals for next generation."""

    def on_generation(self, gen: int, population: list):
        """Optional per-generation callback for adaptive strategies."""
```

Individual interface:
- `ind.fitness.values` -> tuple of objective values (all minimized)
- `ind.fitness.dominates(other.fitness)` -> bool
- `ind.fitness.valid` -> bool (True if evaluated)

Available DEAP utilities:
- `tools.sortNondominated(pop, k)` -> list of fronts
- `tools.selTournamentDCD(pop, k)` -> tournament selection (needs crowding dist)
- `tools.selNSGA3(pop, k, ref_points)` -> NSGA-III selection
- `tools.cxSimulatedBinaryBounded(ind1, ind2, eta, low, up)` -> SBX crossover
- `tools.mutPolynomialBounded(ind, eta, low, up, indpb)` -> polynomial mutation
- `tools.uniform_reference_points(nobj, p)` -> generate reference points
- `compute_crowding_distance(individuals)` -> sets `.fitness.crowding_dist`
- `get_nondominated(population)` -> first non-dominated front

## Evaluation
Evaluated on four benchmark problems (run with multiple seeds):
- **ZDT1** (2D objectives, convex front, 30 variables, 200 generations)
- **ZDT3** (2D objectives, disconnected front, 30 variables, 200 generations)
- **DTLZ2** (3D objectives, spherical front, 12 variables, 250 generations)
- **DTLZ1** (3D objectives, linear front with many local fronts, 7 variables, 400 generations)

Three metrics are reported:
- **Hypervolume (HV)**: Volume of objective space dominated by the Pareto front approximation. **Higher is better.**
- **Inverted Generational Distance (IGD)**: Average distance from true Pareto front points to nearest found solution. **Lower is better.**
- **Spread**: Uniformity of the Pareto front approximation. **Lower is better.**


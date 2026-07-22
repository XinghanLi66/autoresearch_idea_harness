# Evolutionary Optimization Strategy Design

## Research Question
Design a novel combination of selection, crossover, and mutation operators (and/or a novel evolutionary loop) for continuous black-box optimization that outperforms standard approaches across multiple benchmark functions.

## Background
Evolutionary algorithms (EAs) are population-based metaheuristics for black-box optimization. The three core operators — **selection**, **crossover**, and **mutation** — together with the overall evolutionary loop design, determine an EA's performance. Standard approaches include:
- **Genetic Algorithms (GA)**: Tournament selection + Simulated Binary Crossover (SBX) + Polynomial Mutation
- **CMA-ES**: Adapts the covariance matrix of a multivariate Gaussian to guide search
- **Differential Evolution (DE)**: Uses vector differences between population members for mutation

Each has strengths on different function landscapes (multimodal, ill-conditioned, high-dimensional), but no single strategy dominates all.

## Task
Modify the editable section of `custom_evolution.py` (lines 87-225) to implement a novel or improved evolutionary strategy. You may modify:
- `custom_select(population, k, toolbox)` — selection operator
- `custom_crossover(ind1, ind2)` — crossover/recombination operator
- `custom_mutate(individual, lo, hi)` — mutation operator
- `run_evolution(...)` — the full evolutionary loop (you can restructure the algorithm entirely)

The DEAP library (`deap.base`, `deap.creator`, `deap.tools`) is available. You may also use `numpy`, `scipy`, `math`, and `random`.

## Interface
- **Individuals**: Lists of floats, each with a `.fitness.values` attribute (tuple of one float for minimization).
- **`run_evolution`** must return `(best_individual, fitness_history)` where `fitness_history` is a list of best fitness per generation.
- **TRAIN_METRICS**: Print `TRAIN_METRICS gen=G best_fitness=F avg_fitness=A` periodically (every 50 generations).
- Respect the function signature and return types — the evaluation harness below the editable section is fixed.

## Evaluation
Strategies are evaluated on 4 benchmarks (all minimization, lower is better):
| Benchmark | Function | Dimensions | Domain | Global Minimum |
|-----------|----------|------------|--------|---------------|
| rastrigin-30d | Rastrigin | 30 | [-5.12, 5.12] | 0 |
| rosenbrock-30d | Rosenbrock | 30 | [-5, 10] | 0 |
| ackley-30d | Ackley | 30 | [-32.768, 32.768] | 0 |
| rastrigin-100d | Rastrigin | 100 | [-5.12, 5.12] | 0 |

**Metrics**: `best_fitness` (final best value, lower is better) and `convergence_gen` (generation reaching near-final fitness).


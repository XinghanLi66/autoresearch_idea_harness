# Symbolic Regression: GP Search Strategy

## Objective
Design and implement a better genetic programming search strategy for symbolic regression. Your code goes in `custom_sr.py`. Three reference implementations (Standard GP, Parsimony GP, Lexicase GP) demonstrate different approaches.

## Background
Symbolic regression discovers mathematical expressions that fit data. Genetic programming evolves a population of expression trees through selection, crossover, and mutation. Key challenges include balancing exploration vs exploitation, controlling expression complexity (bloat), and escaping local optima. Different approaches address these through fitness shaping, novel selection mechanisms, or improved genetic operators.

## Evaluation
Tested on three standard symbolic regression benchmarks: Nguyen-7 (univariate transcendental), Nguyen-10 (bivariate trigonometric), Koza-3 (univariate polynomial). Metric: R² on held-out test set (higher is better).

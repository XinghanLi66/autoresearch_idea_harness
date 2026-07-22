# Hyperparameter Optimization: Custom Search Strategy Design

## Research Question
Design a novel hyperparameter optimization (HPO) strategy that achieves better final validation scores and faster convergence than standard approaches like Random Search, TPE, Hyperband, and their combinations (BOHB, DEHB).

## Background
Hyperparameter optimization is a fundamental problem in machine learning: given a model and dataset, find the hyperparameter configuration that maximizes validation performance within a limited evaluation budget. This is a black-box optimization problem where each function evaluation (training + validation) is expensive.

Classic strategies include:
- **Random Search**: Samples configurations uniformly. Simple but surprisingly effective, especially when some hyperparameters are more important than others (Bergstra & Bengio, 2012).
- **TPE (Tree-structured Parzen Estimator)**: Models p(x|y<y*) and p(x|y>=y*) using kernel density estimation and maximizes their ratio (Bergstra et al., 2011).
- **Hyperband**: Uses multi-fidelity evaluation (early stopping) with successive halving to allocate resources to promising configurations (Li et al., 2017).

State-of-the-art methods combine these ideas:
- **BOHB**: Replaces random sampling in Hyperband with TPE-guided suggestions (Falkner et al., 2018).
- **DEHB**: Uses Differential Evolution within Hyperband's multi-fidelity framework (Awad et al., 2021).
- **CMA-ES**: Adapts a full covariance matrix of a Gaussian distribution for efficient continuous optimization (Hansen & Ostermeier, 2001).

There is ongoing research into strategies that better adapt to the optimization landscape, leverage multi-fidelity evaluations more effectively, or combine model-based search with evolutionary approaches.

## Task
Implement a custom HPO strategy by modifying the `CustomHPOStrategy` class in `scikit-learn/custom_hpo.py`. You should implement both `__init__` and `suggest` methods. The class is called repeatedly in a sequential loop where each call proposes one configuration to evaluate.

## Interface
```python
class CustomHPOStrategy:
    def __init__(self, seed: int = 42):
        """Initialize the strategy with a random seed."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Propose the next configuration to evaluate.

        Args:
            space: SearchSpace with .params (list of HParam), .dim,
                   .sample_uniform(rng), .clip(config)
            history: list of Trial(config, score, budget) from past evals
            budget_left: remaining budget in full-fidelity units

        Returns:
            config: dict mapping hyperparameter names to values
            fidelity: float in (0, 1] for multi-fidelity evaluation
        """
```

The search space provides:
- `space.params` -- list of HParam objects with name, type ("float"/"int"/"categorical"), low, high, log_scale, choices
- `space.sample_uniform(rng)` -- sample a random valid configuration
- `space.clip(config)` -- clip values to valid ranges

Each Trial records:
- `trial.config` -- the hyperparameter configuration dict
- `trial.score` -- observed validation score (higher is better)
- `trial.budget` -- fidelity fraction used (1.0 = full evaluation)

The fidelity parameter controls evaluation cost: lower fidelity means cheaper but noisier evaluation (e.g., fewer boosting rounds, fewer CV folds, fewer MLP epochs).

## Evaluation
Evaluated on three ML model tuning benchmarks (**higher best_val_score is better, higher convergence_auc is better**):

- **XGBoost** (6D: n_estimators, max_depth, learning_rate, subsample, min_samples_split, min_samples_leaf; GradientBoostingRegressor on California Housing; budget=50)
- **SVM** (3D: C, gamma, kernel; SVC on Breast Cancer; budget=40)
- **Neural Net** (6D: hidden layers, learning rate, alpha, batch_size, activation; MLP on Diabetes; budget=40)

Metrics:
- **best_val_score**: Best validation score found within the budget (primary metric)
- **convergence_auc**: Area under the normalized convergence curve (higher = found good configs earlier)

Each benchmark runs with multiple seeds; mean metrics across seeds are reported.


# SLDBench Scaling Law Discovery

## Research Question
Can you design a better scaling-law model that extrapolates on held-out `SLDBench` scaling tasks while keeping a single functional form per task and fitting group-specific coefficients from observed trials?

## Background
This task is a pure `SLDBench` benchmark inspired by `Can Language Models Discover Scaling Laws?` (`arXiv:2507.21184`).

It keeps three representative and harder subsets (less saturated than the
original `parallel`/`moe`/`sft` trio):

- `sld-vocab`: vocabulary scaling law — unigram-normalised loss as a function
  of non-vocabulary parameters `N`, vocabulary size `V`, and training
  characters `D` (see Tao et al. "Scaling Laws with Vocabulary").
- `sld-lrbsz`: learning-rate & batch-size scaling law — LM loss as a joint
  function of learning rate, batch size, training tokens, and
  non-embedding parameters.
- `sld-dataconstrained`: data-constrained scaling law — loss as a function of
  unique tokens `U`, parameters `N`, and total tokens `D`, where `D` can
  exceed `U` (data repetition). See Muennighoff et al. 2023.

The goal is not generic tabular regression. The intended object is a scaling law: a shared functional form for each benchmark, with coefficients that can vary by experimental `group`.

## Task
Edit the `ScalingLawModel` class in `custom_scaling_law.py`.

Your model receives:

- `X_num`: raw numeric inputs (see per-benchmark list below)
- `X_cat`: categorical metadata, primarily the `group`
- `y`: observed target losses on the training split

The runtime already loads the official `SLDBench` train/test splits from `/data/scaling_law/*.jsonl`.

The observed training trials are also mirrored into the editable workspace as read-only files:

- `scaling-law-lab/observed_trials/sld_vocab_train.jsonl`
- `scaling-law-lab/observed_trials/sld_lrbsz_train.jsonl`
- `scaling-law-lab/observed_trials/sld_dataconstrained_train.jsonl`

You are expected to inspect these raw train trials directly and discover benchmark-specific symbolic laws. Large pretrained LMs are not allowed.

## Benchmarks
- `sld-vocab`
  - numeric inputs: `non_vocab_parameters`, `vocab_size`, `num_characters`
  - categorical input: `group`
  - target: `unigram_normalized_loss` (can be negative)
- `sld-lrbsz`
  - numeric inputs: `lr`, `bsz`, `data_size`, `non_embedding_param_size`
  - categorical input: `group`
  - target: `lm_loss`
- `sld-dataconstrained`
  - numeric inputs: `unique_tokens`, `params`, `tokens`
  - categorical input: `group`
  - target: `loss`

## Interface
Implement:

```python
class ScalingLawModel:
    def __init__(self, benchmark_name, numeric_names, categorical_names):
        ...

    def fit(self, X_num, X_cat, y):
        return self

    def predict(self, X_num, X_cat):
        return y_pred
```

`benchmark_name` lets you use different law families for `vocab`, `lrbsz`, and `dataconstrained`. You should feel free to write different symbolic forms per benchmark, while still keeping one shared expression within each benchmark and fitting group-specific coefficients.

Note: for `sld-vocab` the target (`unigram_normalized_loss`) can be negative, so do not clip your predictions to positive values.

## Evaluation
Primary metric: held-out test `R^2` for each benchmark.

Secondary metrics:

- `MAE`
- `RMSE`
- `NMAE`

Strong solutions usually have two properties:

- they fit coefficients per `group` instead of collapsing all groups together
- they preserve sensible asymptotics on larger or denser test points

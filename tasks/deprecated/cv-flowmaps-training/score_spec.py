"""Score spec for cv-flowmaps-training."""
from mlsbench.scoring.dsl import *

# Metrics are label-prefixed (fid_train_small, fid_train_medium, etc.)
# Keep best_fid_* (peak performance); drop fid_* duplicates

term("best_fid_train_small",
    col("best_fid_train_small").lower().id()
    .bounded_power(bound=0.0))

term("best_fid_train_medium",
    col("best_fid_train_medium").lower().id()
    .bounded_power(bound=0.0))

term("best_fid_train_large",
    col("best_fid_train_large").lower().id()
    .bounded_power(bound=0.0))

setting("train_small", weighted_mean(("best_fid_train_small", 1.0)))
setting("train_medium", weighted_mean(("best_fid_train_medium", 1.0)))
setting("train_large", weighted_mean(("best_fid_train_large", 1.0)))

task(gmean("train_small", "train_medium", "train_large"))

"""Score spec for pde-autoregressive-solver."""
from mlsbench.scoring.dsl import *

term("rel_err_NS",
    col("rel_err_NS").lower().id()
    .bounded_power(bound=0.0))

term("rel_err_DiffSorp",
    col("rel_err_DiffSorp").lower().id()
    .bounded_power(bound=0.0))

term("rel_err_Burgers",
    col("rel_err_Burgers").lower().id()
    .bounded_power(bound=0.0))

setting("NS", weighted_mean(("rel_err_NS", 1.0)))
setting("DiffSorp", weighted_mean(("rel_err_DiffSorp", 1.0)))
setting("Burgers", weighted_mean(("rel_err_Burgers", 1.0)))

task(gmean("NS", "DiffSorp", "Burgers"))

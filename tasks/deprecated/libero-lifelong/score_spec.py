"""Score spec for libero-lifelong."""
from mlsbench.scoring.dsl import *

term("avg_final_success",
    col("avg_final_success").higher().id()
    .bounded_power(bound=1.0))

setting("train", weighted_mean(("avg_final_success", 1.0)))

task(gmean("train"))

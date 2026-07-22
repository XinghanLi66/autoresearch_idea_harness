"""Score spec for robot-imitation-objective."""
from mlsbench.scoring.dsl import *

term("success_rate_lift",
    col("success_rate_lift").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_can",
    col("success_rate_can").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_square",
    col("success_rate_square").higher().id()
    .bounded_power(bound=1.0))

setting("lift", weighted_mean(("success_rate_lift", 1.0)))
setting("can", weighted_mean(("success_rate_can", 1.0)))
setting("square", weighted_mean(("success_rate_square", 1.0)))

task(gmean("lift", "can", "square"))

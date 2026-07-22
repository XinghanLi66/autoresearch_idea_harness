"""Score spec for robot-policy-conditioning."""
from mlsbench.scoring.dsl import *

term("success_rate_lift_ph",
    col("success_rate_lift_ph").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_can_ph",
    col("success_rate_can_ph").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_square_ph",
    col("success_rate_square_ph").higher().id()
    .bounded_power(bound=1.0))

setting("lift_ph", weighted_mean(("success_rate_lift_ph", 1.0)))
setting("can_ph", weighted_mean(("success_rate_can_ph", 1.0)))
setting("square_ph", weighted_mean(("success_rate_square_ph", 1.0)))

task(gmean("lift_ph", "can_ph", "square_ph"))

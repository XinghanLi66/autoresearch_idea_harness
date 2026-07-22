"""Score spec for rl-gcrl-goal-representation."""
from mlsbench.scoring.dsl import *

# Parser emits TEST_METRICS success_rate per OGBench environment.
# Success rate is a fraction in [0, 1].

term("success_rate_antmaze_large_navigate_v0",
    col("success_rate_antmaze_large_navigate_v0").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_cube_single_noisy_v0",
    col("success_rate_cube_single_noisy_v0").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_pointmaze_large_navigate_v0",
    col("success_rate_pointmaze_large_navigate_v0").higher().id()
    .bounded_power(bound=1.0))

setting("antmaze-large-navigate-v0", weighted_mean(("success_rate_antmaze_large_navigate_v0", 1.0)))
setting("cube-single-noisy-v0", weighted_mean(("success_rate_cube_single_noisy_v0", 1.0)))
setting("pointmaze-large-navigate-v0", weighted_mean(("success_rate_pointmaze_large_navigate_v0", 1.0)))

task(gmean("antmaze-large-navigate-v0", "cube-single-noisy-v0", "pointmaze-large-navigate-v0"))

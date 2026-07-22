"""Score spec for rl-offline-policy."""
from mlsbench.scoring.dsl import *

# Parser emits TEST_METRICS success_rate per OGBench offline-control setting.
# Success rate is a fraction in [0, 1].

term("success_rate_antmaze_large_navigate",
    col("success_rate_antmaze_large_navigate").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_cube_single_play",
    col("success_rate_cube_single_play").higher().id()
    .bounded_power(bound=1.0))

term("success_rate_antsoccer_arena_navigate",
    col("success_rate_antsoccer_arena_navigate").higher().id()
    .bounded_power(bound=1.0))

setting("antmaze-large-navigate", weighted_mean(("success_rate_antmaze_large_navigate", 1.0)))
setting("cube-single-play", weighted_mean(("success_rate_cube_single_play", 1.0)))
setting("antsoccer-arena-navigate", weighted_mean(("success_rate_antsoccer_arena_navigate", 1.0)))

task(gmean("antmaze-large-navigate", "cube-single-play", "antsoccer-arena-navigate"))

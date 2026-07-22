"""Score spec for marl-mixing-network."""
from mlsbench.scoring.dsl import *

# test_return_mean: higher is better (less negative), unbounded — use sigmoid.
# test_battle_won_mean: higher is better, bounded at 1.
# test_return_std is informational variance across episodes and is dropped.

term("test_return_mean_3s5z_vs_3s6z",
    col("test_return_mean_3s5z_vs_3s6z").higher().id()
    .sigmoid())

term("test_battle_won_mean_3s5z_vs_3s6z",
    col("test_battle_won_mean_3s5z_vs_3s6z").higher().id()
    .bounded_power(bound=1.0))

term("test_return_mean_corridor",
    col("test_return_mean_corridor").higher().id()
    .sigmoid())

term("test_battle_won_mean_corridor",
    col("test_battle_won_mean_corridor").higher().id()
    .bounded_power(bound=1.0))

term("test_return_mean_MMM2",
    col("test_return_mean_MMM2").higher().id()
    .sigmoid())

term("test_battle_won_mean_MMM2",
    col("test_battle_won_mean_MMM2").higher().id()
    .bounded_power(bound=1.0))

setting("3s5z_vs_3s6z", weighted_mean(("test_return_mean_3s5z_vs_3s6z", 1.0), ("test_battle_won_mean_3s5z_vs_3s6z", 1.0)))
setting("corridor", weighted_mean(("test_return_mean_corridor", 1.0), ("test_battle_won_mean_corridor", 1.0)))
setting("MMM2", weighted_mean(("test_return_mean_MMM2", 1.0), ("test_battle_won_mean_MMM2", 1.0)))

task(gmean("3s5z_vs_3s6z", "corridor", "MMM2"))

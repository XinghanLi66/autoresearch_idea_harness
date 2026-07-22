"""Score spec for humanoid-ppo-extractor."""
from mlsbench.scoring.dsl import *

# mean_reward is the primary metric; std_reward is within-run episode variance — dropped
term("mean_reward_h1_stand",
    col("mean_reward_h1_stand").higher().id()
    .sigmoid())

term("mean_reward_h1_walk",
    col("mean_reward_h1_walk").higher().id()
    .sigmoid())

term("mean_reward_h1_run",
    col("mean_reward_h1_run").higher().id()
    .sigmoid())

setting("h1-stand", weighted_mean(("mean_reward_h1_stand", 1.0)))
setting("h1-walk", weighted_mean(("mean_reward_h1_walk", 1.0)))
setting("h1-run", weighted_mean(("mean_reward_h1_run", 1.0)))

task(gmean("h1-stand", "h1-walk", "h1-run"))

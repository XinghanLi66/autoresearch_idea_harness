"""Score spec for rl-offline-discrete (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("eval_return_breakout",
    col("eval_return_breakout").higher().id()
    .sigmoid())

term("eval_return_qbert",
    col("eval_return_qbert").higher().id()
    .sigmoid())

term("eval_return_pong",
    col("eval_return_pong").higher().id()
    .sigmoid())

setting("breakout", weighted_mean(("eval_return_breakout", 1.0)))
setting("qbert", weighted_mean(("eval_return_qbert", 1.0)))
setting("pong", weighted_mean(("eval_return_pong", 1.0)))

task(gmean("breakout", "qbert", "pong"))

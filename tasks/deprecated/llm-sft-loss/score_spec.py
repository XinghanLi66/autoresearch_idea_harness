"""Score spec for llm-sft-loss."""
from mlsbench.scoring.dsl import *

term("hellaswag_acc_norm",
    col("hellaswag_acc_norm").higher().id()
    .bounded_power(bound=1.0))

term("arc_challenge_acc_norm",
    col("arc_challenge_acc_norm").higher().id()
    .bounded_power(bound=1.0))

term("piqa_acc",
    col("piqa_acc").higher().id()
    .bounded_power(bound=1.0))

setting("lm-eval", weighted_mean(
    ("hellaswag_acc_norm", 1.0),
    ("arc_challenge_acc_norm", 1.0),
    ("piqa_acc", 1.0),
))

task(gmean("lm-eval"))

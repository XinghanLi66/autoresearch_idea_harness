"""Score spec for llm-pretrain-weight-averaging."""
from mlsbench.scoring.dsl import *

term("val_loss_gpt_345m",
    col("val_loss_gpt-345m").lower().id()
    .bounded_power(bound=0.0))

setting("gpt-345m", weighted_mean(("val_loss_gpt_345m", 1.0)))

task(gmean("gpt-345m"))

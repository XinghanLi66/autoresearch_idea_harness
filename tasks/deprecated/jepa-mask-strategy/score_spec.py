"""Score spec for jepa-mask-strategy."""
from mlsbench.scoring.dsl import *

term("val_acc_vit_tiny",
    col("val_acc_vit_tiny").higher().id()
    .bounded_power(bound=100.0))

term("val_acc_vit_small",
    col("val_acc_vit_small").higher().id()
    .bounded_power(bound=100.0))

term("val_acc_vit_base",
    col("val_acc_vit_base").higher().id()
    .bounded_power(bound=100.0))

setting("vit_tiny", weighted_mean(("val_acc_vit_tiny", 1.0)))
setting("vit_small", weighted_mean(("val_acc_vit_small", 1.0)))
setting("vit_base", weighted_mean(("val_acc_vit_base", 1.0)))

task(gmean("vit_tiny", "vit_small", "vit_base"))

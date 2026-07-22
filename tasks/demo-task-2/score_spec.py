"""Score spec for demo-task-2 (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("test_accuracy",
    col("test_accuracy").higher().id()
    .bounded_power(bound=100.0))

term("test_loss",
    col("test_loss").lower().id()
    .bounded_power(bound=0.0))

setting("default", weighted_mean(("test_accuracy", 1.0), ("test_loss", 1.0)))

task(gmean("default"))

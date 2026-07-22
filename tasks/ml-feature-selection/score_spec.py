"""Score spec for ml-feature-selection."""
from mlsbench.scoring.dsl import *

# accuracy is on 0-1 scale based on leaderboard values (0.5556, 0.8985, 0.6115)
term("accuracy_20newsgroups",
    col("accuracy_20newsgroups").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_mnist",
    col("accuracy_mnist").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_madelon",
    col("accuracy_madelon").higher().id()
    .bounded_power(bound=1.0))

setting("20newsgroups", weighted_mean(("accuracy_20newsgroups", 1.0)))
setting("mnist", weighted_mean(("accuracy_mnist", 1.0)))
setting("madelon", weighted_mean(("accuracy_madelon", 1.0)))

task(gmean("20newsgroups", "mnist", "madelon"))

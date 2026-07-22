"""Score spec for llm-hybrid-posttraining."""
from mlsbench.scoring.dsl import *

term("test_score_aime24",
    col("test_score_aime24").higher().id()
    .bounded_power(bound=1.0))

term("test_score_amc23",
    col("test_score_amc23").higher().id()
    .bounded_power(bound=1.0))

term("test_score_math_500",
    col("test_score_math_500").higher().id()
    .bounded_power(bound=1.0))

setting("AIME24", weighted_mean(("test_score_aime24", 1.0)))
setting("AMC23", weighted_mean(("test_score_amc23", 1.0)))
setting("MATH-500", weighted_mean(("test_score_math_500", 1.0)))

task(gmean("AIME24", "AMC23", "MATH-500"))

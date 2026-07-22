"""Score spec for speech-speaker-embedding (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("eer_voxceleb1_test",
    col("eer_voxceleb1-test").lower().id()
    .bounded_power(bound=0.0))  # TODO: set ref (no baseline data)

term("eer_voxceleb1_hard",
    col("eer_voxceleb1-hard").lower().id()
    .bounded_power(bound=0.0))  # TODO: set ref (no baseline data)

term("eer_cnceleb_eval",
    col("eer_cnceleb-eval").lower().id()
    .bounded_power(bound=0.0))  # TODO: set ref (no baseline data)

setting("voxceleb1-test", weighted_mean(("eer_voxceleb1_test", 1.0)))
setting("voxceleb1-hard", weighted_mean(("eer_voxceleb1_hard", 1.0)))
setting("cnceleb-eval", weighted_mean(("eer_cnceleb_eval", 1.0)))

task(gmean("voxceleb1-test", "voxceleb1-hard", "cnceleb-eval"))

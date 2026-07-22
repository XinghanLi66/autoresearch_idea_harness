"""Score spec for speech-asr-encoder."""
from mlsbench.scoring.dsl import *

term("wer_librispeech_100h",
    col("wer_librispeech-100h").lower().id()
    .bounded_power(bound=0.0))

term("cer_librispeech_100h",
    col("cer_librispeech-100h").lower().id()
    .bounded_power(bound=0.0))

term("wer_aishell_1",
    col("wer_aishell-1").lower().id()
    .bounded_power(bound=0.0))

term("cer_aishell_1",
    col("cer_aishell-1").lower().id()
    .bounded_power(bound=0.0))

term("wer_mls_spanish",
    col("wer_mls-spanish").lower().id()
    .bounded_power(bound=0.0))

term("cer_mls_spanish",
    col("cer_mls-spanish").lower().id()
    .bounded_power(bound=0.0))

setting("librispeech-100h", weighted_mean(("wer_librispeech_100h", 1.0), ("cer_librispeech_100h", 1.0)))
setting("aishell-1", weighted_mean(("wer_aishell_1", 1.0), ("cer_aishell_1", 1.0)))
setting("mls-spanish", weighted_mean(("wer_mls_spanish", 1.0), ("cer_mls_spanish", 1.0)))

task(gmean("librispeech-100h", "aishell-1", "mls-spanish"))

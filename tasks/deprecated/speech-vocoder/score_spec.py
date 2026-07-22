"""Score spec for speech-vocoder."""
from mlsbench.scoring.dsl import *

# mel_loss: lower better, bounded at 0
# pesq: higher better, bounded ~[1,5] -> bounded_power(bound=5.0)

term("mel_loss_ljspeech",
    col("mel_loss_ljspeech").lower().id()
    .bounded_power(bound=0.0))  # TODO: set ref when baseline data available

term("pesq_ljspeech",
    col("pesq_ljspeech").higher().id()
    .bounded_power(bound=5.0))

term("mel_loss_vctk_5spk",
    col("mel_loss_vctk-5spk").lower().id()
    .bounded_power(bound=0.0))

term("pesq_vctk_5spk",
    col("pesq_vctk-5spk").higher().id()
    .bounded_power(bound=5.0))

term("mel_loss_aishell3_5spk",
    col("mel_loss_aishell3-5spk").lower().id()
    .bounded_power(bound=0.0))  # TODO: set ref when baseline data available

term("pesq_aishell3_5spk",
    col("pesq_aishell3-5spk").higher().id()
    .bounded_power(bound=5.0))

setting("ljspeech", weighted_mean(("mel_loss_ljspeech", 1.0), ("pesq_ljspeech", 1.0)))
setting("vctk-5spk", weighted_mean(("mel_loss_vctk_5spk", 1.0), ("pesq_vctk_5spk", 1.0)))
setting("aishell3-5spk", weighted_mean(("mel_loss_aishell3_5spk", 1.0), ("pesq_aishell3_5spk", 1.0)))

task(gmean("ljspeech", "vctk-5spk", "aishell3-5spk"))

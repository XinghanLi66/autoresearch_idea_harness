"""Score spec for speech-enhancement."""
from mlsbench.scoring.dsl import *

# si_snr: higher better, unbounded -> sigmoid
# pesq: higher better, bounded ~[1,5] -> bounded_power(bound=1.0)
# stoi: higher better, bounded [0,1] -> bounded_power(bound=1.0)
term("si_snr_voicebank_demand",
    col("si_snr_voicebank-demand").higher().id()
    .sigmoid())

term("pesq_voicebank_demand",
    col("pesq_voicebank-demand").higher().id()
    .bounded_power(bound=5.0))

term("stoi_voicebank_demand",
    col("stoi_voicebank-demand").higher().id()
    .bounded_power(bound=1.0))

term("si_snr_noisy_vctk_56spk",
    col("si_snr_noisy-vctk-56spk").higher().id()
    .sigmoid())

term("pesq_noisy_vctk_56spk",
    col("pesq_noisy-vctk-56spk").higher().id()
    .bounded_power(bound=5.0))

term("stoi_noisy_vctk_56spk",
    col("stoi_noisy-vctk-56spk").higher().id()
    .bounded_power(bound=1.0))

term("si_snr_librimix_2spk",
    col("si_snr_librimix-2spk").higher().id()
    .sigmoid())

term("pesq_librimix_2spk",
    col("pesq_librimix-2spk").higher().id()
    .bounded_power(bound=5.0))

term("stoi_librimix_2spk",
    col("stoi_librimix-2spk").higher().id()
    .bounded_power(bound=1.0))

setting("voicebank-demand", weighted_mean(("si_snr_voicebank_demand", 1.0), ("pesq_voicebank_demand", 1.0), ("stoi_voicebank_demand", 1.0)))
setting("noisy-vctk-56spk", weighted_mean(("si_snr_noisy_vctk_56spk", 1.0), ("pesq_noisy_vctk_56spk", 1.0), ("stoi_noisy_vctk_56spk", 1.0)))
setting("librimix-2spk", weighted_mean(("si_snr_librimix_2spk", 1.0), ("pesq_librimix_2spk", 1.0), ("stoi_librimix_2spk", 1.0)))

task(gmean("voicebank-demand", "noisy-vctk-56spk", "librimix-2spk"))

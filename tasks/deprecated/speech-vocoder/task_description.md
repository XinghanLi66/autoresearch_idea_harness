# Neural Vocoder Generator Design

## Research Question
Design an improved neural vocoder generator network. Given a mel-spectrogram, the generator synthesizes high-quality raw audio waveform. The entire generator architecture — including upsampling strategy, residual block design, activation functions, and conditioning method — is open for modification. The multi-scale discriminator and GAN training loop are fixed.

## Background
Neural vocoders convert mel-spectrograms to waveforms for text-to-speech and voice conversion. Key architectures:
- **HiFi-GAN** (Kong et al., 2020): Multi-Receptive Field Fusion (MRF) with transposed conv upsampling — current gold standard for quality-speed tradeoff
- **MelGAN** (Kumar et al., 2019): Lightweight generator using reflection padding and dilated residual stacks
- **UnivNet** (Jang et al., 2021): Location-Variable Convolution with mel conditioning at each upsampling stage

## Task
Modify the `VocoderGenerator` class in `speechbrain/custom_vocoder.py` (lines 54-124). The class must implement:
- `__init__(self, n_mels=80, upsample_rates=(8,8,2,2), d_model=512, dropout=0.1)`
- `forward(self, mel) -> waveform`

Input: mel-spectrogram `(B, 80, T_mel)`.
Output: audio waveform `(B, 1, T_audio)`.

## Evaluation
- **Metrics**: Mel spectrogram loss (lower is better), PESQ (higher is better)
- **Environments**: LJSpeech (single speaker), VCTK 5-speaker (multi-speaker English), AISHELL-3 5-speaker (multi-speaker Chinese)
- **Baselines**: HiFi-GAN V1, MelGAN, UnivNet


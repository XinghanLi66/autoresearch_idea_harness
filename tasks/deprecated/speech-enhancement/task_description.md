# Speech Enhancement Model Design

## Research Question
Design an improved speech enhancement / denoising model. Given a noisy speech waveform, the model should output the corresponding clean speech. The entire model architecture — including the choice of time-domain vs frequency-domain processing, encoder-decoder design, mask estimation strategy, and sequence modeling approach — is open for modification.

## Background
Speech enhancement removes background noise, reverberation, or interfering speakers from an audio signal. Key paradigms include:
- **Frequency-domain masking**: Estimate a mask (IRM, cIRM, PSM) in STFT domain to filter noise
- **Time-domain processing**: Directly map noisy to clean waveform using learned encoder-decoder (Conv-TasNet)
- **Complex-domain methods**: Operate on complex STFT for joint magnitude and phase enhancement (DCCRN)
- **Dual-path models**: Handle long sequences by segmenting and processing locally + globally (DPT-Net)

## Task
Modify the `EnhancementModel` class in `speechbrain/custom_speech_enhancement.py` (lines 56-140). The class must implement:
- `__init__(self, n_fft=512, hop_length=256, d_model=256, n_layers=4, n_head=4, dropout=0.1)`
- `forward(self, noisy_waveform) -> enhanced_waveform`

Input: noisy 16kHz waveform `(B, T)`.
Output: enhanced waveform `(B, T)` (same length as input).

Training loss (SI-SNR + multi-resolution STFT), data loading, and evaluation are fixed.

## Evaluation
- **Metrics**: SI-SNR improvement (higher is better), PESQ (higher is better), STOI (higher is better)
- **Environments**: VoiceBank-DEMAND (stationary noise), Noisy-VCTK-56spk (multi-speaker diverse noise), LibriMix 2-speaker (separation)
- **Baselines**: Conv-TasNet (time-domain), DCCRN (complex frequency-domain), DPT-Net (dual-path Transformer)


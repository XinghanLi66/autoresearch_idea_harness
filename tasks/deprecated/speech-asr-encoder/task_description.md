# Speech ASR Encoder Design

## Research Question
Design an improved end-to-end speech recognition encoder architecture. Given raw audio waveform input, the encoder must produce frame-level representations for CTC (Connectionist Temporal Classification) decoding. The entire encoder design — including feature extraction, subsampling, encoder blocks, attention mechanism, position encoding, and normalization — is open for modification.

## Background
Modern ASR encoders have evolved from simple RNN-based models to Transformer variants. The Conformer (Gulati et al., 2020) introduced convolution-augmented Transformers that capture both local and global patterns in speech. The Branchformer (Peng et al., 2022) uses parallel attention and convolution branches instead of sequential composition. Key design decisions include:
- **Feature extraction**: Log-mel filterbanks are standard, but learnable frontends (SincNet, LEAF) can adapt to the data
- **Subsampling**: CNN-based downsampling reduces the long speech sequences (100 frames/sec) to manageable lengths
- **Encoder blocks**: The arrangement of attention, convolution, and feed-forward modules significantly affects performance
- **Position encoding**: Speech sequences are long and variable-length, requiring effective positional information

## Task
Modify the `SpeechEncoder` class in `speechbrain/custom_asr_encoder.py` (lines 61-220). The class must implement:
- `__init__(self, n_vocab, d_model=256, n_head=4, n_layers=4, d_ffn=1024, kernel_size=31, dropout=0.1)`
- `forward(self, waveform, wav_lens) -> (log_probs, out_lens)`

Input: raw 16kHz waveform `(B, T)` and relative lengths `(B,)`.
Output: CTC log-probabilities `(B, T', n_vocab)` and relative output lengths `(B,)`.

The CTC decoder head, data loading, training loop, and evaluation are fixed.

## Evaluation
- **Metric**: Word Error Rate (WER, lower is better) for English/Spanish; Character Error Rate (CER, lower is better) for Chinese
- **Environments**: LibriSpeech train-clean-100 (English), AISHELL-1 (Mandarin Chinese), MLS Spanish
- **Baselines**: Conformer (~SOTA), pure Transformer (simpler), Branchformer (parallel branches)


"""Transformer baseline for speech-asr-encoder.

Reference: vendor/external_packages/speechbrain/speechbrain/lobes/models/transformer/TransformerASR.py
Paper: Vaswani et al., "Attention Is All You Need", 2017 (adapted for speech)
"""

_FILE = "speechbrain/custom_asr_encoder.py"

_TRANSFORMER_ENCODER = '''\
class SpeechEncoder(nn.Module):
    """Pure Transformer encoder for ASR (Vaswani et al., 2017 adapted for speech).

    Architecture:
      - 80-dim log-mel filterbank features
      - 2-layer CNN subsampling (stride 2 each → 4x reduction)
      - N Transformer blocks, each:
          LayerNorm → MultiHeadAttention → Residual → LayerNorm → FFN → Residual
      - Sinusoidal positional encoding (absolute)
      - CTC output projection

    Key design choices:
      - Standard pre-norm Transformer (no convolution module)
      - Absolute sinusoidal position encoding
      - GELU activation in FFN
    """

    def __init__(self, n_vocab, d_model=256, n_head=4, n_layers=4,
                 d_ffn=1024, kernel_size=31, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_vocab = n_vocab
        self.n_mels = 80
        self.n_fft = 400
        self.hop_length = 160

        self.register_buffer(
            "mel_basis",
            self._create_mel_filterbank(16000, self.n_fft, self.n_mels),
        )

        # CNN subsampling
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.subsample_proj = nn.Linear(d_model * (self.n_mels // 4), d_model)

        # Sinusoidal positional encoding (registered as buffer so it moves with .to(device))
        self.register_buffer("pos_enc", self._make_sinusoidal_pe(5000, d_model))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # CTC output
        self.ctc_proj = nn.Linear(d_model, n_vocab)

    def forward(self, waveform, wav_lens):
        feats = self._extract_features(waveform)
        x = feats.unsqueeze(1)
        x = self.subsample(x)
        B, C, Freq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Freq)
        x = self.subsample_proj(x)

        # Add positional encoding
        x = x + self.pos_enc[:, :T, :]
        x = self.dropout(x)

        out_lens = wav_lens
        seq_len = T
        abs_lens = (out_lens * seq_len).long()
        padding_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= abs_lens.unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        log_probs = F.log_softmax(self.ctc_proj(x), dim=-1)
        return log_probs, out_lens

    def _extract_features(self, waveform):
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(waveform, self.n_fft, self.hop_length, self.n_fft,
                          window=window, return_complex=True)
        power_spec = stft.abs().pow(2)
        mel_spec = torch.matmul(self.mel_basis, power_spec)
        log_mel = torch.log(mel_spec.clamp(min=1e-9))
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (log_mel - mean) / std

    @staticmethod
    def _create_mel_filterbank(sample_rate, n_fft, n_mels, fmin=0.0, fmax=8000.0):
        def hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = (hz_points * n_fft / sample_rate).long()
        filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            start, center, end = bin_points[i], bin_points[i + 1], bin_points[i + 2]
            if start < center:
                filterbank[i, start:center] = torch.linspace(0, 1, center - start)
            if center < end:
                filterbank[i, center:end] = torch.linspace(1, 0, end - center)
        return filterbank

    @staticmethod
    def _make_sinusoidal_pe(max_len, d_model):
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 61,
        "end_line": 220,
        "content": _TRANSFORMER_ENCODER,
    },
]

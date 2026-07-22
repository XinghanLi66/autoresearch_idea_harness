"""Conformer baseline for speech-asr-encoder.

Reference: vendor/external_packages/speechbrain/speechbrain/lobes/models/transformer/Conformer.py
Paper: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", 2020
"""

_FILE = "speechbrain/custom_asr_encoder.py"

_CONFORMER_ENCODER = '''\
class SpeechEncoder(nn.Module):
    """Conformer encoder: convolution-augmented Transformer (Gulati et al., 2020).

    Architecture:
      - 80-dim log-mel filterbank features
      - 2-layer CNN subsampling (stride 2 each → 4x reduction)
      - N Conformer blocks, each:
          FFN(0.5) → MHSA(RelPos) → ConvModule → FFN(0.5) → LayerNorm
      - CTC output projection

    Key design choices:
      - Macaron-style FFN (half-step before and after attention)
      - Relative positional encoding in attention (TransformerXL style)
      - Depthwise separable convolution with GLU gating
      - Pre-norm (LayerNorm before each sub-layer)
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

        # Relative positional encoding
        self.rel_pos_enc = RelativeSinusoidalPE(d_model)

        self.dropout = nn.Dropout(dropout)

        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_head, d_ffn, kernel_size, dropout)
            for _ in range(n_layers)
        ])
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

        out_lens = wav_lens
        seq_len = T
        abs_lens = (out_lens * seq_len).long()
        padding_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= abs_lens.unsqueeze(1)

        x = self.dropout(x)
        pos_emb = self.rel_pos_enc(x)

        for layer in self.layers:
            x = layer(x, pos_emb, padding_mask)

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


class RelativeSinusoidalPE(nn.Module):
    """Relative sinusoidal positional encoding (TransformerXL style)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN → MHSA → Conv → FFN → LN."""

    def __init__(self, d_model, n_head, d_ffn, kernel_size, dropout):
        super().__init__()
        # First FFN (half-step)
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model), nn.Dropout(dropout),
        )
        # Multi-head self-attention with relative position
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RelPosMultiHeadAttention(d_model, n_head, dropout)
        self.attn_drop = nn.Dropout(dropout)
        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_module = ConformerConvModule(d_model, kernel_size, dropout)
        # Second FFN (half-step)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model), nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, padding_mask=None):
        # FFN 1 (half-step)
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        # MHSA
        attn_out = self.attn(self.attn_norm(x), pos_emb, padding_mask)
        x = x + self.attn_drop(attn_out)
        # Conv
        x = x + self.conv_module(self.conv_norm(x))
        # FFN 2 (half-step)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


class RelPosMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding."""

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_pos = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model)

        self.pos_bias_u = nn.Parameter(torch.zeros(n_head, self.d_head))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_head, self.d_head))
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x, pos_emb, padding_mask=None):
        B, T, _ = x.shape

        q = self.w_q(x).view(B, T, self.n_head, self.d_head).permute(0, 2, 1, 3)
        k = self.w_k(x).view(B, T, self.n_head, self.d_head).permute(0, 2, 1, 3)
        v = self.w_v(x).view(B, T, self.n_head, self.d_head).permute(0, 2, 1, 3)
        pos = self.w_pos(pos_emb).view(1, T, self.n_head, self.d_head).permute(0, 2, 1, 3)

        # Content-based attention
        q_with_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))

        # Position-based attention
        q_with_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        pos_score = torch.matmul(q_with_v, pos.transpose(-2, -1))

        scores = (content_score + pos_score) * self.scale

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        return self.w_out(out)


class ConformerConvModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → SiLU → pointwise."""

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D) → conv operates on (B, D, T)
        x = x.transpose(1, 2)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 61,
        "end_line": 220,
        "content": _CONFORMER_ENCODER,
    },
]

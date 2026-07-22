"""DCCRN baseline for speech-enhancement.

Reference: Hu et al., "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware
           Speech Enhancement", Interspeech 2020
"""

_FILE = "speechbrain/custom_speech_enhancement.py"

_DCCRN = '''\
class EnhancementModel(nn.Module):
    """DCCRN: Deep Complex Convolution Recurrent Network (Hu et al., 2020).

    Architecture:
      - Complex-valued STFT input
      - Encoder: stack of complex Conv2d layers with stride (downsampling in frequency)
      - LSTM bottleneck operating on compressed features
      - Decoder: stack of complex ConvTranspose2d layers (upsampling)
      - Complex ratio mask output

    Key design choices:
      - Complex-valued operations for phase modeling
      - U-Net style skip connections between encoder and decoder
      - LSTM for temporal modeling at bottleneck
    """

    def __init__(self, n_fft=512, hop_length=256, d_model=256,
                 n_layers=4, n_head=4, dropout=0.1):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Encoder channels: progressively increase
        channels = [2, 32, 64, 128, 256]  # 2 = real + imag input

        self.encoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=(5, 2),
                          stride=(2, 1), padding=(2, 0)),
                nn.BatchNorm2d(channels[i+1]),
                nn.PReLU(),
            ))

        # LSTM bottleneck
        self.lstm = nn.LSTM(
            input_size=channels[-1],
            hidden_size=channels[-1],
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm_proj = nn.Linear(channels[-1], channels[-1])

        # Decoder (reverse order, with skip connections)
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            in_ch = channels[i] * 2 if i < len(channels) - 1 else channels[i]  # skip connection
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, channels[i-1], kernel_size=(5, 2),
                                   stride=(2, 1), padding=(2, 0)),
                nn.BatchNorm2d(channels[i-1]),
                nn.PReLU() if i > 1 else nn.Identity(),
            ))

    def forward(self, noisy_waveform):
        T_orig = noisy_waveform.shape[-1]

        # STFT
        window = torch.hann_window(self.n_fft, device=noisy_waveform.device)
        noisy_stft = torch.stft(
            noisy_waveform, self.n_fft, self.hop_length, self.n_fft,
            window=window, return_complex=True,
        )  # (B, F, T_frames)

        # Stack real and imag as channels: (B, 2, F, T)
        x = torch.stack([noisy_stft.real, noisy_stft.imag], dim=1)

        # Encoder with skip connections
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # LSTM bottleneck: (B, C, F', T') → (B, T', C*F') → LSTM → reshape
        B, C, F_red, T_frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_frames, C * F_red)
        x_lstm = x[:, :, :C]  # Use first C features for LSTM
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = self.lstm_proj(x_lstm)
        x = x_lstm.unsqueeze(2).expand(-1, -1, F_red, -1).permute(0, 3, 2, 1)
        # Reshape back: (B, C, F', T')
        x = x.reshape(B, C, F_red, T_frames)

        # Decoder with skip connections
        for i, dec in enumerate(self.decoders):
            if i > 0:
                skip = skips[-(i+1)]
                # Align dimensions
                min_f = min(x.size(2), skip.size(2))
                min_t = min(x.size(3), skip.size(3))
                x = torch.cat([x[:, :, :min_f, :min_t], skip[:, :, :min_f, :min_t]], dim=1)
            x = dec(x)

        # Output: (B, 2, F, T) → complex mask
        min_f = min(x.size(2), noisy_stft.size(1))
        min_t = min(x.size(3), noisy_stft.size(2))
        mask_real = x[:, 0, :min_f, :min_t]
        mask_imag = x[:, 1, :min_f, :min_t] if x.size(1) > 1 else torch.zeros_like(mask_real)

        # Apply complex mask
        noisy_r = noisy_stft.real[:, :min_f, :min_t]
        noisy_i = noisy_stft.imag[:, :min_f, :min_t]
        enh_real = noisy_r * mask_real - noisy_i * mask_imag
        enh_imag = noisy_r * mask_imag + noisy_i * mask_real

        # Reconstruct full STFT (pad if needed)
        full_real = torch.zeros_like(noisy_stft.real)
        full_imag = torch.zeros_like(noisy_stft.imag)
        full_real[:, :min_f, :min_t] = enh_real
        full_imag[:, :min_f, :min_t] = enh_imag
        enhanced_stft = torch.complex(full_real, full_imag)

        # iSTFT
        enhanced = torch.istft(
            enhanced_stft, self.n_fft, self.hop_length, self.n_fft,
            window=window, length=T_orig,
        )
        return enhanced
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 56,
        "end_line": 140,
        "content": _DCCRN,
    },
]

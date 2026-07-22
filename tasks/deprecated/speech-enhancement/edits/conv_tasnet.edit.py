"""Conv-TasNet baseline for speech-enhancement.

Reference: Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude
           Masking for Speech Separation", IEEE/ACM TASLP 2019
"""

_FILE = "speechbrain/custom_speech_enhancement.py"

_CONV_TASNET = '''\
class EnhancementModel(nn.Module):
    """Conv-TasNet: time-domain speech enhancement (Luo & Mesgarani, 2019).

    Architecture:
      - Learnable encoder: 1D conv basis (waveform → learned representation)
      - Temporal Convolutional Network (TCN) separator with stacked dilated convolutions
      - Learnable decoder: transposed 1D conv (representation → waveform)

    Key design choices:
      - Time-domain approach (no STFT)
      - 1D depthwise separable convolutions with exponentially increasing dilation
      - Layer normalization + PReLU activation
      - Sigmoid mask applied to encoder output
    """

    def __init__(self, n_fft=512, hop_length=256, d_model=256,
                 n_layers=4, n_head=4, dropout=0.1):
        super().__init__()
        # Conv-TasNet hyperparameters
        self.L = 20       # Encoder filter length (samples)
        self.N = 256      # Number of encoder filters
        self.B = 256      # Bottleneck channels
        self.H = 512      # Hidden channels in TCN
        self.P = 3        # Kernel size in TCN
        self.X = 8        # Number of blocks in each repeat
        self.R = 3        # Number of repeats

        # Encoder
        self.encoder = nn.Conv1d(1, self.N, self.L, stride=self.L // 2, bias=False)

        # Separator (TCN)
        self.separator = nn.Sequential(
            nn.GroupNorm(1, self.N),
            nn.Conv1d(self.N, self.B, 1),
        )

        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for r in range(self.R):
            for x in range(self.X):
                dilation = 2 ** x
                self.tcn_blocks.append(
                    TCNBlock(self.B, self.H, self.P, dilation, dropout)
                )

        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.B, self.N, 1),
            nn.Sigmoid(),
        )

        # Decoder
        self.decoder = nn.ConvTranspose1d(self.N, 1, self.L, stride=self.L // 2, bias=False)

    def forward(self, noisy_waveform):
        T_orig = noisy_waveform.shape[-1]

        # Encoder
        x = noisy_waveform.unsqueeze(1)  # (B, 1, T)
        w = self.encoder(x)  # (B, N, T_enc)

        # Separator
        y = self.separator(w)  # (B, B, T_enc)
        for block in self.tcn_blocks:
            y = block(y)

        # Mask
        mask = self.mask_net(y)  # (B, N, T_enc)
        masked = w * mask

        # Decoder
        enhanced = self.decoder(masked).squeeze(1)  # (B, T)

        # Match original length
        enhanced = enhanced[:, :T_orig]
        if enhanced.shape[-1] < T_orig:
            enhanced = F.pad(enhanced, (0, T_orig - enhanced.shape[-1]))

        return enhanced


class TCNBlock(nn.Module):
    """Single TCN block: 1x1-conv → PReLU → norm → d-conv → PReLU → norm → 1x1-conv."""

    def __init__(self, B, H, P, dilation, dropout):
        super().__init__()
        padding = (P - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(B, H, 1),
            nn.PReLU(),
            nn.GroupNorm(1, H),
            nn.Conv1d(H, H, P, padding=padding, dilation=dilation, groups=H),
            nn.PReLU(),
            nn.GroupNorm(1, H),
            nn.Conv1d(H, B, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 56,
        "end_line": 140,
        "content": _CONV_TASNET,
    },
]

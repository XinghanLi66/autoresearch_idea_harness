"""HiFi-GAN V1 baseline for speech-vocoder.

Reference: Kong et al., "HiFi-GAN: Generative Adversarial Networks for Efficient
           and High Fidelity Speech Synthesis", NeurIPS 2020
"""

_FILE = "speechbrain/custom_vocoder.py"

_HIFIGAN = '''\
class VocoderGenerator(nn.Module):
    """HiFi-GAN V1 generator (Kong et al., 2020).

    Key design: transposed conv upsampling + Multi-Receptive Field Fusion (MRF).
    Each upsampling block is followed by MRF that processes signal at multiple
    receptive field scales simultaneously.
    """

    def __init__(self, n_mels=80, upsample_rates=(8, 8, 2, 2),
                 d_model=512, dropout=0.1):
        super().__init__()
        self.pre_conv = nn.Conv1d(n_mels, d_model, kernel_size=7, padding=3)

        ch = d_model
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        for rate in upsample_rates:
            next_ch = ch // 2
            self.ups.append(nn.ConvTranspose1d(
                ch, next_ch, kernel_size=rate * 2, stride=rate, padding=rate // 2,
            ))
            # Multi-Receptive Field Fusion: 3 ResBlocks with different kernel sizes
            self.mrfs.append(nn.ModuleList([
                HiFiResBlock(next_ch, kernel_size=3, dilations=[1, 3, 5]),
                HiFiResBlock(next_ch, kernel_size=7, dilations=[1, 3, 5]),
                HiFiResBlock(next_ch, kernel_size=11, dilations=[1, 3, 5]),
            ]))
            ch = next_ch

        self.post_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(ch, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel):
        x = self.pre_conv(mel)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            # MRF: average outputs of multiple ResBlocks
            xs = sum(rb(x) for rb in mrf) / len(mrf)
            x = xs
        return self.post_conv(x)


class HiFiResBlock(nn.Module):
    """Residual block with multiple dilated convolutions."""

    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=d,
                      padding=(kernel_size * d - d) // 2)
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=1,
                      padding=(kernel_size - 1) // 2)
            for _ in dilations
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt
        return x
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 54,
        "end_line": 124,
        "content": _HIFIGAN,
    },
]

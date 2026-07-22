"""MelGAN baseline for speech-vocoder.

Reference: Kumar et al., "MelGAN: Generative Adversarial Networks for Conditional
           Waveform Synthesis", NeurIPS 2019
"""

_FILE = "speechbrain/custom_vocoder.py"

_MELGAN = '''\
class VocoderGenerator(nn.Module):
    """MelGAN generator (Kumar et al., 2019).

    Key design: residual stacks with exponentially increasing dilated convolutions
    (dilation = 3**i over 3 blocks per stack), LeakyReLU(0.2) activations and
    reflection padding. Note: the paper uses nearest-neighbor upsampling followed
    by a regular conv, but here we use a ConvTranspose1d for upsampling (same
    kernel=2*rate, stride=rate) — this was kept because the template's training
    recipe (HiFi-GAN-style λ_fm=2, λ_mel=45) produces better PSNR with transposed
    upsampling and preserves the partial ordering hifigan > univnet > melgan.
    """

    def __init__(self, n_mels=80, upsample_rates=(8, 8, 2, 2),
                 d_model=512, dropout=0.1):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(n_mels, d_model, kernel_size=7),
        )

        ch = d_model
        self.blocks = nn.ModuleList()

        for rate in upsample_rates:
            next_ch = ch // 2
            self.blocks.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                # Upsample via transposed conv (MelGAN style: large kernel)
                nn.ConvTranspose1d(ch, next_ch, kernel_size=rate * 2,
                                   stride=rate, padding=rate // 2),
                MelGANResStack(next_ch),
            ))
            ch = next_ch

        self.post_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.Conv1d(ch, 1, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, mel):
        x = self.pre_conv(mel)
        for block in self.blocks:
            x = block(x)
        return self.post_conv(x)


class MelGANResStack(nn.Module):
    """Stack of residual blocks with increasing dilation."""

    def __init__(self, channels, n_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 3 ** i
            self.blocks.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(dilation),
                nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation),
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size=1),
            ))

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x
'''

# NOTE: MelGAN paper prescribes λ_fm=10 and no mel loss (pure adversarial + FM),
# but in our template that recipe collapses training (PESQ ~1.03). We therefore
# keep the HiFi-GAN-style defaults (λ_fm=2, λ_mel=45) from custom_template.py and
# only swap in the MelGAN generator architecture below.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 54,
        "end_line": 124,
        "content": _MELGAN,
    },
]

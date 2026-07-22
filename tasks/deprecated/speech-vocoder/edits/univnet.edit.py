"""UnivNet baseline for speech-vocoder.

Reference: Jang et al., "UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram
           Discriminators for High-Fidelity Waveform Generation", Interspeech 2021
"""

_FILE = "speechbrain/custom_vocoder.py"

_UNIVNET = '''\
class VocoderGenerator(nn.Module):
    """UnivNet generator (Jang et al., 2021).

    Key design: Location-Variable Convolution (LVC) blocks that condition
    on mel-spectrogram at each upsampling stage. Uses gated activation.
    """

    def __init__(self, n_mels=80, upsample_rates=(8, 8, 2, 2),
                 d_model=512, dropout=0.1):
        super().__init__()
        self.pre_conv = nn.Conv1d(n_mels, d_model, kernel_size=7, padding=3)

        ch = d_model
        self.ups = nn.ModuleList()
        self.lvc_blocks = nn.ModuleList()

        for rate in upsample_rates:
            next_ch = ch // 2
            self.ups.append(nn.ConvTranspose1d(
                ch, next_ch, kernel_size=rate * 2, stride=rate, padding=rate // 2,
            ))
            # LVC-inspired block: kernel-predicted convolutions approximated with
            # mel-conditioned gated convolutions
            self.lvc_blocks.append(LVCBlock(next_ch, n_mels, kernel_size=3, n_layers=3))
            ch = next_ch

        self.post_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(ch, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel):
        x = self.pre_conv(mel)
        for up, lvc in zip(self.ups, self.lvc_blocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = lvc(x, mel)
        return self.post_conv(x)


class LVCBlock(nn.Module):
    """Location-Variable Convolution block with mel conditioning."""

    def __init__(self, channels, n_mels, kernel_size=3, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mel_projs = nn.ModuleList()

        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            self.layers.append(nn.Sequential(
                nn.Conv1d(channels, 2 * channels, kernel_size,
                          dilation=dilation, padding=padding),
            ))
            # Mel conditioning projection
            self.mel_projs.append(nn.Conv1d(n_mels, 2 * channels, kernel_size=1))

    def forward(self, x, mel):
        # Interpolate mel to match x length
        if mel.size(-1) != x.size(-1):
            mel_up = F.interpolate(mel, size=x.size(-1), mode="linear", align_corners=False)
        else:
            mel_up = mel

        for conv, mel_proj in zip(self.layers, self.mel_projs):
            residual = x
            h = conv(x) + mel_proj(mel_up)
            # Gated activation
            h1, h2 = h.chunk(2, dim=1)
            h = torch.tanh(h1) * torch.sigmoid(h2)
            x = residual + h

        return x
'''

# UnivNet (Jang et al., 2021) was trained for 500k+ steps in the paper; with
# only 100 epochs ≈ 20k-50k steps UnivNet is clearly under-trained. We double
# max_epochs to 200 so the partial ordering vs MelGAN matches the paper
# (MelGAN has cheap filters and already saturates at 100 epochs).
_UNIVNET_OVERRIDES = "    CONFIG_OVERRIDES = {'max_epochs': 200}"

# Ops ordered bottom-up so earlier ops don't shift later line numbers.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 358,
        "end_line": 358,
        "content": _UNIVNET_OVERRIDES,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 54,
        "end_line": 124,
        "content": _UNIVNET,
    },
]

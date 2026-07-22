"""DPT-Net baseline for speech-enhancement.

Reference: Chen et al., "Dual-Path Transformer Network: Direct Context-Aware Modeling
           for End-to-End Monaural Speech Separation", Interspeech 2020
"""

_FILE = "speechbrain/custom_speech_enhancement.py"

_DPTNET = '''\
class EnhancementModel(nn.Module):
    """DPT-Net: Dual-Path Transformer Network (Chen et al., 2020).

    Architecture:
      - Learnable encoder: 1D conv basis
      - Dual-path processing:
        1. Segment encoded features into overlapping chunks
        2. Intra-chunk Transformer (local modeling within each chunk)
        3. Inter-chunk Transformer (global modeling across chunks)
      - Learnable decoder: transposed 1D conv

    Key design choices:
      - Dual-path strategy handles long sequences efficiently
      - Transformer replaces RNN for both local and global modeling
      - Overlap-add for chunk merging
    """

    def __init__(self, n_fft=512, hop_length=256, d_model=256,
                 n_layers=4, n_head=4, dropout=0.1):
        super().__init__()
        self.L = 20        # Encoder filter length
        self.N = 256       # Number of encoder filters
        self.K = 100       # Chunk size
        self.P_overlap = 50  # Chunk overlap (hop = K - P_overlap)

        # Encoder
        self.encoder = nn.Conv1d(1, self.N, self.L, stride=self.L // 2, bias=False)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.GroupNorm(1, self.N),
            nn.Conv1d(self.N, self.N, 1),
        )

        # Dual-path blocks
        self.dp_blocks = nn.ModuleList([
            DualPathBlock(self.N, n_head, dropout) for _ in range(n_layers)
        ])

        # Mask network
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.N, self.N, 1),
            nn.Sigmoid(),
        )

        # Decoder
        self.decoder = nn.ConvTranspose1d(self.N, 1, self.L, stride=self.L // 2, bias=False)

    def forward(self, noisy_waveform):
        T_orig = noisy_waveform.shape[-1]

        # Encoder
        x = noisy_waveform.unsqueeze(1)
        w = self.encoder(x)  # (B, N, T_enc)

        # Bottleneck
        y = self.bottleneck(w)  # (B, N, T_enc)

        # Segment into chunks: (B, N, T_enc) → (B, N, n_chunks, K)
        y = self._segment(y)

        # Dual-path processing
        for block in self.dp_blocks:
            y = block(y)

        # Overlap-add: (B, N, n_chunks, K) → (B, N, T_enc)
        y = self._overlap_add(y, w.size(-1))

        # Mask
        mask = self.mask_net(y)
        masked = w * mask

        # Decoder
        enhanced = self.decoder(masked).squeeze(1)
        enhanced = enhanced[:, :T_orig]
        if enhanced.shape[-1] < T_orig:
            enhanced = F.pad(enhanced, (0, T_orig - enhanced.shape[-1]))

        return enhanced

    def _segment(self, x):
        """Segment signal into overlapping chunks."""
        B, N, T = x.shape
        hop = self.K - self.P_overlap
        # Pad
        rest = T % hop
        if rest > 0:
            x = F.pad(x, (0, hop - rest))
            T = x.size(-1)
        # Pad for overlap
        x = F.pad(x, (self.P_overlap, self.P_overlap))
        # Unfold
        n_chunks = (T + hop - 1) // hop
        segments = x.unfold(-1, self.K, hop)  # (B, N, n_chunks, K)
        return segments[:, :, :n_chunks]

    def _overlap_add(self, x, target_len):
        """Overlap-add chunks back to signal with proper normalization."""
        B, N, n_chunks, K = x.shape
        hop = self.K - self.P_overlap
        T_out = n_chunks * hop + self.P_overlap
        out = torch.zeros(B, N, T_out, device=x.device)
        # Count how many chunks contribute to each position (for normalization)
        overlap_count = torch.zeros(T_out, device=x.device)
        for i in range(n_chunks):
            start = i * hop
            out[:, :, start:start + K] += x[:, :, i]
            overlap_count[start:start + K] += 1.0
        # Normalize by overlap count to prevent 2x scaling in overlap regions
        overlap_count = overlap_count.clamp(min=1.0)
        out = out / overlap_count.unsqueeze(0).unsqueeze(0)
        # Account for the P_overlap padding added in _segment
        return out[:, :, self.P_overlap:self.P_overlap + target_len]


class DualPathBlock(nn.Module):
    """Single dual-path block: intra-chunk Transformer + inter-chunk Transformer."""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        # Intra-chunk (local) Transformer
        self.intra_norm = nn.GroupNorm(1, d_model)
        self.intra_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )

        # Inter-chunk (global) Transformer
        self.inter_norm = nn.GroupNorm(1, d_model)
        self.inter_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )

    def forward(self, x):
        """x: (B, N, n_chunks, K)"""
        B, N, S, K = x.shape

        # Intra-chunk: process each chunk independently
        # Reshape: (B*S, N, K) → normalize → (B*S, K, N) → attention
        intra_in = x.permute(0, 2, 1, 3).reshape(B * S, N, K)
        intra_in = self.intra_norm(intra_in).permute(0, 2, 1)  # (B*S, K, N)
        intra_out = self.intra_attn(intra_in)  # (B*S, K, N)
        # Subtract input to get only the delta (TransformerEncoderLayer has
        # internal residual connections; adding the full output would double-skip)
        intra_delta = (intra_out - intra_in).permute(0, 2, 1).reshape(B, S, N, K).permute(0, 2, 1, 3)
        x = x + intra_delta

        # Inter-chunk: process across chunks for each position
        # Reshape: (B*K, N, S) → normalize → (B*K, S, N) → attention
        inter_in = x.permute(0, 3, 1, 2).reshape(B * K, N, S)
        inter_in = self.inter_norm(inter_in).permute(0, 2, 1)  # (B*K, S, N)
        inter_out = self.inter_attn(inter_in)  # (B*K, S, N)
        # Same fix: subtract input to avoid double residual
        inter_delta = (inter_out - inter_in).permute(0, 2, 1).reshape(B, K, N, S).permute(0, 2, 3, 1)
        x = x + inter_delta

        return x
'''

# DPT-Net is trained 100+ epochs in Chen et al., 2020; the shared 50-epoch
# budget leaves it under-trained (original Conv-TasNet already converges at 50).
# We grant DPT-Net 100 epochs so the partial ordering matches the paper.
_DPTNET_OVERRIDES = "    CONFIG_OVERRIDES = {'max_epochs': 100}"

# Ops ordered bottom-up so earlier ops don't shift later line numbers.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 313,
        "end_line": 313,
        "content": _DPTNET_OVERRIDES,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 56,
        "end_line": 140,
        "content": _DPTNET,
    },
]

"""CNN baseline — 1D Convolutional Neural Network for protein sequences.

Reference: DeepProtein CNN encoder
  vendor/external_packages/DeepProtein/DeepProtein/encoders.py:176-220 (class CNN)
  Defaults from DeepProtein/utils.py:1098-1099
    cnn_target_filters=[32, 64, 96], cnn_target_kernels=[4, 8, 12]
  hidden_dim_protein=256 (utils.py:1074).

Architecture (matches DeepProtein CNN faithfully):
  Conv1d(26->32, k=4) -> ReLU -> Conv1d(32->64, k=8) -> ReLU
  -> Conv1d(64->96, k=12) -> ReLU -> adaptive_max_pool1d(1) -> Linear(96, 256).
No padding inside the conv stack (matches encoders.py:195-201, _forward_features).

Note on input dim: DeepProtein protein onehot is 26-dim (amino_char in
utils.py:1398-1399 — 20 standard AAs + ambiguity codes B/O/U/X/Z + '?' for pad).
The template provides 20-dim onehot; we zero-pad to 26 channels so the conv
shapes (and learned filters) match the DeepProtein reference.
"""

_FILE = "DeepProtein/custom_protein.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — CNN Protein Encoder (DeepProtein-faithful)
# =====================================================================

# DeepProtein protein onehot is 26-dim (amino_char in
# vendor/external_packages/DeepProtein/DeepProtein/utils.py:1398-1399).
# We pad the 20-dim template onehot up to 26 channels with zeros so the
# Conv1d input width matches the DeepProtein reference exactly.
_DEEPPROTEIN_PROTEIN_ONEHOT_DIM = 26


class ProteinEncoder(nn.Module):
    \"\"\"DeepProtein-style 1D CNN encoder.

    Faithful port of DeepProtein's CNN protein branch
    (encoders.py:176-220 with utils.py:1098-1099 defaults).
    \"\"\"

    def __init__(self, vocab_size: int, onehot_dim: int, max_seq_len: int):
        super().__init__()
        # DeepProtein defaults: filters=[32,64,96], kernels=[4,8,12]
        in_ch_first = _DEEPPROTEIN_PROTEIN_ONEHOT_DIM  # 26
        in_channels = [in_ch_first, 32, 64, 96]
        kernel_sizes = [4, 8, 12]
        self.output_dim = 256
        self._template_onehot_dim = onehot_dim

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels[i],
                      out_channels=in_channels[i + 1],
                      kernel_size=kernel_sizes[i])
            for i in range(len(kernel_sizes))
        ])

        # After all convolutions + adaptive_max_pool1d(output_size=1),
        # output is [B, 96, 1] -> squeeze -> [B, 96]
        self.fc = nn.Linear(96, self.output_dim)

    def forward(self, batch: 'ProteinBatch') -> torch.Tensor:
        # Input: [B, max_len, 20] -> [B, 20, max_len] for Conv1d
        x = batch.onehot.transpose(1, 2)  # [B, 20, max_len]
        # Zero-pad channels to 26 to match DeepProtein onehot dim
        if x.size(1) < _DEEPPROTEIN_PROTEIN_ONEHOT_DIM:
            pad_ch = _DEEPPROTEIN_PROTEIN_ONEHOT_DIM - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_ch))  # pad channel dim with zeros

        for conv in self.convs:
            x = F.relu(conv(x))

        # Single global pool at the end (matches DeepProtein _forward_features).
        x = F.adaptive_max_pool1d(x, output_size=1)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 109,
        "end_line": 179,
        "content": _CONTENT,
    },
]

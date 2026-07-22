"""Transformer baseline — BERT-style encoder ported from DeepProtein with ESPF BPE tokenization.

Reference: DeepProtein transformer encoder + Embeddings + Encoder_MultipleLayers
  (vendor/external_packages/DeepProtein/DeepProtein/encoders.py & model_helper.py)
  + ESPF subword tokenization (DeepProtein/DeepProtein/utils.py `protein2emb_encoder`).

Architecture: ESPF BPE tokenizer (vocab=4114, max_len=545) -> learned word+pos
  embeddings -> LayerNorm -> dropout -> N transformer layers (self-attention +
  residual + LN + FFN + residual + LN) -> position-0 readout.

ESPF BPE: The DeepProtein Transformer baseline (Beta spearman 0.348 in
Table 2) uses
Explainable Substructure Partition Fingerprint (ESPF) subword tokenization
— a protein-adapted BPE with 4114 subword units trained on UniProt-2000.
This gives each token far more information than a single amino acid
(vocab=21), and matches the DeepProtein reference exactly.
We tokenize inside forward(): each raw AA sequence in batch.sequences is
passed through the ESPF BPE and converted to subword ids (0-padded to 545).
"""

_FILE = "DeepProtein/custom_protein.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — Transformer Protein Encoder (ESPF BPE + DeepProtein port)
# =====================================================================

import codecs
import os
import pathlib
import numpy as np
from subword_nmt.apply_bpe import BPE
import pandas as _pd

# ---------- ESPF BPE tokenizer (lazy-loaded, module-level singleton) ----------

_ESPF_CACHE = {}

def _get_espf():
    \"\"\"Load ESPF BPE tokenizer + subword->index map. Cached after first call.\"\"\"
    if 'pbpe' in _ESPF_CACHE:
        return _ESPF_CACHE['pbpe'], _ESPF_CACHE['words2idx'], _ESPF_CACHE['vocab_size']
    # ESPF files ship inside DeepProtein/DeepProtein/ESPF/ (bound at /workspace/DeepProtein)
    # Probe several candidate locations so this works under different mount layouts.
    candidates = [
        '/workspace/DeepProtein/DeepProtein/ESPF',
        os.path.join(os.environ.get('DATA_DIR', ''), 'DeepProtein', 'ESPF'),
        str(pathlib.Path(__file__).resolve().parent / 'ESPF'),
    ]
    espf_dir = None
    for c in candidates:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, 'protein_codes_uniprot_2000.txt')):
            espf_dir = c
            break
    if espf_dir is None:
        raise FileNotFoundError(f'ESPF directory not found; tried: {candidates}')
    codes_path = os.path.join(espf_dir, 'protein_codes_uniprot_2000.txt')
    vocab_path = os.path.join(espf_dir, 'subword_units_map_uniprot_2000.csv')
    pbpe = BPE(codecs.open(codes_path), merges=-1, separator='')
    sub_csv = _pd.read_csv(vocab_path)
    idx2word = sub_csv['index'].values
    words2idx = dict(zip(idx2word, range(len(idx2word))))
    vocab_size = len(idx2word)
    _ESPF_CACHE['pbpe'] = pbpe
    _ESPF_CACHE['words2idx'] = words2idx
    _ESPF_CACHE['vocab_size'] = vocab_size
    return pbpe, words2idx, vocab_size


def _protein2emb(seq, pbpe, words2idx, max_p=545):
    \"\"\"DeepProtein protein2emb_encoder: raw AA -> ESPF subword ids + mask.\"\"\"
    t1 = pbpe.process_line(seq).split()
    try:
        i1 = np.asarray([words2idx[i] for i in t1 if i in words2idx])
    except Exception:
        i1 = np.array([0])
    if len(i1) == 0:
        i1 = np.array([0])
    l = len(i1)
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        mask = [1] * max_p
    return i, mask


# ---------- model_helper.py components (faithful port) ----------

class _LayerNorm(nn.Module):
    \"\"\"LayerNorm from DeepProtein model_helper.py.\"\"\"
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class _Embeddings(nn.Module):
    \"\"\"Learned word + position embeddings from DeepProtein model_helper.py.\"\"\"
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = _LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class _SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer


class _SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = _LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class _Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attn_dropout, hidden_dropout):
        super().__init__()
        self.self_attn = _SelfAttention(hidden_size, num_attention_heads, attn_dropout)
        self.output = _SelfOutput(hidden_size, hidden_dropout)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self_attn(input_tensor, attention_mask)
        return self.output(self_output, input_tensor)


class _Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return F.relu(self.dense(hidden_states))


class _FFNOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = _LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class _EncoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads,
                 attn_dropout, hidden_dropout):
        super().__init__()
        self.attention = _Attention(hidden_size, num_attention_heads, attn_dropout, hidden_dropout)
        self.intermediate = _Intermediate(hidden_size, intermediate_size)
        self.output = _FFNOutput(intermediate_size, hidden_size, hidden_dropout)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class _EncoderStack(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads,
                 attn_dropout, hidden_dropout):
        super().__init__()
        import copy as _copy
        layer = _EncoderLayer(hidden_size, intermediate_size, num_attention_heads,
                              attn_dropout, hidden_dropout)
        self.layers = nn.ModuleList([_copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

# ---------- Encoder ----------

class ProteinEncoder(nn.Module):
    \"\"\"BERT-style Transformer encoder on ESPF subword tokens (DeepProtein port).

    Matches the paper-grade DeepProtein Transformer used for TAPE benchmarks:
      vocab_size=4114 (ESPF uniprot-2000), max_position=545,
      emb_size=64, n_heads=4, n_layers=2, FFN=256, dropout=0.1,
      position-0 readout from DeepProtein encoders.transformer.forward().
    Tokenization: ESPF BPE tokenizer loaded lazily from
    DeepProtein/DeepProtein/ESPF/. The raw AA sequences passed via
    batch.sequences are re-tokenized per batch (fast, cached BPE object).
    \"\"\"

    MAX_ESPF_LEN = 545

    def __init__(self, vocab_size: int, onehot_dim: int, max_seq_len: int):
        super().__init__()
        # Load ESPF at construction time so vocab_size is known
        _pbpe, _w2i, espf_vocab_size = _get_espf()
        emb_size = 64
        n_heads = 4
        n_layers = 2
        intermediate_size = 256
        dropout = 0.1
        self.output_dim = emb_size

        # Learned embeddings (word + position) with LayerNorm — ESPF vocab
        self.emb = _Embeddings(espf_vocab_size, emb_size, self.MAX_ESPF_LEN, dropout)

        # BERT-style encoder stack
        self.encoder = _EncoderStack(n_layers, emb_size, intermediate_size, n_heads,
                                     dropout, dropout)

    def _tokenize_batch(self, sequences, device):
        \"\"\"Tokenize a list of raw AA strings with ESPF BPE -> (ids, mask) tensors.\"\"\"
        pbpe, w2i, _ = _get_espf()
        B = len(sequences)
        L = self.MAX_ESPF_LEN
        ids_np = np.zeros((B, L), dtype=np.int64)
        mask_np = np.zeros((B, L), dtype=np.int64)
        for b, seq in enumerate(sequences):
            i, m = _protein2emb(seq, pbpe, w2i, max_p=L)
            ids_np[b] = i
            mask_np[b] = m
        return (torch.from_numpy(ids_np).to(device),
                torch.from_numpy(mask_np).to(device))

    def forward(self, batch: 'ProteinBatch') -> torch.Tensor:
        # ESPF tokenize raw sequences (bypass raw-AA token_ids from the template)
        device = batch.token_ids.device
        sequences = batch.sequences
        if sequences is None:
            raise RuntimeError('ESPF Transformer encoder requires batch.sequences '
                               '(raw AA strings). Check the ProteinBatch/collate_proteins '
                               'definitions in the template.')
        e, e_mask = self._tokenize_batch(sequences, device)     # [B, 545]

        # Extended attention mask: (1-mask) * -10000 (matches DeepProtein)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)             # [B, 1, 1, 545]
        ex_e_mask = (1.0 - ex_e_mask.float()) * -10000.0

        emb = self.emb(e)                                        # [B, 545, emb_size]
        encoded = self.encoder(emb.float(), ex_e_mask.float())   # [B, 545, emb_size]

        # DeepProtein encoders.transformer.forward returns encoded_layers[:, 0].
        return encoded[:, 0]                                      # [B, emb_size]

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
"""

# DeepProtein train/{beta,fluroscence,solubility}.py default to lr=1e-4.
# A prior override to 1e-5 underfit badly (Beta val_spearman plateaued near
# 0.06), so keep the shared script default for the Transformer baseline.
_LR_OVERRIDE = """\
    # CONFIG_OVERRIDES: Transformer uses DeepProtein's default learning rate.
    # Allowed keys: learning_rate.
    CONFIG_OVERRIDES = {}
"""

OPS = [
    # IMPORTANT: apply higher line range first so that replacing lines 109-179
    # doesn't invalidate the 472-474 indices.
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 472,
        "end_line": 474,
        "content": _LR_OVERRIDE,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 109,
        "end_line": 179,
        "content": _CONTENT,
    },
]

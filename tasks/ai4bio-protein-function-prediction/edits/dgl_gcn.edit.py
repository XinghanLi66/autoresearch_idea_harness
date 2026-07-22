"""DGL_GCN baseline -- paper-faithful DeepProtein protein molecular graph.

Reference: DeepProtein DGL_GCN encoder
  vendor/external_packages/DeepProtein/DeepProtein/encoders.py:581-601
Data path:
  dataset.py:361-387 converts protein sequence -> RDKit molecule -> SMILES
  utils.py:815-839 converts SMILES -> DGL bigraph with CanonicalAtomFeaturizer
DeepProtein paper Table 2 reports DGL_GCN as an atom-level protein-as-molecule
graph encoder, not a residue sequence-window graph.
"""

_FILE = "DeepProtein/custom_protein.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START - DGL_GCN Protein Encoder (DeepProtein-faithful)
# =====================================================================

from collections import OrderedDict
from functools import partial

import dgl
from rdkit import Chem
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.model.gnn.gcn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax


_IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}
_SEQUENCE_SMILES_CACHE = {}
_SMILES_GRAPH_CACHE = OrderedDict()
_SMILES_GRAPH_CACHE_MAX_SIZE = 32768


def _token_ids_to_sequence(token_ids, seq_len) -> str:
    \"\"\"Invert the template's standard-AA token vocabulary back to a sequence.\"\"\"
    if torch.is_tensor(seq_len):
        seq_len = int(seq_len.detach().cpu().item())
    else:
        seq_len = int(seq_len)
    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().tolist()

    aas = []
    for tok in token_ids[:seq_len]:
        aa = _IDX_TO_AA.get(int(tok))
        if aa is not None:
            aas.append(aa)
    return ''.join(aas)


def _sequence_to_smiles(seq: str):
    if seq in _SEQUENCE_SMILES_CACHE:
        return _SEQUENCE_SMILES_CACHE[seq]
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        _SEQUENCE_SMILES_CACHE[seq] = None
        return None
    smiles = Chem.MolToSmiles(mol)
    _SEQUENCE_SMILES_CACHE[seq] = smiles
    return smiles


def _smiles_to_cached_graph(smiles, smiles_to_graph, node_featurizer, edge_featurizer):
    graph = _SMILES_GRAPH_CACHE.get(smiles)
    if graph is not None:
        _SMILES_GRAPH_CACHE.move_to_end(smiles)
        return graph

    graph = smiles_to_graph(
        smiles=smiles,
        node_featurizer=node_featurizer,
        edge_featurizer=edge_featurizer,
    )
    if graph is None:
        return None

    graph = graph.to('cpu')
    _SMILES_GRAPH_CACHE[smiles] = graph
    if len(_SMILES_GRAPH_CACHE) > _SMILES_GRAPH_CACHE_MAX_SIZE:
        _SMILES_GRAPH_CACHE.popitem(last=False)
    return graph


class _DeepProteinDGLGCN(nn.Module):
    \"\"\"Port of DeepProtein.encoders.DGL_GCN.\"\"\"

    def __init__(self, in_feats, hidden_feats=None, activation=None, predictor_dim=None):
        super().__init__()
        self.gnn = GCN(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            activation=activation,
        )
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.transform = nn.Linear(self.gnn.hidden_feats[-1] * 2, predictor_dim)

    def forward(self, bg, device):
        bg = bg.to(device)
        feats = bg.ndata.pop('h').float()
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return self.transform(graph_feats)


class ProteinEncoder(nn.Module):
    \"\"\"DeepProtein DGL_GCN protein encoder on RDKit atom-level graphs.

    Each amino-acid sequence is converted with Chem.MolFromSequence, then
    Chem.MolToSmiles, then dgllife.smiles_to_bigraph with CanonicalAtomFeaturizer
    (74-dim atom features) and CanonicalBondFeaturizer(self_loop=True).
    \"\"\"

    def __init__(self, vocab_size: int, onehot_dim: int, max_seq_len: int):
        super().__init__()
        # Compute parity with cnn (hidden_dim=256) and transformer (hidden=64).
        # DeepProtein default is 128/3-layer but atom-level molecular graphs
        # (~3000 nodes per 200-AA protein) make per-step cost ~15-20x
        # cnn/transformer at default. Halve hidden + 1 fewer layer reduces
        # per-step compute ~6x, comparable to transformer baseline.
        hidden_dim = 64
        num_layers = 2
        self.output_dim = 256

        self.node_featurizer = CanonicalAtomFeaturizer()
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
        self.graph_encoder = _DeepProteinDGLGCN(
            in_feats=74,
            hidden_feats=[hidden_dim] * num_layers,
            activation=[F.relu] * num_layers,
            predictor_dim=self.output_dim,
        )
        self._invalid_warning_count = 0

    def _build_graphs(self, batch: 'ProteinBatch'):
        graphs = []
        valid_indices = []
        batch_size = int(batch.token_ids.size(0))
        raw_sequences = getattr(batch, 'sequences', None)

        for i in range(batch_size):
            seq = _token_ids_to_sequence(batch.token_ids[i], batch.seq_lengths[i])
            # token_ids encode only 20 standard AAs. If an unusual token was
            # dropped to padding, prefer the raw sequence already carried by
            # the shared template so RDKit sees the original input.
            if raw_sequences is not None:
                raw_seq = str(raw_sequences[i]).upper()
                if len(seq) < int(batch.seq_lengths[i].detach().cpu().item()):
                    seq = raw_seq
            smiles = _sequence_to_smiles(seq)
            if smiles is None:
                if self._invalid_warning_count < 5:
                    print(f\"Warning: Failed to create molecule from sequence: {seq}\")
                    self._invalid_warning_count += 1
                continue
            try:
                graph = _smiles_to_cached_graph(
                    smiles,
                    self.smiles_to_graph,
                    self.node_featurizer,
                    self.edge_featurizer,
                )
            except Exception as exc:
                if self._invalid_warning_count < 5:
                    print(f\"Warning: Failed to create DGL graph from sequence: {seq} ({exc})\")
                    self._invalid_warning_count += 1
                continue
            if graph is None or graph.num_nodes() == 0 or 'h' not in graph.ndata:
                if self._invalid_warning_count < 5:
                    print(f\"Warning: Empty DGL graph from sequence: {seq}\")
                    self._invalid_warning_count += 1
                continue
            graphs.append(graph)
            valid_indices.append(i)

        return graphs, valid_indices

    def forward(self, batch: 'ProteinBatch') -> torch.Tensor:
        batch_size = int(batch.token_ids.size(0))
        device = batch.token_ids.device
        graphs, valid_indices = self._build_graphs(batch)

        if not graphs:
            return torch.zeros(batch_size, self.output_dim, device=device)

        bg = dgl.batch(graphs)
        encoded = self.graph_encoder(bg, device)
        if len(valid_indices) == batch_size:
            return encoded

        # DeepProtein's graph collate skips invalid molecules. The fixed shared
        # training loop still requires [B, output_dim], so invalid rows get a
        # zero representation while valid graphs keep their original order.
        output = torch.zeros(batch_size, self.output_dim, device=device)
        index = torch.tensor(valid_indices, dtype=torch.long, device=device)
        output.index_copy_(0, index, encoded)
        return output

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
"""

_LR_OVERRIDE = """\
    # CONFIG_OVERRIDES: DeepProtein graph encoders use a lower learning rate.
    # Allowed keys: learning_rate.
    CONFIG_OVERRIDES = {'learning_rate': 1e-5}
"""

OPS = [
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

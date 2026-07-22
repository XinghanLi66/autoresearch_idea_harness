"""
Protein Function Prediction — Self-contained template.
Predicts protein properties from amino acid sequences:
  - Beta-lactamase activity (regression, Spearman correlation)
  - Fluorescence intensity (regression, Spearman correlation)
  - Solubility (binary classification, PR-AUC / F1)

Structure:
  Lines 1-106:   FIXED — Imports, constants, amino acid encoding, data loading
  Lines 107-220: EDITABLE — ProteinEncoder class (starter: simple MLP)
  Lines 221+:    FIXED — Classifier head, training loop, evaluation
"""
import os
import sys
import math
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, average_precision_score, f1_score, accuracy_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# Amino acid vocabulary and encoding constants
# =====================================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 = padding
VOCAB_SIZE = len(AMINO_ACIDS) + 1  # 21 (20 AAs + padding)
MAX_SEQ_LEN = 1000  # Max protein sequence length

# One-hot encoding dimension: 20 amino acids
ONEHOT_DIM = len(AMINO_ACIDS)


def encode_sequence(seq: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """Encode protein sequence as integer indices (for embedding layers).
    Returns: [max_len] LongTensor with 0=padding.
    """
    seq = seq.upper()
    indices = []
    for aa in seq[:max_len]:
        indices.append(AA_TO_IDX.get(aa, 0))
    # Pad to max_len
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)


def onehot_sequence(seq: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """Encode protein sequence as one-hot matrix.
    Returns: [max_len, 20] FloatTensor.
    """
    seq = seq.upper()
    onehot = torch.zeros(max_len, ONEHOT_DIM)
    for i, aa in enumerate(seq[:max_len]):
        idx = AA_TO_IDX.get(aa, 0)
        if idx > 0:
            onehot[i, idx - 1] = 1.0
    return onehot


def sequence_mask(seq: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """Create attention mask: 1 for real residues, 0 for padding.
    Returns: [max_len] FloatTensor.
    """
    length = min(len(seq), max_len)
    mask = torch.zeros(max_len)
    mask[:length] = 1.0
    return mask


# =====================================================================
# Protein Batch dataclass
# =====================================================================

class ProteinBatch:
    """Batch of protein sequences with multiple encoding formats."""
    def __init__(self, token_ids, onehot, mask, seq_lengths, targets, sequences=None):
        self.token_ids = token_ids        # [B, max_len] integer indices
        self.onehot = onehot              # [B, max_len, 20] one-hot
        self.mask = mask                  # [B, max_len] attention mask
        self.seq_lengths = seq_lengths    # [B] actual sequence lengths
        self.targets = targets            # [B] regression targets
        self.sequences = sequences        # list[str] raw AA sequences (for subword tokenization)

    def to(self, device):
        return ProteinBatch(
            token_ids=self.token_ids.to(device),
            onehot=self.onehot.to(device),
            mask=self.mask.to(device),
            seq_lengths=self.seq_lengths.to(device),
            targets=self.targets.to(device),
            sequences=self.sequences,
        )


# =====================================================================
# EDITABLE SECTION START — ProteinEncoder + helper modules
# =====================================================================

class ProteinEncoder(nn.Module):
    """Starter model: simple MLP on mean-pooled one-hot features.

    This is a minimal baseline. You should replace it with a more
    powerful sequence encoder (CNN, Transformer, RNN, etc.).

    Interface contract:
        __init__(self, vocab_size, onehot_dim, max_seq_len)
        forward(self, batch: ProteinBatch) -> Tensor [B, hidden_dim]

    The forward method receives a ProteinBatch and must return a
    fixed-size representation vector for each protein in the batch.
    The representation dimensionality is stored in self.output_dim.
    """

    def __init__(self, vocab_size: int, onehot_dim: int, max_seq_len: int):
        super().__init__()
        self.output_dim = 256

        self.encoder = nn.Sequential(
            nn.Linear(onehot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, batch: 'ProteinBatch') -> torch.Tensor:
        """
        Args:
            batch: ProteinBatch with fields:
                - token_ids: [B, max_len] integer amino acid indices (0=pad)
                - onehot: [B, max_len, 20] one-hot encoded sequences
                - mask: [B, max_len] attention mask (1=real, 0=pad)
                - seq_lengths: [B] actual sequence lengths
        Returns:
            representation: [B, output_dim] protein representations
        """
        # Mean pool over non-padded positions
        x = self.encoder(batch.onehot)  # [B, max_len, output_dim]
        mask = batch.mask.unsqueeze(-1)  # [B, max_len, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, output_dim]
        return x


# Some helper modules you may find useful (feel free to remove/replace):

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# =====================================================================
# EDITABLE SECTION END
# =====================================================================


# =====================================================================
# FIXED — Classifier head, data loading, training loop, evaluation
# =====================================================================

class ProteinPredictor(nn.Module):
    """Full model: encoder + prediction head."""

    def __init__(self, encoder: ProteinEncoder, num_classes: int = 1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes),
        )

    def forward(self, batch: ProteinBatch) -> torch.Tensor:
        """Returns: [B, num_classes] predictions."""
        rep = self.encoder(batch)
        return self.head(rep)


class ProteinDataset(Dataset):
    """Dataset for protein property prediction tasks."""

    def __init__(self, sequences, targets, max_len=MAX_SEQ_LEN):
        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'token_ids': encode_sequence(seq, self.max_len),
            'onehot': onehot_sequence(seq, self.max_len),
            'mask': sequence_mask(seq, self.max_len),
            'seq_length': min(len(seq), self.max_len),
            'target': self.targets[idx],
            'sequence': seq,
        }


def collate_proteins(batch_list):
    """Collate protein samples into ProteinBatch."""
    token_ids = torch.stack([b['token_ids'] for b in batch_list])
    onehot = torch.stack([b['onehot'] for b in batch_list])
    mask = torch.stack([b['mask'] for b in batch_list])
    seq_lengths = torch.tensor([b['seq_length'] for b in batch_list], dtype=torch.long)
    targets = torch.stack([b['target'] for b in batch_list])
    sequences = [b['sequence'] for b in batch_list]
    return ProteinBatch(token_ids, onehot, mask, seq_lengths, targets, sequences=sequences)


def load_tdc_dataset(dataset_name, data_dir):
    """Load protein property dataset from LMDB files (PEER benchmark format)."""
    import lmdb
    import pickle as pkl

    # Map dataset names to LMDB subdirectory and target field
    DATASET_MAP = {
        'Beta': ('beta_lactamase', 'scaled_effect1'),
        'Fluorescence': ('fluorescence', 'log_fluorescence'),
        'Solubility': ('solubility', 'solubility'),
    }

    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_MAP.keys())}")

    subdir, target_field = DATASET_MAP[dataset_name]
    splits = {}
    for split_name in ['train', 'valid', 'test']:
        lmdb_path = os.path.join(data_dir, 'DeepProtein', 'data', subdir, f'{subdir}_{split_name}.lmdb')
        env = lmdb.open(lmdb_path, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)
        sequences = []
        targets = []
        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
            for i in range(num_examples):
                item = pkl.loads(txn.get(str(i).encode()))
                sequences.append(item['primary'])
                target_val = item[target_field]
                if isinstance(target_val, (list, np.ndarray)) and np.ndim(target_val) > 0 and len(target_val) > 0:
                    target_val = float(target_val[0])
                else:
                    target_val = float(target_val)
                targets.append(target_val)
        env.close()
        splits[split_name] = (sequences, targets)
        print(f"  Loaded {split_name}: {len(sequences)} samples from {lmdb_path}")

    return splits


# =====================================================================
# Training and evaluation
# =====================================================================

# Solubility is binary classification (DeepProtein uses binary=True with PR-AUC/F1
# metrics; see vendor/external_packages/DeepProtein/train/solubility.py:53 and
# DeepProtein/ProteinPred.py:199-203 which return roc_auc/average_precision/f1).
def is_classification_task(dataset_name: str) -> bool:
    return dataset_name == 'Solubility'


def train_epoch(model, loader, optimizer, device, classification=False):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch).squeeze(-1)  # [B]
        if classification:
            loss = F.binary_cross_entropy_with_logits(logits, batch.targets.float())
        else:
            loss = F.mse_loss(logits, batch.targets)

        loss.backward()
        # DeepProtein reference training does not use gradient clipping.
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_regression(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch).squeeze(-1)
        all_preds.append(preds.cpu())
        all_targets.append(batch.targets.cpu())
    if not all_preds:
        return {'mse': float('inf'), 'spearman': 0.0, 'pearson': 0.0}
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    mse = mean_squared_error(targets, preds)
    spearman_corr, _ = spearmanr(targets, preds)
    pearson_corr, _ = pearsonr(targets, preds)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0
    if np.isnan(pearson_corr):
        pearson_corr = 0.0
    return {'mse': mse, 'spearman': spearman_corr, 'pearson': pearson_corr}


@torch.no_grad()
def evaluate_classification(model, loader, device):
    """Binary classification metrics: PR-AUC, F1, accuracy, ROC-AUC."""
    model.eval()
    all_logits, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).squeeze(-1)
        all_logits.append(logits.cpu())
        all_targets.append(batch.targets.cpu())
    if not all_logits:
        return {'pr_auc': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'roc_auc': 0.0}
    logits = torch.cat(all_logits).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    targets = torch.cat(all_targets).numpy().astype(np.int64)
    preds_bin = (probs >= 0.5).astype(np.int64)
    try:
        pr_auc = float(average_precision_score(targets, probs))
    except Exception:
        pr_auc = 0.0
    try:
        roc_auc = float(roc_auc_score(targets, probs))
    except Exception:
        roc_auc = 0.0
    f1 = float(f1_score(targets, preds_bin, average='binary', zero_division=0))
    acc = float(accuracy_score(targets, preds_bin))
    return {'pr_auc': pr_auc, 'f1': f1, 'accuracy': acc, 'roc_auc': roc_auc}


def evaluate(model, loader, device, classification=False):
    if classification:
        return evaluate_classification(model, loader, device)
    return evaluate_regression(model, loader, device)


def _format_metrics(m: Dict[str, float]) -> str:
    return ' '.join(f"{k}={v:.6f}" for k, v in m.items())


def train_and_evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    classification = is_classification_task(args.dataset)
    print(f"Task type: {'classification (binary)' if classification else 'regression'}")

    # Load data
    splits = load_tdc_dataset(args.dataset, args.data_dir)
    train_seqs, train_y = splits['train']
    val_seqs, val_y = splits['valid']
    test_seqs, test_y = splits['test']
    print(f"Dataset: {args.dataset}, train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")

    train_ds = ProteinDataset(train_seqs, train_y)
    val_ds = ProteinDataset(val_seqs, val_y)
    test_ds = ProteinDataset(test_seqs, test_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_proteins, num_workers=2, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_proteins, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_proteins, num_workers=2)

    # Model
    encoder = ProteinEncoder(
        vocab_size=VOCAB_SIZE,
        onehot_dim=ONEHOT_DIM,
        max_seq_len=MAX_SEQ_LEN,
    )
    model = ProteinPredictor(encoder, num_classes=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Training with DeepProtein-style validation checkpoint selection.
    best_metric = -float('inf') if classification else float('inf')
    best_epoch = 0
    select_key = 'roc_auc' if classification else 'mse'

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, classification=classification)
        val_metrics = evaluate(model, val_loader, device, classification=classification)

        # Always emit val_spearman / val_pearson / val_mse keys for parser back-compat
        # (parser doesn't actually consume them — TEST_METRICS is the source of truth).
        if classification:
            print(f"TRAIN_METRICS epoch={epoch} loss={train_loss:.6f} "
                  f"val_pr_auc={val_metrics['pr_auc']:.6f} val_f1={val_metrics['f1']:.6f} "
                  f"val_accuracy={val_metrics['accuracy']:.6f} val_roc_auc={val_metrics['roc_auc']:.6f}")
        else:
            print(f"TRAIN_METRICS epoch={epoch} loss={train_loss:.6f} "
                  f"val_mse={val_metrics['mse']:.6f} val_spearman={val_metrics['spearman']:.6f} "
                  f"val_pearson={val_metrics['pearson']:.6f}")

        cur = val_metrics[select_key]
        if (cur > best_metric) if classification else (cur < best_metric):
            best_metric = cur
            best_epoch = epoch
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model_{args.dataset}.pt'))

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_model_{args.dataset}.pt'), weights_only=True))
    test_metrics = evaluate(model, test_loader, device, classification=classification)
    if classification:
        print(f"TEST_METRICS pr_auc={test_metrics['pr_auc']:.6f} f1={test_metrics['f1']:.6f} "
              f"accuracy={test_metrics['accuracy']:.6f} roc_auc={test_metrics['roc_auc']:.6f}")
    else:
        print(f"TEST_METRICS spearman={test_metrics['spearman']:.6f} "
              f"pearson={test_metrics['pearson']:.6f} mse={test_metrics['mse']:.6f}")
    print(f"Best val {select_key}: {best_metric:.6f} at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Protein Function Prediction")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['Beta', 'Fluorescence', 'Solubility'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./output')
    args = parser.parse_args()

    # =====================================================================
    # EDITABLE — CONFIG_OVERRIDES
    # =====================================================================
    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate.
    CONFIG_OVERRIDES = {}

    # =====================================================================
    # FIXED — Apply config overrides and set seeds
    # =====================================================================
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': args.lr = _v

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_and_evaluate(args)


if __name__ == '__main__':
    main()

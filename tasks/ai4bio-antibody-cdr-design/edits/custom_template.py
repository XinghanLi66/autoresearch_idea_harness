"""
Antibody CDR Design — Custom generative model for CDR loop sequence-structure co-design.

This template provides a unified training and evaluation pipeline for antibody
complementarity-determining region (CDR) design using the CHIMERA-Bench framework.

Structure:
  Lines 1-221:     FIXED — Imports, data loading, utilities, constants
  Lines 222-514:   EDITABLE — CustomCDRModel class (starter: simple EGNN denoiser)
  Lines 515-end:   FIXED — Training loop, evaluation, CLI entry point

Interface:
  forward(batch) -> loss_dict: Dict[str, Tensor]
    Training forward pass. Returns dict of named losses to be summed.
  sample(batch) -> Dict[str, Tensor]
    Generate CDR sequences and structures. Returns:
      'seq': LongTensor [N_cdr] — predicted amino acid indices
      'pos': FloatTensor [N_cdr, 3] — predicted CA coordinates
      'mask': BoolTensor [N_cdr] — which residues are CDR (True)

Batch format (dict of tensors):
  'heavy_seq':      [L_h]       — heavy chain amino acid indices (0-19)
  'light_seq':      [L_l]       — light chain amino acid indices (0-19)
  'heavy_coords':   [L_h, 3]    — heavy chain CA coordinates
  'light_coords':   [L_l, 3]    — light chain CA coordinates
  'antigen_seq':    [L_ag]      — antigen amino acid indices
  'antigen_coords': [L_ag, 3]   — antigen CA coordinates
  'cdr_mask':       [L_h]       — CDR mask: -1=framework, 0=H1, 1=H2, 2=H3
  'epitope_mask':   [L_ag]      — True for epitope residues
  'batch_idx':      [L_total]   — batch index for batched processing
"""
import os
import sys
import json
import math
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import knn_graph, radius_graph

# ── Constants ──────────────────────────────────────────────────────
NUM_AA_TYPES = 20
CDR_LABELS = ['H1', 'H2', 'H3']  # Heavy-chain CDRs
MAX_CDR_LEN = {'H1': 16, 'H2': 14, 'H3': 32}

# Amino acid alphabet for sequence <-> index conversion
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
UNK_IDX = NUM_AA_TYPES  # mask/unknown token index


def seq_to_idx(seq_str: str) -> torch.Tensor:
    """Convert amino acid string to index tensor (0-19, unknown->UNK_IDX)."""
    return torch.tensor([AA_TO_IDX.get(c, UNK_IDX) for c in seq_str], dtype=torch.long)


def idx_to_seq(idx_tensor: torch.Tensor) -> str:
    """Convert index tensor back to amino acid string."""
    return ''.join(AA_ALPHABET[i] if 0 <= i < NUM_AA_TYPES else 'X'
                   for i in idx_tensor.tolist())


# ── Data loading ───────────────────────────────────────────────────

def load_chimera_split(data_root: str, split_name: str):
    """Load train/val/test complex IDs for a given split."""
    split_dir = Path(data_root) / "splits"
    split_file = split_dir / f"{split_name}.json"
    with open(split_file) as f:
        split_data = json.load(f)
    return split_data['train'], split_data['val'], split_data['test']


def _build_cdr_info(h_cdr_mask, l_cdr_mask, heavy_seq, light_seq,
                    heavy_coords, light_coords):
    """Build per-CDR info dict from masks and data."""
    cdr_info = {}
    cdr_names = {0: 'H1', 1: 'H2', 2: 'H3', 3: 'L1', 4: 'L2', 5: 'L3'}
    # Heavy chain CDRs
    for cdr_id in [0, 1, 2]:
        indices = np.where(h_cdr_mask == cdr_id)[0]
        if len(indices) > 0:
            cdr_info[cdr_names[cdr_id]] = {
                'chain': 'heavy',
                'indices': indices,
                'seq': ''.join(heavy_seq[i] for i in indices),
                'coords': heavy_coords[indices],
            }
    # Light chain CDRs
    for cdr_id in [3, 4, 5]:
        indices = np.where(l_cdr_mask == cdr_id)[0]
        if len(indices) > 0:
            cdr_info[cdr_names[cdr_id]] = {
                'chain': 'light',
                'indices': indices,
                'seq': ''.join(light_seq[i] for i in indices),
                'coords': light_coords[indices],
            }
    return cdr_info


class ChimeraDataset(Dataset):
    """Load pre-computed .pt feature files from CHIMERA-Bench."""

    def __init__(self, data_root: str, complex_ids: List[str],
                 numbering: str = 'imgt'):
        self.data_root = Path(data_root)
        self.features_dir = self.data_root / "complex_features"
        self.complex_ids = [cid for cid in complex_ids
                           if (self.features_dir / f"{cid}.pt").exists()]
        self.numbering = numbering

    def __len__(self):
        return len(self.complex_ids)

    def __getitem__(self, idx):
        cid = self.complex_ids[idx]
        raw = torch.load(self.features_dir / f"{cid}.pt", weights_only=False)

        # Extract CDR masks (numpy int arrays)
        # CDR masks cover the Fv (variable) region only, which is a prefix
        # of the full chain. Pad to full chain length with -1 (framework).
        cdr_masks = raw['cdr_masks'][self.numbering]
        h_cdr_fv = np.array(cdr_masks['heavy'], dtype=np.int64)
        l_cdr_fv = np.array(cdr_masks['light'], dtype=np.int64)
        h_full_len = len(raw['heavy_sequence'])
        l_full_len = len(raw['light_sequence'])
        h_cdr_mask = np.full(h_full_len, -1, dtype=np.int64)
        h_cdr_mask[:len(h_cdr_fv)] = h_cdr_fv
        l_cdr_mask = np.full(l_full_len, -1, dtype=np.int64)
        l_cdr_mask[:len(l_cdr_fv)] = l_cdr_fv

        # Convert coordinates to tensors
        heavy_coords = torch.from_numpy(raw['heavy_ca_coords']).float()
        light_coords = torch.from_numpy(raw['light_ca_coords']).float()
        ag_coords = torch.from_numpy(raw['antigen_ca_coords']).float()

        # Antigen surface chemical features (some complexes lack surface data)
        if 'ag_surface_chemical_feats' in raw:
            ag_surface = torch.from_numpy(raw['ag_surface_chemical_feats']).float()
        else:
            # Fallback: create zero features with expected shape [128, 6]
            ag_surface = torch.zeros(128, 6)

        # Build per-CDR info for evaluation
        cdr_info = _build_cdr_info(
            h_cdr_mask, l_cdr_mask,
            raw['heavy_sequence'], raw['light_sequence'],
            raw['heavy_ca_coords'], raw['light_ca_coords'],
        )

        sample = {
            'complex_id': cid,
            'heavy_seq': raw['heavy_sequence'],       # str
            'light_seq': raw['light_sequence'],        # str
            'heavy_coords': heavy_coords,              # [L_h, 3]
            'light_coords': light_coords,              # [L_l, 3]
            'ag_coords': ag_coords,                    # [L_ag, 3]
            'ag_surface': ag_surface,                   # [N_surf, 6]
            'h_cdr_mask': h_cdr_mask,                  # [L_h] int64
            'l_cdr_mask': l_cdr_mask,                  # [L_l] int64
            'cdr_info': cdr_info,                      # dict
        }
        return sample


def collate_fn(batch_list):
    """Return batch as a list of individual sample dicts.

    Each model handles per-sample processing internally (variable-length
    antibody complexes are not trivially stackable).
    """
    return batch_list


# ── Evaluation metrics (from chimera-bench/evaluation/metrics.py) ──

def compute_aar(pred_seq, true_seq):
    """Amino acid recovery rate."""
    return (pred_seq == true_seq).float().mean().item()


def compute_kabsch_rmsd(pred_pos, true_pos):
    """CA RMSD after Kabsch alignment."""
    pred_c = pred_pos - pred_pos.mean(0)
    true_c = true_pos - true_pos.mean(0)
    H = pred_c.T @ true_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    sign_mat = torch.diag(torch.tensor([1.0, 1.0, d.sign()], device=pred_pos.device))
    R = Vt.T @ sign_mat @ U.T
    pred_aligned = pred_c @ R.T
    return torch.sqrt(((pred_aligned - true_c) ** 2).sum(-1).mean()).item()


def compute_tm_score(pred_pos, true_pos):
    """TM-score between predicted and true CA coordinates."""
    L = true_pos.shape[0]
    if L < 5:
        return 0.0
    d0_arg = L - 15
    d0 = 1.24 * (abs(d0_arg) ** (1.0 / 3.0)) * (1 if d0_arg >= 0 else -1) - 1.8
    d0 = max(d0, 0.5)
    pred_c = pred_pos - pred_pos.mean(0)
    true_c = true_pos - true_pos.mean(0)
    H = pred_c.T @ true_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    sign_mat = torch.diag(torch.tensor([1.0, 1.0, d.sign()], device=pred_pos.device))
    R = Vt.T @ sign_mat @ U.T
    pred_aligned = pred_c @ R.T
    di = torch.sqrt(((pred_aligned - true_c) ** 2).sum(-1))
    tm = (1.0 / (1.0 + (di / d0) ** 2)).sum() / L
    return tm.item()

# =====================================================================
# EDITABLE SECTION START — CustomCDRModel
# =====================================================================

class EquivariantLayer(nn.Module):
    """Simple E(n)-equivariant message passing layer."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, h, x, edge_index):
        src, dst = edge_index
        rel_pos = x[src] - x[dst]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)

        edge_feat = torch.cat([h[src], h[dst], dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_feat)
        agg = scatter_add(m_ij, dst, dim=0, dim_size=h.size(0))
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        coord_w = self.coord_mlp(m_ij)
        coord_agg = scatter_add(coord_w * rel_pos, dst, dim=0, dim_size=x.size(0))
        x_new = x + coord_agg

        return h_new, x_new


class CustomCDRModel(nn.Module):
    """Starter: Simple EGNN denoiser for CDR sequence-structure co-design.

    This is a basic DDPM-style model that denoises CDR coordinates and
    predicts amino acid types. It treats the antibody framework and antigen
    as conditioning context.

    The agent should replace this with a better architecture (e.g., SE(3)
    equivariant networks, flow matching, iterative refinement, attention
    over epitope, etc.).
    """

    def __init__(self, hidden_dim=128, num_layers=6, num_diffusion_steps=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_diffusion_steps

        # Residue embeddings
        self.aa_emb = nn.Embedding(NUM_AA_TYPES + 1, hidden_dim)  # +1 for mask token
        self.cdr_emb = nn.Embedding(4, hidden_dim)  # -1=fw, 0=H1, 1=H2, 2=H3
        self.role_emb = nn.Embedding(3, hidden_dim)  # 0=heavy, 1=light, 2=antigen

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Equivariant layers
        self.layers = nn.ModuleList([
            EquivariantLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Output heads
        self.seq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, NUM_AA_TYPES),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())

    def _sinusoidal_emb(self, t, dim):
        half = dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def _encode_context(self, sample, device, x_cdr_noisy):
        """Encode antibody + antigen context with noisy CDR positions."""
        # Heavy chain
        h_seq_idx = seq_to_idx(sample['heavy_seq']).to(device)
        h_coords = sample['heavy_coords'].to(device).clone()
        cdr_mask = torch.from_numpy(sample['h_cdr_mask']).to(device)
        is_cdr = (cdr_mask >= 0)

        # Replace CDR coords with noisy versions
        h_coords[is_cdr] = x_cdr_noisy

        # Node features: aa embedding + role + CDR type
        h_h = self.aa_emb(h_seq_idx) + self.role_emb(torch.zeros_like(h_seq_idx))
        cdr_shifted = cdr_mask + 1  # -1->0, 0->1, 1->2, 2->3
        h_h = h_h + self.cdr_emb(cdr_shifted)

        # Light chain
        l_seq_idx = seq_to_idx(sample['light_seq']).to(device)
        l_coords = sample['light_coords'].to(device)
        h_l = self.aa_emb(l_seq_idx) + self.role_emb(torch.ones_like(l_seq_idx))

        # Antigen
        ag_coords = sample['ag_coords'].to(device)
        # Use surface features as a proxy for antigen node encoding
        ag_surf = sample['ag_surface'].to(device)  # [N_surf, 6]
        ag_surf_pooled = ag_surf.mean(0, keepdim=True).expand(ag_coords.shape[0], -1)
        # Project 6-dim surface features to hidden_dim via a simple linear
        ag_feat = torch.zeros(ag_coords.shape[0], self.hidden_dim, device=device)
        ag_feat = ag_feat + self.role_emb(2 * torch.ones(ag_coords.shape[0],
                                          dtype=torch.long, device=device))

        # Concatenate all
        h_all = torch.cat([h_h, h_l, ag_feat], dim=0)
        x_all = torch.cat([h_coords, l_coords, ag_coords], dim=0)

        return h_all, x_all, is_cdr

    def forward(self, batch):
        """Training forward: add noise to CDR, predict noise and sequence."""
        device = next(self.parameters()).device
        total_pos_loss = torch.tensor(0.0, device=device)
        total_seq_loss = torch.tensor(0.0, device=device)
        n = 0

        for sample in batch:
            cdr_mask = torch.from_numpy(sample['h_cdr_mask']).to(device)
            is_cdr = (cdr_mask >= 0)
            n_cdr = is_cdr.sum().item()
            if n_cdr == 0:
                continue

            x_cdr_0 = sample['heavy_coords'].to(device)[is_cdr]
            seq_cdr_0 = seq_to_idx(sample['heavy_seq']).to(device)[is_cdr]

            # Sample timestep
            t = torch.randint(0, self.num_steps, (n_cdr,), device=device)

            # Add noise to CDR positions
            sqrt_a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
            sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            noise = torch.randn_like(x_cdr_0)
            x_cdr_noisy = sqrt_a * x_cdr_0 + sqrt_1ma * noise

            # Encode full complex with noisy CDR
            t_emb = self.time_emb(self._sinusoidal_emb(t.float(), self.hidden_dim))
            h_all, x_all, is_cdr_full = self._encode_context(sample, device, x_cdr_noisy)

            # Add time embedding to CDR nodes
            cdr_indices_in_all = torch.where(is_cdr_full)[0]
            h_all[cdr_indices_in_all] = h_all[cdr_indices_in_all] + t_emb

            # Build graph
            edge_index = knn_graph(x_all, k=16, batch=None, flow='source_to_target')

            # Run equivariant layers
            for layer in self.layers:
                h_all, x_all = layer(h_all, x_all, edge_index)

            # Extract CDR outputs
            h_cdr = h_all[cdr_indices_in_all]

            # Predict noise and sequence
            eps_pred = self.pos_head(h_cdr)
            seq_logits = self.seq_head(h_cdr)

            total_pos_loss = total_pos_loss + F.mse_loss(eps_pred, noise)
            total_seq_loss = total_seq_loss + F.cross_entropy(seq_logits, seq_cdr_0)
            n += 1

        if n > 0:
            total_pos_loss = total_pos_loss / n
            total_seq_loss = total_seq_loss / n

        return {'pos': total_pos_loss, 'seq': total_seq_loss}

    @torch.no_grad()
    def sample(self, batch):
        """Generate CDR sequences and structures via reverse diffusion.

        Returns a list of per-CDR prediction dicts (compatible with baseline
        evaluation interface).
        """
        device = next(self.parameters()).device
        self.eval()
        predictions = []

        for sample in batch:
            cdr_info = sample['cdr_info']
            cdr_mask = torch.from_numpy(sample['h_cdr_mask']).to(device)
            is_cdr = (cdr_mask >= 0)
            n_cdr = is_cdr.sum().item()

            if n_cdr == 0:
                for label, info in cdr_info.items():
                    predictions.append({
                        'complex_id': sample['complex_id'],
                        'cdr_type': label,
                        'pred_sequence': info['seq'],
                        'true_sequence': info['seq'],
                        'pred_coords': info['coords'],
                        'true_coords': info['coords'],
                        'ppl': 0.0,
                    })
                continue

            # Start from noise
            x_cdr_t = torch.randn(n_cdr, 3, device=device)

            for t_idx in reversed(range(self.num_steps)):
                t_tensor = torch.full((n_cdr,), t_idx, device=device, dtype=torch.long)
                t_emb = self.time_emb(self._sinusoidal_emb(t_tensor.float(), self.hidden_dim))

                h_all, x_all, is_cdr_full = self._encode_context(sample, device, x_cdr_t)
                cdr_indices_in_all = torch.where(is_cdr_full)[0]
                h_all[cdr_indices_in_all] = h_all[cdr_indices_in_all] + t_emb

                edge_index = knn_graph(x_all, k=16, batch=None, flow='source_to_target')
                for layer in self.layers:
                    h_all, x_all = layer(h_all, x_all, edge_index)

                h_cdr = h_all[cdr_indices_in_all]
                eps_pred = self.pos_head(h_cdr)

                # DDPM reverse step
                alpha_t = self.alphas[t_idx]
                beta_t = self.betas[t_idx]
                coeff = beta_t / self.sqrt_one_minus_alphas_cumprod[t_idx]
                x_mean = (1.0 / alpha_t.sqrt()) * (x_cdr_t - coeff * eps_pred)
                if t_idx > 0:
                    x_cdr_t = x_mean + beta_t.sqrt() * torch.randn_like(x_cdr_t)
                else:
                    x_cdr_t = x_mean

            # Final sequence prediction
            h_all, x_all, is_cdr_full = self._encode_context(sample, device, x_cdr_t)
            cdr_indices_in_all = torch.where(is_cdr_full)[0]
            t_zero = torch.zeros(n_cdr, device=device, dtype=torch.long)
            t_emb = self.time_emb(self._sinusoidal_emb(t_zero.float(), self.hidden_dim))
            h_all[cdr_indices_in_all] = h_all[cdr_indices_in_all] + t_emb
            edge_index = knn_graph(x_all, k=16, batch=None, flow='source_to_target')
            for layer in self.layers:
                h_all, x_all = layer(h_all, x_all, edge_index)

            h_cdr = h_all[cdr_indices_in_all]
            seq_logits = self.seq_head(h_cdr)
            seq_pred = seq_logits.argmax(dim=-1)

            # Extract per-CDR predictions
            cdr_positions = torch.where(is_cdr)[0]
            for label, info in cdr_info.items():
                if info['chain'] == 'heavy' and n_cdr > 0:
                    local_idx = torch.from_numpy(info['indices']).to(device)
                    mask = torch.isin(cdr_positions, local_idx)
                    p_coords = x_cdr_t[mask].cpu().numpy()
                    p_seq = idx_to_seq(seq_pred[mask])
                else:
                    p_coords = info['coords']
                    p_seq = info['seq']

                predictions.append({
                    'complex_id': sample['complex_id'],
                    'cdr_type': label,
                    'pred_sequence': p_seq,
                    'true_sequence': info['seq'],
                    'pred_coords': p_coords,
                    'true_coords': info['coords'],
                    'ppl': 0.0,
                })

        return predictions

# =====================================================================
# EDITABLE SECTION END
# =====================================================================


# ── Training loop (FIXED) ─────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device, epoch, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        loss_dict = model(batch)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 100 == 0:
            parts = ' '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())
            print(f"TRAIN_METRICS epoch={epoch} step={n_batches} loss={loss.item():.4f} {parts}",
                  flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, cdr_filter='H3'):
    """Evaluate model predictions.

    Following CHIMERA-Bench Track 1 (Luo et al., CHIMERA-Bench paper Tables
    1/9/10) and the original DiffAb/MEAN/dyMEAN papers, AAR/RMSD/TM-score are
    reported on CDR-H3 ONLY by default. CDR-H3 is the most variable and
    challenging loop; the conserved H1/H2/L1/L2/L3 loops would inflate
    AAR (~1.0 when the light chain is trivially copied from GT) and are
    not what the paper reports.

    Set cdr_filter=None to evaluate on all CDRs returned by model.sample().
    """
    model.eval()
    all_aar, all_rmsd, all_tm = [], [], []

    for batch in dataloader:
        predictions = model.sample(batch)

        for pred in predictions:
            # Restrict to the designated CDR (default: H3, matching paper)
            if cdr_filter is not None and pred.get('cdr_type') != cdr_filter:
                continue
            pred_seq_str = pred['pred_sequence']
            true_seq_str = pred['true_sequence']
            pred_coords = pred['pred_coords']
            true_coords = pred['true_coords']

            # Convert strings to index tensors
            pred_seq_t = seq_to_idx(pred_seq_str)
            true_seq_t = seq_to_idx(true_seq_str)

            # Convert coords to tensors if needed
            if isinstance(pred_coords, np.ndarray):
                pred_coords = torch.from_numpy(pred_coords).float()
            if isinstance(true_coords, np.ndarray):
                true_coords = torch.from_numpy(true_coords).float()

            if len(pred_seq_t) < 2 or len(true_seq_t) < 2:
                continue
            min_len = min(len(pred_seq_t), len(true_seq_t))
            aar = compute_aar(pred_seq_t[:min_len], true_seq_t[:min_len])
            if pred_coords.shape[0] >= 2 and true_coords.shape[0] >= 2:
                min_pos = min(pred_coords.shape[0], true_coords.shape[0])
                rmsd = compute_kabsch_rmsd(pred_coords[:min_pos], true_coords[:min_pos])
                tm = compute_tm_score(pred_coords[:min_pos], true_coords[:min_pos])
            else:
                rmsd = 999.0
                tm = 0.0
            all_aar.append(aar)
            all_rmsd.append(rmsd)
            all_tm.append(tm)

    return {
        'aar': np.mean(all_aar) if all_aar else 0.0,
        'rmsd': np.mean(all_rmsd) if all_rmsd else 999.0,
        'tm_score': np.mean(all_tm) if all_tm else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True,
                        choices=['epitope_group', 'antigen_fold', 'temporal'])
    parser.add_argument('--data-root', type=str,
                        default=os.environ.get('CHIMERA_DATA_ROOT', '/data/chimera-bench-v1.0'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # CONFIG_OVERRIDES: override training hyperparameters for your method.
    # Allowed keys: learning_rate.
    CONFIG_OVERRIDES = {}

    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': args.lr = _v

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else Path(f'./results/custom/{args.split}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_ids, val_ids, test_ids = load_chimera_split(args.data_root, args.split)
    train_ds = ChimeraDataset(args.data_root, train_ids)
    val_ds = ChimeraDataset(args.data_root, val_ids)
    test_ds = ChimeraDataset(args.data_root, test_ids)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model
    model = CustomCDRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_rmsd = float('inf')
    patience_counter = 0
    max_patience = 10

    print(f"Training on split={args.split}, device={device}", flush=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}", flush=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics['rmsd'])

        elapsed = time.time() - t0
        print(f"TRAIN_METRICS epoch={epoch} train_loss={train_loss:.4f} "
              f"val_aar={val_metrics['aar']:.4f} val_rmsd={val_metrics['rmsd']:.4f} "
              f"val_tm={val_metrics['tm_score']:.4f} time={elapsed:.1f}s", flush=True)

        if val_metrics['rmsd'] < best_val_rmsd:
            best_val_rmsd = val_metrics['rmsd']
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(output_dir / 'best_model.pt', weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    print(f"TEST_METRICS aar={test_metrics['aar']:.6f}", flush=True)
    print(f"TEST_METRICS rmsd={test_metrics['rmsd']:.6f}", flush=True)
    print(f"TEST_METRICS tm_score={test_metrics['tm_score']:.6f}", flush=True)

    # Save results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == '__main__':
    main()

"""
Unconditional 3D Structure Generation — Self-contained template.
Generates molecules (QM9, GEOM-DRUG) and crystals (MP-20).

Structure:
  Lines 1-40:   FIXED — Imports + Constants
  Lines 41-300: EDITABLE — StructureGenerator + helpers (starter EGNN + DDPM)
  Lines 301+:   FIXED — Data loading, evaluation, training loop, DDP, metrics
"""
import os
import sys
import math
import json
import argparse
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import Counter
from multiprocessing import Pool
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# Atom type mappings
# QM9: H=1, C=6, N=7, O=8, F=9  (indices 1-5 in our mapping)
# GEOM-DRUG: extends with more elements
# MP-20: up to ~100 elements (atomic numbers)
MOLECULE_ATOM_TYPES = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
MAX_ATOM_TYPE = 120  # covers all elements

class DatasetType(Enum):
    MOLECULE = "molecule"
    CRYSTAL = "crystal"

# =====================================================================
# EDITABLE SECTION START — StructureGenerator + helper modules
# =====================================================================

class EGNNLayer(nn.Module):
    """E(n) Equivariant Graph Neural Network layer."""

    def __init__(self, hidden_dim, edge_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Edge MLP: takes h_i, h_j, ||x_i - x_j||^2, edge_feat
        edge_input = 2 * hidden_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.01)

    def forward(self, h, x, mask, edge_attr=None):
        """
        h: (B, N, D) node features
        x: (B, N, 3) coordinates
        mask: (B, N) binary mask
        edge_attr: optional (B, N, N, edge_dim) edge features
        """
        B, N, D = h.shape
        # Pairwise differences
        dx = x.unsqueeze(2) - x.unsqueeze(1)  # (B, N, N, 3)
        dist_sq = (dx ** 2).sum(-1, keepdim=True)  # (B, N, N, 1)

        # Edge inputs
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([hi, hj, dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        # Edge messages
        m_ij = self.edge_mlp(edge_input)  # (B, N, N, D)

        # Mask: zero out padding-to-padding and self interactions
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # (B, N, N)
        eye_mask = 1.0 - torch.eye(N, device=h.device).unsqueeze(0)
        pair_mask = pair_mask * eye_mask
        m_ij = m_ij * pair_mask.unsqueeze(-1)

        # Coordinate update (equivariant) — tanh to prevent explosion
        coord_weights = self.coord_mlp(m_ij)  # (B, N, N, 1)
        coord_weights = torch.tanh(coord_weights)
        x_update = (dx * coord_weights).sum(dim=2)  # (B, N, 3)
        # Normalize by neighbor count
        num_neighbors = pair_mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B, N, 1)
        x = x + (x_update / num_neighbors) * mask.unsqueeze(-1)

        # Node update — normalize aggregation by neighbor count
        agg = m_ij.sum(dim=2) / num_neighbors  # (B, N, D)
        h = self.node_norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)) * mask.unsqueeze(-1))

        return h, x


class EGNN(nn.Module):
    """Equivariant GNN with multiple layers."""

    def __init__(self, in_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, h, x, mask):
        h = self.embed(h)
        for layer in self.layers:
            h, x = layer(h, x, mask)
        return self.out(h), x


class StructureGenerator(nn.Module):
    """Generative model for 3D structures.

    Starter implementation: Simple EGNN + DDPM diffusion on coordinates
    with a categorical diffusion on atom types.

    Interface:
        compute_loss(batch) -> loss
        sample(n_samples, atom_counts, device, **kwargs) -> dict
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = getattr(config, 'hidden_dim', 256)
        self.num_layers = getattr(config, 'num_layers', 6)
        self.num_timesteps = getattr(config, 'num_timesteps', 1000)
        self.max_atom_type = MAX_ATOM_TYPE

        # Atom type embedding + time embedding -> input features
        self.atom_embed = nn.Embedding(MAX_ATOM_TYPE + 1, self.hidden_dim, padding_idx=0)
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Score network (EGNN-based)
        self.score_net = EGNN(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

        # Atom type prediction head
        self.type_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, MAX_ATOM_TYPE + 1),
        )

        # Noise schedule (linear beta schedule)
        betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def _center_positions(self, pos, mask):
        """Center positions to zero center of mass (for equivariance)."""
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        com = (pos * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)
        pos = (pos - com) * mask.unsqueeze(-1)
        return pos

    def compute_loss(self, batch):
        """Compute training loss.

        batch contains:
          'positions' or 'frac_coords': (B, N, 3) coordinates
          'atom_types': (B, N) integer types (0=padding)
          'num_atoms': (B,)
          'mask': (B, N)
          'dataset_type': str 'molecule' or 'crystal'
          'lattice': (B, 3, 3) for crystals only
        """
        if batch['dataset_type'] == 'molecule':
            pos = batch['positions']
        else:
            pos = batch['frac_coords']

        atom_types = batch['atom_types']
        mask = batch['mask']
        B, N = atom_types.shape

        # Center positions for molecules
        if batch['dataset_type'] == 'molecule':
            pos = self._center_positions(pos, mask)

        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=pos.device)

        # Forward diffusion on coordinates
        noise = torch.randn_like(pos) * mask.unsqueeze(-1)
        if batch['dataset_type'] == 'molecule':
            noise = self._center_positions(noise, mask)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        pos_noisy = sqrt_alpha * pos + sqrt_one_minus_alpha * noise

        # Predict noise
        t_norm = t.float() / self.num_timesteps
        h_input = self.atom_embed(atom_types) + self.time_embed(t_norm.unsqueeze(-1)).unsqueeze(1)
        h_pred, pos_pred = self.score_net(h_input, pos_noisy, mask)

        # Coordinate loss (predict noise)
        coord_loss = F.mse_loss(pos_pred * mask.unsqueeze(-1),
                                (pos_noisy - noise) * mask.unsqueeze(-1))

        # Atom type loss (auxiliary)
        type_logits = self.type_head(h_pred)
        type_loss = F.cross_entropy(
            type_logits.reshape(-1, self.max_atom_type + 1),
            atom_types.reshape(-1),
            ignore_index=0,
        )

        loss = coord_loss + 0.1 * type_loss
        return loss

    @torch.no_grad()
    def sample(self, n_samples, atom_counts, device, **kwargs):
        """Generate new 3D structures.

        Args:
            n_samples: number of structures to generate
            atom_counts: (n_samples,) tensor of atom counts per structure
            device: torch device
            **kwargs: 'dataset_type' (str), 'atom_type_map' (list of valid types)

        Returns:
            dict with 'positions'/'frac_coords', 'atom_types'
            For crystals: also 'lattice'
        """
        dataset_type = kwargs.get('dataset_type', 'molecule')
        max_atoms = int(atom_counts.max().item())

        # Create mask
        arange = torch.arange(max_atoms, device=device).unsqueeze(0)
        mask = (arange < atom_counts.unsqueeze(1)).float()

        # Start from noise
        pos = torch.randn(n_samples, max_atoms, 3, device=device) * mask.unsqueeze(-1)
        if dataset_type == 'molecule':
            pos = self._center_positions(pos, mask)

        # Reverse diffusion
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            t_norm = t.float() / self.num_timesteps

            # Dummy atom types for initial pass (uniform random)
            if t_idx == self.num_timesteps - 1:
                atom_types_cur = torch.ones(n_samples, max_atoms, device=device, dtype=torch.long)
            h_input = self.atom_embed(atom_types_cur) + self.time_embed(t_norm.unsqueeze(-1)).unsqueeze(1)
            h_pred, pos_denoised = self.score_net(h_input, pos, mask)

            # Update atom type prediction
            type_logits = self.type_head(h_pred)
            atom_types_cur = type_logits.argmax(dim=-1) * mask.long()

            # DDPM reverse step
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]

            if t_idx > 0:
                noise = torch.randn_like(pos) * mask.unsqueeze(-1)
                if dataset_type == 'molecule':
                    noise = self._center_positions(noise, mask)
                sigma = torch.sqrt(beta_t)
                # DDPM update: x0 parameterization → convert to epsilon first
                sqrt_ab = torch.sqrt(alpha_bar_t)
                sqrt_1mab = torch.sqrt(1 - alpha_bar_t)
                eps_pred = (pos - sqrt_ab * pos_denoised) / sqrt_1mab.clamp(min=1e-8)
                pos = (1.0 / torch.sqrt(alpha_t)) * (
                    pos - (beta_t / sqrt_1mab.clamp(min=1e-8)) * eps_pred
                ) + sigma * noise
            else:
                pos = pos_denoised

            pos = pos * mask.unsqueeze(-1)

        # Clamp fractional coords for crystals
        if dataset_type == 'crystal':
            pos = pos % 1.0

        result = {'atom_types': atom_types_cur}
        if dataset_type == 'molecule':
            result['positions'] = pos
        else:
            result['frac_coords'] = pos
            # Generate simple cubic lattice as placeholder
            lattice = torch.eye(3, device=device).unsqueeze(0).expand(n_samples, -1, -1) * 5.0
            result['lattice'] = lattice

        return result

# =====================================================================
# EDITABLE SECTION END
# =====================================================================

# =====================================================================
# FIXED SECTION — Data loading, evaluation, training loop
# =====================================================================

# ----- Bond analysis data (from bond_analyze.py) -----
BONDS1 = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92, "B": 119, "Si": 148,
           "P": 144, "As": 152, "S": 134, "Cl": 127, "Br": 141, "I": 161},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135, "Si": 185,
           "P": 184, "S": 182, "Cl": 177, "Br": 194, "I": 214},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136, "Cl": 175,
           "Br": 214, "S": 168, "I": 222, "P": 177},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142, "Br": 172,
           "S": 151, "P": 163, "Si": 163, "Cl": 164, "I": 194},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142, "S": 158,
           "Si": 160, "Cl": 166, "Br": 178, "P": 156, "I": 187},
    "B": {"H": 119, "Cl": 175},
    "Si": {"Si": 233, "H": 148, "C": 185, "O": 163, "S": 200, "F": 160,
            "Cl": 202, "Br": 215, "I": 243},
    "Cl": {"Cl": 199, "H": 127, "C": 177, "N": 175, "O": 164, "P": 203,
            "S": 207, "B": 175, "Si": 202, "F": 166, "Br": 214},
    "S": {"H": 134, "C": 182, "N": 168, "O": 151, "S": 204, "F": 158,
           "Cl": 207, "Br": 225, "Si": 200, "P": 210, "I": 234},
    "Br": {"Br": 228, "H": 141, "C": 194, "O": 172, "N": 214, "Si": 215,
            "S": 225, "F": 178, "Cl": 214, "P": 222},
    "P": {"P": 221, "H": 144, "C": 184, "O": 163, "Cl": 203, "S": 210,
           "F": 156, "N": 177, "Br": 222},
    "I": {"H": 161, "C": 214, "Si": 243, "N": 222, "O": 194, "S": 234,
           "F": 187, "I": 266},
    "As": {"H": 152},
}
BONDS2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}
BONDS3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}
MARGIN1, MARGIN2, MARGIN3 = 10, 5, 3
ALLOWED_BONDS = {
    "H": 1, "C": 4, "N": 3, "O": 2, "F": 1, "B": 3, "Al": 3, "Si": 4,
    "P": [3, 5], "S": 4, "Cl": 1, "As": 3, "Br": 1, "I": 1, "Hg": [1, 2], "Bi": [3, 5],
}

ATOM_LIST = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

def atomic_number_to_symbol(z):
    if 1 <= z <= len(ATOM_LIST):
        return ATOM_LIST[z - 1]
    return "X"


# ----- Molecule evaluation -----
def get_bond_order(atom1, atom2, distance, single_bond=False):
    distance = 100 * distance
    if atom1 not in BONDS1 or atom2 not in BONDS1.get(atom1, {}):
        return 0
    if distance < BONDS1[atom1][atom2] + MARGIN1:
        if atom1 in BONDS2 and atom2 in BONDS2.get(atom1, {}):
            thr2 = BONDS2[atom1][atom2] + MARGIN2
            if distance < thr2:
                if atom1 in BONDS3 and atom2 in BONDS3.get(atom1, {}):
                    thr3 = BONDS3[atom1][atom2] + MARGIN3
                    if distance < thr3:
                        return 3 if not single_bond else 1
                return 2 if not single_bond else 1
        return 1
    return 0


def build_molecule(positions, atom_types_str, single_bond=False):
    """Build RDKit mol from positions and atom type strings."""
    try:
        from rdkit import Chem
    except ImportError:
        return None, False, 0, len(atom_types_str)

    # Filter out invalid atom symbols (e.g. 'X' from atomic_number_to_symbol
    # when the model generates atom type index 0 or out-of-range values)
    n_original = len(atom_types_str)
    valid_indices = [i for i, sym in enumerate(atom_types_str) if sym != "X"]
    if len(valid_indices) == 0:
        return None, False, 0, n_original
    positions = positions[valid_indices]
    atom_types_str = [atom_types_str[i] for i in valid_indices]

    n = len(atom_types_str)
    E = np.zeros((n, n), dtype=int)
    dists = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    for i in range(n):
        for j in range(i):
            order = get_bond_order(atom_types_str[i], atom_types_str[j],
                                   dists[i, j], single_bond=single_bond)
            if order > 0:
                E[i, j] = E[j, i] = order

    mol = Chem.RWMol()
    for sym in atom_types_str:
        try:
            mol.AddAtom(Chem.Atom(sym))
        except RuntimeError:
            return None, False, 0, n

    bond_dict = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE,
                 3: Chem.rdchem.BondType.TRIPLE}
    for i in range(n):
        for j in range(i):
            if E[i, j] > 0 and E[i, j] in bond_dict:
                mol.AddBond(i, j, bond_dict[E[i, j]])

    nr_stable = 0
    nr_bonds = E.sum(1)
    for sym, nb in zip(atom_types_str, nr_bonds):
        possible = ALLOWED_BONDS.get(sym, 0)
        if isinstance(possible, int):
            nr_stable += int(possible == nb)
        else:
            nr_stable += int(nb in possible)

    mol_stable = (nr_stable == n)
    return mol, mol_stable, nr_stable, n


def mol2smiles(mol):
    try:
        from rdkit import Chem, RDLogger
        RDLogger.logger().setLevel(RDLogger.CRITICAL)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def evaluate_molecules(generated_positions, generated_types, dataset="qm9"):
    """Evaluate generated molecules.
    generated_positions: list of (N_i, 3) arrays
    generated_types: list of (N_i,) arrays of atomic numbers
    Returns dict of metrics.
    """
    from rdkit import Chem, RDLogger
    RDLogger.logger().setLevel(RDLogger.CRITICAL)

    n_total = len(generated_positions)
    n_atom_stable = 0
    n_mol_stable = 0
    n_valid = 0
    n_atoms_total = 0
    valid_smiles = []
    single_bond = (dataset == "drug" or dataset == "geom_drug")

    for pos, types in zip(generated_positions, generated_types):
        atom_syms = [atomic_number_to_symbol(int(z)) for z in types]
        mol, mol_stable_i, nr_stable_i, n_atoms = build_molecule(
            pos, atom_syms, single_bond=single_bond
        )
        n_atoms_total += n_atoms
        n_atom_stable += nr_stable_i
        n_mol_stable += int(mol_stable_i)

        if mol is not None:
            smi = mol2smiles(mol)
            if smi is not None:
                n_valid += 1
                valid_smiles.append(smi)

    unique_smiles = list(set(valid_smiles))

    metrics = {
        "atom_stable": n_atom_stable / max(n_atoms_total, 1),
        "mol_stable": n_mol_stable / max(n_total, 1),
        "valid": n_valid / max(n_total, 1),
        "unique": len(unique_smiles) / max(len(valid_smiles), 1),
    }
    return metrics


# ----- Crystal evaluation -----
def evaluate_crystals(generated_frac_coords, generated_types, generated_lattices,
                      gt_data, n_eval_samples=1000):
    """Evaluate generated crystals.
    Returns dict of metrics.
    """
    try:
        import itertools
        from scipy.stats import wasserstein_distance
        from scipy.spatial.distance import pdist, cdist
        from pymatgen.core.structure import Structure, Composition, Lattice
        from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
        from matminer.featurizers.composition.composite import ElementProperty
        import smact
        from smact.screening import pauling_test
        from pymatgen.core.periodic_table import Element
    except ImportError as e:
        print(f"Crystal evaluation dependencies missing: {e}")
        return {"valid": 0.0}

    CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
    CompFP = ElementProperty.from_preset("magpie")
    COV_Cutoffs = {"mp20": {"struc": 0.4, "comp": 10.0}}

    # Per-crystal timeout for CrystalNNFingerprint (some unusual geometries hang)
    import signal
    CRYSTAL_FP_TIMEOUT = 60  # seconds per crystal

    class _CrystalFPTimeout(Exception):
        pass

    def _crystal_fp_timeout_handler(signum, frame):
        raise _CrystalFPTimeout()

    def _compute_struct_fp(structure, crystal_nn_fp, timeout=CRYSTAL_FP_TIMEOUT):
        """Compute CrystalNNFingerprint with a per-crystal timeout.
        Returns the mean site fingerprint, or None if timed out / failed."""
        old_handler = signal.signal(signal.SIGALRM, _crystal_fp_timeout_handler)
        try:
            signal.alarm(timeout)
            site_fps = [crystal_nn_fp.featurize(structure, i)
                        for i in range(len(structure))]
            signal.alarm(0)  # cancel alarm
            return np.array(site_fps).mean(axis=0)
        except _CrystalFPTimeout:
            return None
        except Exception:
            signal.alarm(0)
            return None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    # CompScaler (pre-computed from MP-20 training set)
    CompScalerMeans = np.array([
        21.194441759304013, 58.20212663122281, 37.0076848719188, 36.52738520455582,
        13.350626389725019, 29.468922184630255, 28.71735137747704, 78.8868535524408,
        50.16950217496375, 59.56764743604155, 19.020429484306277, 61.335572740454325,
        47.14515893344343, 141.75135923307818, 94.60620029962553, 85.95794070476977,
        34.07300576173523, 68.06189371516912, 637.9862061297893, 1817.2394155466848,
        1179.2532094169414, 1127.2743149568837, 431.51034284549826, 909.1060025135899,
        3.7744320927984534, 13.673707104881585, 9.899275012083132, 9.620186927095652,
        3.8426065581251856, 9.96950217496375, 3.305461575640406, 5.483035282745288,
        2.1775737071048815, 4.215114560306594, 0.8206087101824266, 3.732092798453359,
        109.16732721121315, 179.5570323827936, 70.38970517158047, 136.0978305229613,
        27.027545809538527, 119.16713388110198, 1.2721433060967857, 2.4614001837260617,
        1.1892568776289631, 1.9844483610247092, 0.4691462290494881, 2.100143582306204,
        1.4829869502174964, 1.9899951667472209, 0.5070082165297245, 1.7956250375970633,
        0.2056251946617602, 1.745867568873852, 0.05650072498791687, 2.3618656355727405,
        2.3053649105848235, 1.2829636137262992, 0.9995555685850794, 1.5150314161430642,
        0.7731271145480909, 7.4648139197680035, 6.691686805219913, 4.010677272036105,
        2.612307566507693, 3.303528274528758, 0.2739487675205413, 5.889753504108265,
        5.615804736587724, 2.3244356612494683, 2.1426251769710905, 1.4464475592073465,
        4.739246012566457, 14.578395360077332, 9.839149347510874, 9.413701584608935,
        3.537059747455868, 8.550410826486225, 0.008119864668922184, 0.43286611889801835,
        0.4247462542290962, 0.16687837041055423, 0.17139889490813626, 0.10898985016916385,
        0.06283228612856452, 2.6573707104881583, 2.594538424359594, 1.219602938224228,
        1.0596390454742999, 1.1120831319478008, 0.14842919284678588, 3.8473658772353794,
        3.6989366843885936, 1.4541605082183982, 1.3862277372859781, 0.8018849685838569,
        0.03542774287095215, 2.4474625422909617, 2.4120347994200095, 0.7745217539010397,
        0.9145812330586208, 0.3198646689221846, 1.552730787820203, 6.910681488641856,
        5.357950700821653, 3.615163570754227, 1.9072256165179793, 2.6702271628806185,
        14.608536589568727, 34.83222477045747, 20.223688180890715, 22.47901710732293,
        7.17674504190757, 18.641837024143584, 0.009066988883518605, 0.9185191396809959,
        0.9094521507974755, 0.4368550481994018, 0.38905942883427047, 0.48375558240695804,
        0.0012985909686158003, 0.21708593995837092, 0.21578734898975546, 0.08167977375391729,
        0.08155386250705281, 0.06036340747305611, 116.32010633156113, 217.5905751570807,
        101.27046882551957, 162.87154200548844, 41.920624308665566, 136.4664572257129,
    ])
    CompScalerStds = np.array([
        16.35781741152948, 20.189540126474725, 20.516298414514758, 16.816765336550194,
        7.966591328222124, 22.270791076753067, 21.802116630115243, 12.804546460581966,
        24.756629388687983, 13.930306216047477, 10.214535652334533, 27.801612936980938,
        39.74031558353379, 54.269739685575814, 53.70466607591569, 42.852342044453444,
        20.78341194242935, 56.28783510219931, 563.8004405882157, 732.0722574247563,
        736.2122907972664, 606.351603075103, 272.62646060896407, 810.6156779688841,
        3.0362262146833428, 3.2075174256751606, 4.0633818989245665, 2.9738244769894764,
        1.7805586029644034, 5.643243225066782, 1.1994336274579853, 0.8939013979423364,
        1.2297581799896975, 1.0066021334519983, 0.49129747526397105, 1.4159553146070951,
        31.754756468836774, 28.054241463256226, 38.16336054795611, 25.83485338379922,
        15.388376641904662, 39.67137484594156, 0.31988340032011076, 0.6833658037760536,
        0.7464197945553585, 0.4881349085029781, 0.3176591553643101, 0.8601748146737138,
        0.5864801661863596, 0.10048913710210677, 0.5836289120986499, 0.2811748167435902,
        0.2468696279341553, 0.5007375747433073, 0.37237566669029587, 1.7235989187720187,
        1.7058836077743305, 1.1558859351244697, 0.7677842566598179, 1.9203550253462733,
        2.1289400248865182, 3.5326064169848332, 3.708508303762512, 2.8709941136664567,
        1.6110681295257014, 4.310192504023775, 1.6644182118209292, 6.228287671164213,
        6.1200848808512305, 3.1986202996110302, 2.4492978142248867, 4.030497343977163,
        3.662028270049814, 6.8192125550358345, 6.614243783887738, 4.334987449618594,
        2.568319610320196, 5.9494890200106925, 0.08974370432893491, 0.4954725441517777,
        0.494304434278516, 0.2309340434963803, 0.2072873961103969, 0.31162647950590266,
        0.39805702757060923, 1.8111691089355726, 1.7973395144505941, 0.9486995373104102,
        0.7538753151875139, 1.5233177017753785, 0.7952606701778913, 3.711190225170556,
        3.638721437232604, 1.7171165424006831, 1.4307904413917036, 2.1047820817622904,
        0.49193748323158065, 4.064840532426175, 4.035286619587313, 1.4858577214526643,
        1.5799117659864677, 1.6130080156145745, 1.555249156140194, 4.776932951077492,
        4.569790780459629, 2.224617778217326, 1.7217507416156546, 2.5969733650703763,
        7.215001918238936, 19.252513469778584, 18.775394044177858, 9.447222764774764,
        6.7467931836261235, 11.106825644766616, 0.27206794253092115, 1.6449321034573106,
        1.6236282792648686, 0.8506917026741503, 0.7020945355184042, 1.2281895279350408,
        0.04134438177238229, 0.5508855867341717, 0.5486095551438679, 0.24239297524046477,
        0.2127779137935831, 0.3036750942874694, 80.06063945615361, 21.345794811194104,
        80.16475677581042, 52.58533928558554, 35.40836791039412, 85.980205895116,
    ])

    def smact_validity_check(comp, count):
        try:
            elem_symbols = tuple([Element.from_Z(int(e)).symbol for e in comp])
            space = smact.element_dictionary(elem_symbols)
            smact_elems = [e[1] for e in space.items()]
            electronegs = [e.pauling_eneg for e in smact_elems]
            ox_combos = [e.oxidation_states for e in smact_elems]
            if len(set(elem_symbols)) == 1:
                return True
            is_metal_list = [s in smact.metals for s in elem_symbols]
            if all(is_metal_list):
                return True
            threshold = np.max(count)
            for ox_states in itertools.product(*ox_combos):
                stoichs = [(c,) for c in count]
                cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
                if cn_e:
                    try:
                        electroneg_OK = pauling_test(ox_states, electronegs)
                    except TypeError:
                        electroneg_OK = True
                    if electroneg_OK:
                        return True
            return False
        except Exception:
            return False

    def structure_validity_check(crystal, cutoff=0.5):
        dist_mat = crystal.distance_matrix
        dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
        return not (dist_mat.min() < cutoff or crystal.volume < 0.1)

    class CrystalObj:
        def __init__(self, frac_coords, atom_types, lattice_matrix):
            self.constructed = False
            self.valid = False
            self.comp_valid = False
            self.struct_valid = False
            self.struct_fp = None
            self.comp_fp = None
            try:
                lat = Lattice(lattice_matrix)
                species = [Element.from_Z(int(z)).symbol for z in atom_types]
                self.structure = Structure(lat, species, frac_coords)
                self.constructed = True
                self.atom_types = [int(z) for z in atom_types]

                # Composition
                elem_counter = Counter(self.atom_types)
                composition = [(e, elem_counter[e]) for e in sorted(elem_counter.keys())]
                elems, counts = zip(*composition)
                counts = np.array(counts)
                counts = counts / np.gcd.reduce(counts)
                self.elems = elems
                self.comps = tuple(counts.astype(int).tolist())

                # Validity
                self.comp_valid = smact_validity_check(self.elems, self.comps)
                self.struct_valid = structure_validity_check(self.structure)
                self.valid = self.comp_valid and self.struct_valid

                # Fingerprints
                comp = Composition(Counter(self.atom_types))
                self.comp_fp = CompFP.featurize(comp)
                fp = _compute_struct_fp(self.structure, CrystalNNFP)
                if fp is not None:
                    self.struct_fp = fp
                else:
                    self.valid = False
            except Exception as e:
                pass

    # Build crystal objects for predictions (limit to n_eval_samples for speed)
    pred_crys = []
    for i, (fc, at, lat) in enumerate(zip(generated_frac_coords, generated_types, generated_lattices)):
        if i >= n_eval_samples:
            break
        pred_crys.append(CrystalObj(fc, at, lat))

    # Build crystal objects for ground truth (limit to n_eval_samples for speed)
    gt_crys = []
    for i, rec in enumerate(gt_data):
        if i >= n_eval_samples:
            break
        fc = rec['frac_coords'].numpy() if torch.is_tensor(rec['frac_coords']) else rec['frac_coords']
        at = rec['atom_types'].numpy() if torch.is_tensor(rec['atom_types']) else rec['atom_types']
        lat = rec['lattice'].numpy() if torch.is_tensor(rec['lattice']) else rec['lattice']
        gt_crys.append(CrystalObj(fc, at, lat))

    # Validity
    n_total = len(pred_crys)
    comp_valid = np.mean([c.comp_valid for c in pred_crys])
    struct_valid = np.mean([c.struct_valid for c in pred_crys])
    valid = np.mean([c.valid for c in pred_crys])

    metrics = {
        "comp_valid": comp_valid,
        "struct_valid": struct_valid,
        "valid": valid,
    }

    # Coverage + property stats (only if enough valid samples)
    valid_crys = [c for c in pred_crys if c.valid]
    gt_valid = [c for c in gt_crys if c.valid]
    if len(valid_crys) >= min(n_eval_samples, 100) and len(gt_valid) >= 10:
        try:
            # Density Wasserstein
            pred_densities = [c.structure.density for c in valid_crys[:n_eval_samples]]
            gt_densities = [c.structure.density for c in gt_valid]
            metrics["prop_wdist_density"] = wasserstein_distance(pred_densities, gt_densities)

            # Num elements Wasserstein
            pred_nelems = [len(set(c.structure.species)) for c in valid_crys[:n_eval_samples]]
            gt_nelems = [len(set(c.structure.species)) for c in gt_valid]
            metrics["prop_wdist_num_elems"] = wasserstein_distance(pred_nelems, gt_nelems)

            # Coverage
            struc_fps = [c.struct_fp for c in pred_crys]
            comp_fps = [c.comp_fp for c in pred_crys]
            gt_struc_fps = [c.struct_fp for c in gt_valid]
            gt_comp_fps = [c.comp_fp for c in gt_valid]

            # Filter None fps
            valid_idx = [i for i in range(len(struc_fps))
                         if struc_fps[i] is not None and comp_fps[i] is not None]
            if len(valid_idx) > 0:
                s_fps = np.array([struc_fps[i] for i in valid_idx])
                c_fps = np.array([comp_fps[i] for i in valid_idx])
                gt_s = np.array([f for f in gt_struc_fps if f is not None])
                gt_c = np.array([f for f in gt_comp_fps if f is not None])

                # Scale comp fps
                c_fps = (c_fps - CompScalerMeans) / CompScalerStds
                gt_c = (gt_c - CompScalerMeans) / CompScalerStds

                if len(gt_s) > 0 and len(s_fps) > 0:
                    cutoff = COV_Cutoffs["mp20"]
                    struc_pdist = cdist(s_fps, gt_s)
                    comp_pdist = cdist(c_fps, gt_c[:len(gt_s)])

                    struc_recall = struc_pdist.min(axis=0)
                    comp_recall = comp_pdist.min(axis=0)
                    cov_recall = np.mean(np.logical_and(
                        struc_recall <= cutoff["struc"],
                        comp_recall <= cutoff["comp"]
                    ))
                    struc_prec = struc_pdist.min(axis=1)
                    comp_prec = comp_pdist.min(axis=1)
                    cov_precision = np.sum(np.logical_and(
                        struc_prec <= cutoff["struc"],
                        comp_prec <= cutoff["comp"]
                    )) / len(pred_crys)

                    metrics["cov_recall"] = cov_recall
                    metrics["cov_precision"] = cov_precision
        except Exception as e:
            print(f"Coverage computation failed: {e}")

    return metrics


# ----- Dataset classes -----
class MoleculeDataset(Dataset):
    """Load molecule data from .pt file."""

    def __init__(self, data_dir, split="train"):
        pt_path = os.path.join(data_dir, split, "molecule_data.pt")
        self.data = torch.load(pt_path, map_location='cpu', weights_only=False)
        # Compute statistics
        self.atom_counts = [d['num_atoms'] for d in self.data]
        all_types = set()
        for d in self.data:
            all_types.update(d['atom_types'].tolist())
        self.atom_type_set = sorted(all_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CrystalDataset(Dataset):
    """Load crystal data from .pt file."""

    def __init__(self, data_dir, split="train"):
        pt_path = os.path.join(data_dir, split, "crystal_data.pt")
        self.data = torch.load(pt_path, map_location='cpu', weights_only=False)
        self.atom_counts = [d['num_atoms'] for d in self.data]
        all_types = set()
        for d in self.data:
            all_types.update(d['atom_types'].tolist())
        self.atom_type_set = sorted(all_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_molecules(batch):
    """Collate variable-size molecules into padded batch."""
    max_atoms = max(d['num_atoms'] for d in batch)
    B = len(batch)
    positions = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    mask = torch.zeros(B, max_atoms)
    num_atoms = torch.zeros(B, dtype=torch.long)

    for i, d in enumerate(batch):
        n = d['num_atoms']
        positions[i, :n] = d['positions'][:n]
        atom_types[i, :n] = d['atom_types'][:n]
        mask[i, :n] = 1.0
        num_atoms[i] = n

    return {
        'positions': positions,
        'atom_types': atom_types,
        'mask': mask,
        'num_atoms': num_atoms,
        'dataset_type': 'molecule',
    }


def collate_crystals(batch):
    """Collate variable-size crystals into padded batch."""
    max_atoms = max(d['num_atoms'] for d in batch)
    B = len(batch)
    frac_coords = torch.zeros(B, max_atoms, 3)
    atom_types = torch.zeros(B, max_atoms, dtype=torch.long)
    mask = torch.zeros(B, max_atoms)
    num_atoms = torch.zeros(B, dtype=torch.long)
    lattice = torch.zeros(B, 3, 3)

    for i, d in enumerate(batch):
        n = d['num_atoms']
        frac_coords[i, :n] = d['frac_coords'][:n]
        atom_types[i, :n] = d['atom_types'][:n]
        mask[i, :n] = 1.0
        num_atoms[i] = n
        lattice[i] = d['lattice']

    return {
        'frac_coords': frac_coords,
        'atom_types': atom_types,
        'mask': mask,
        'num_atoms': num_atoms,
        'lattice': lattice,
        'dataset_type': 'crystal',
    }


# ----- Config -----
@dataclass
class GeneratorConfig:
    dataset: str = "qm9"
    data_dir: str = "/data/qm9"
    dataset_type: str = "molecule"
    epochs: int = 1000
    batch_size: int = 64
    lr: float = 1e-4
    hidden_dim: int = 256
    num_layers: int = 6
    num_timesteps: int = 1000
    num_samples: int = 10000
    seed: int = 42
    output_dir: str = "./output"
    max_train_hours: float = 20.0


# ----- Training + Evaluation -----
def train_and_evaluate(config):
    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Data
    if config.dataset_type == "molecule":
        train_dataset = MoleculeDataset(config.data_dir, "train")
        collate_fn = collate_molecules
    else:
        train_dataset = CrystalDataset(config.data_dir, "train")
        collate_fn = collate_crystals

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True,
    )

    # Model
    model = StructureGenerator(config).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    raw_model = model.module if is_distributed else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training loop
    import time as _time
    _train_start = _time.time()
    _max_train_secs = config.max_train_hours * 3600
    for epoch in range(config.epochs):
        # Time-based early stop to leave room for sampling + evaluation
        if _time.time() - _train_start > _max_train_secs:
            if is_main:
                print(f"Time limit reached after {epoch} epochs "
                      f"({((_time.time() - _train_start) / 3600):.1f}h), stopping training.",
                      flush=True)
            break
        if is_distributed:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Move to device
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_dev[k] = v.to(device)
                else:
                    batch_dev[k] = v

            loss = raw_model.compute_loss(batch_dev)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if is_main and (epoch + 1) % 10 == 0:
            print(f"TRAIN_METRICS epoch={epoch+1} loss={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}",
                  flush=True)

    # Save model
    if is_main:
        os.makedirs(config.output_dir, exist_ok=True)
        torch.save(raw_model.state_dict(), os.path.join(config.output_dir, "model.pt"))

    # Evaluation (main process only)
    if is_main:
        raw_model.eval()
        print("Starting evaluation...", flush=True)

        # Sample atom counts from training distribution
        atom_counts_dist = train_dataset.atom_counts
        sampled_counts = np.random.choice(atom_counts_dist, size=config.num_samples, replace=True)
        atom_counts_tensor = torch.tensor(sampled_counts, dtype=torch.long, device=device)

        # Generate in batches (with per-batch timeout to avoid hangs)
        import signal
        gen_batch_size = 100
        all_results = []
        _sample_timeout = 300  # 5 min per batch
        _skipped = 0
        for i in range(0, config.num_samples, gen_batch_size):
            end = min(i + gen_batch_size, config.num_samples)
            counts = atom_counts_tensor[i:end]
            try:
                def _alarm_handler(signum, frame):
                    raise TimeoutError
                old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(_sample_timeout)
                with torch.no_grad():
                    result = raw_model.sample(
                        end - i, counts, device,
                        dataset_type=config.dataset_type,
                        atom_type_map=train_dataset.atom_type_set,
                    )
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                # Move to CPU
                result_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}
                all_results.append(result_cpu)
            except (TimeoutError, Exception) as e:
                signal.alarm(0)
                _skipped += (end - i)
                print(f"WARNING: batch {i}-{end} skipped ({e})", flush=True)
                continue

            if (i // gen_batch_size) % 10 == 0:
                print(f"Generated {end}/{config.num_samples} samples"
                      f"{f' ({_skipped} skipped)' if _skipped else ''}", flush=True)

        # Merge results (pad to max size across batches)
        if not all_results:
            print("ERROR: no samples generated", flush=True)
            sys.exit(1)
        merged = {}
        for k in all_results[0]:
            if torch.is_tensor(all_results[0][k]):
                tensors = [r[k] for r in all_results]
                if tensors[0].dim() >= 2:
                    # Pad dim=1 (max_atoms) to match across batches
                    max_dim1 = max(t.shape[1] for t in tensors)
                    padded = []
                    for t in tensors:
                        if t.shape[1] < max_dim1:
                            pad_size = list(t.shape)
                            pad_size[1] = max_dim1 - t.shape[1]
                            t = torch.cat([t, torch.zeros(pad_size, dtype=t.dtype)], dim=1)
                        padded.append(t)
                    merged[k] = torch.cat(padded, dim=0)
                else:
                    merged[k] = torch.cat(tensors, dim=0)
        n_generated = merged[list(merged.keys())[0]].shape[0]
        # Rebuild sampled_counts to match actual generated count
        _gen_counts = []
        _idx = 0
        for r in all_results:
            bs = r[list(r.keys())[0]].shape[0] if torch.is_tensor(r[list(r.keys())[0]]) else len(r[list(r.keys())[0]])
            _gen_counts.extend(sampled_counts[_idx:_idx+bs].tolist())
            _idx += bs
        sampled_counts = np.array(_gen_counts)
        print(f"Total generated: {n_generated} ({_skipped} skipped)", flush=True)

        # Evaluate
        if config.dataset_type == "molecule":
            # Extract individual molecules
            positions_list = []
            types_list = []
            for i in range(n_generated):
                n = int(sampled_counts[i])
                positions_list.append(merged['positions'][i, :n].numpy())
                types_list.append(merged['atom_types'][i, :n].numpy())

            metrics = evaluate_molecules(positions_list, types_list,
                                         dataset=config.dataset)
        else:
            # Crystal evaluation
            frac_list = []
            types_list = []
            lat_list = []
            for i in range(n_generated):
                n = int(sampled_counts[i])
                frac_list.append(merged['frac_coords'][i, :n].numpy())
                types_list.append(merged['atom_types'][i, :n].numpy())
                lat_list.append(merged['lattice'][i].numpy())

            # Load GT for coverage
            gt_data = CrystalDataset(config.data_dir, "valid").data
            metrics = evaluate_crystals(frac_list, types_list, lat_list, gt_data)

        # Print TEST_METRICS
        for k, v in metrics.items():
            print(f"TEST_METRICS {k}={v:.6f}", flush=True)

        # Save metrics
        with open(os.path.join(config.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="3D Structure Generation")
    parser.add_argument("--dataset", type=str, default="qm9")
    parser.add_argument("--data-dir", type=str, default="/data/qm9")
    parser.add_argument("--dataset-type", type=str, default="molecule",
                        choices=["molecule", "crystal"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--max-train-hours", type=float, default=20.0,
                        help="Maximum training time in hours; stops early to leave time for eval")
    args = parser.parse_args()

    config = GeneratorConfig(**vars(args))
    train_and_evaluate(config)


if __name__ == "__main__":
    main()

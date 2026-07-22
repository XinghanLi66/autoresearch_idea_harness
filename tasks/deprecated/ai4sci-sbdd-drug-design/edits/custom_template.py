"""
Structure-Based Drug Design — Custom model for pocket-conditioned molecule generation.

This model is registered as 'custom' in CBGBench and used for de novo, linker, and
fragment-based drug design tasks on CrossDocked2020.

Structure:
  Lines 1-41:    FIXED — Imports, registration, constants
  Lines 42-310:  EDITABLE — CustomSBDD model class (starter: simple EGNN + DDPM)
  Lines 311-end: (none — CBGBench handles training/sampling/evaluation)

Interface:
  forward(batch) -> (loss_dict: Dict[str, Tensor], results: Dict)
  sample(batch) -> trajectory dict {timestep: (pos, atom_type, batch_idx)}

Batch keys (PyG HeteroData flattened):
  ligand_pos:              [N_lig, 3]   — ligand atom positions
  ligand_atom_type:        [N_lig]      — ligand atom type indices (0-12, add_aromatic mode)
  ligand_lig_flag:         [N_lig]      — True for all ligand atoms
  ligand_gen_flag:         [N_lig]      — True for atoms to generate (=lig_flag in denovo)
  ligand_element_batch:    [N_lig]      — batch index per ligand atom
  protein_pos:             [N_rec, 3]   — protein atom positions
  protein_atom_feature:    [N_rec]      — protein atom features
  protein_aa_type:         [N_rec]      — amino acid type indices
  protein_lig_flag:        [N_rec]      — False for all protein atoms
  protein_element_batch:   [N_rec]      — batch index per protein atom

For linker/frag tasks: ligand_gen_flag marks which atoms to generate;
context atoms have gen_flag=False but lig_flag=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm.auto import tqdm
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import radius_graph, knn_graph
from repo.models._base import register_model
from repo.modules.common import compose_context, get_dict_mean, GaussianSmearing, SinusoidalPosEmb, MLP

NUM_ATOM_TYPES = 13  # add_aromatic mode: 13 types

# =====================================================================
# EDITABLE SECTION START — CustomSBDD model
# =====================================================================

class EGNNLayer(nn.Module):
    """E(n) Equivariant Graph Neural Network layer."""
    def __init__(self, hidden_dim, edge_feat_dim=0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_feat_dim, hidden_dim),
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

    def forward(self, h, x, edge_index, edge_attr=None):
        src, dst = edge_index
        rel_pos = x[src] - x[dst]
        dist = (rel_pos ** 2).sum(dim=-1, keepdim=True)

        edge_input = torch.cat([h[src], h[dst], dist], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)

        m_ij = self.edge_mlp(edge_input)
        agg = scatter_add(m_ij, dst, dim=0, dim_size=h.size(0))
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        coord_weight = self.coord_mlp(m_ij)
        coord_agg = scatter_add(coord_weight * rel_pos, dst, dim=0, dim_size=x.size(0))
        x_new = x + coord_agg

        return h_new, x_new


@register_model('custom')
class CustomSBDD(nn.Module):
    """Starter: Simple EGNN + DDPM for pocket-conditioned molecule generation."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.get('num_atomtype', NUM_ATOM_TYPES)
        self.num_diffusion_timesteps = cfg.generator.num_diffusion_timesteps
        hidden_dim = cfg.encoder.get('node_feat_dim', 128)
        num_layers = cfg.encoder.get('num_layers', 6)

        # Embeddings
        self.atom_emb = nn.Linear(self.num_classes, hidden_dim)
        self.protein_emb = nn.Linear(27 + 1, hidden_dim)  # atomic_numbers(27) + is_backbone
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lig_indicator = nn.Linear(1, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])

        # Output heads
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_classes),
        )

        # Noise schedule (linear beta)
        betas = torch.linspace(1e-4, 0.02, self.num_diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, batch):
        x_lig_0 = batch['ligand_pos']
        v_lig_0 = batch['ligand_atom_type']
        x_rec = batch['protein_pos']
        v_rec = batch['protein_atom_feature']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag = batch.get('ligand_gen_flag', lig_flag)
        batch_lig = batch['ligand_element_batch']
        batch_rec = batch['protein_element_batch']
        B = batch_lig.max() + 1

        if self.training:
            t = torch.randint(0, self.num_diffusion_timesteps, (B,), device=x_lig_0.device)
            return self._get_loss(x_lig_0, v_lig_0, x_rec, v_rec,
                                  lig_flag, rec_flag, gen_flag,
                                  batch_lig, batch_rec, t)
        else:
            loss_dicts = []
            results = {}
            eval_times = np.linspace(0, self.num_diffusion_timesteps - 1, 10)
            for t_val in eval_times:
                t = torch.tensor([t_val] * B).long().to(x_lig_0.device)
                loss_dict, results = self._get_loss(x_lig_0, v_lig_0, x_rec, v_rec,
                                              lig_flag, rec_flag, gen_flag,
                                              batch_lig, batch_rec, t)
                loss_dicts.append(loss_dict)
            return get_dict_mean(loss_dicts), results

    def _get_loss(self, x_lig_0, v_lig_0, x_rec, v_rec,
                  lig_flag, rec_flag, gen_flag,
                  batch_lig, batch_rec, t):
        # Add noise to positions
        t_lig = t[batch_lig]
        sqrt_alpha = self.sqrt_alphas_cumprod[t_lig].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_lig].unsqueeze(-1)
        noise_pos = torch.randn_like(x_lig_0)
        x_lig_t = sqrt_alpha * x_lig_0 + sqrt_one_minus * noise_pos
        # Only noise gen atoms
        x_lig_t = torch.where(gen_flag.unsqueeze(-1), x_lig_t, x_lig_0)

        # Add noise to atom types (uniform noise)
        c_lig_0 = F.one_hot(v_lig_0, self.num_classes).float()
        noise_type = torch.randn_like(c_lig_0)
        c_lig_t = sqrt_alpha * c_lig_0 + sqrt_one_minus * noise_type
        c_lig_t = torch.where(gen_flag.unsqueeze(-1), c_lig_t, c_lig_0)

        # Encode and denoise
        eps_pos, c_pred = self._denoise(x_lig_t, c_lig_t, x_rec, v_rec,
                                         lig_flag, rec_flag, gen_flag,
                                         batch_lig, batch_rec, t)

        # Position loss (MSE on noise)
        loss_pos = F.mse_loss(eps_pos[gen_flag], noise_pos[gen_flag])

        # Atom type loss (CE)
        loss_atom = F.cross_entropy(c_pred[gen_flag], v_lig_0[gen_flag])

        results = {
            'v0': v_lig_0[gen_flag],
            'c_pred': F.softmax(c_pred[gen_flag], dim=-1),
            'mask_gen': gen_flag[gen_flag],
        }
        return {'pos': loss_pos, 'atom': loss_atom}, results

    def _denoise(self, x_lig, c_lig, x_rec, v_rec,
                 lig_flag, rec_flag, gen_flag,
                 batch_lig, batch_rec, t):
        B = batch_lig.max() + 1

        # Embed ligand
        h_lig = self.atom_emb(c_lig)
        t_emb = self.time_emb(t[batch_lig].float())
        h_lig = h_lig + t_emb + self.lig_indicator(lig_flag.float().unsqueeze(-1))

        # Embed protein
        if v_rec.dim() == 1:
            v_rec_oh = F.one_hot(v_rec.long(), num_classes=28).float()
        else:
            v_rec_oh = v_rec.float()
            if v_rec_oh.shape[-1] < 28:
                v_rec_oh = F.pad(v_rec_oh, (0, 28 - v_rec_oh.shape[-1]))
        h_rec = self.protein_emb(v_rec_oh)
        h_rec = h_rec + self.time_emb(t[batch_rec].float())
        h_rec = h_rec + self.lig_indicator(rec_flag.float().unsqueeze(-1))

        # Compose context
        ctx_lig = {'x': x_lig, 'h': h_lig, 'gen_flag': gen_flag, 'lig_flag': lig_flag}
        ctx_rec = {'x': x_rec, 'h': h_rec, 'gen_flag': torch.zeros_like(rec_flag), 'lig_flag': rec_flag}
        composed, batch_idx, _ = compose_context(ctx_lig, ctx_rec, batch_lig, batch_rec)

        x_all = composed['x']
        h_all = composed['h']

        # Build knn graph
        edge_index = knn_graph(x_all, k=32, batch=batch_idx, flow='source_to_target')

        # Run EGNN layers
        for layer in self.layers:
            h_all, x_all = layer(h_all, x_all, edge_index)

        # Extract ligand outputs
        h_lig_out = h_all[composed['lig_flag']]
        x_lig_out = x_all[composed['lig_flag']]

        # Predict noise and atom type
        eps_pred = self.pos_head(h_lig_out)
        c_pred = self.atom_head(h_lig_out)

        return eps_pred, c_pred

    def sample(self, batch):
        x_lig_t = batch['ligand_pos']
        v_lig_in = batch['ligand_atom_type']
        x_rec = batch['protein_pos']
        v_rec = batch['protein_atom_feature']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag = batch.get('ligand_gen_flag', lig_flag)
        batch_lig = batch['ligand_element_batch']
        batch_rec = batch['protein_element_batch']
        B = batch_lig.max() + 1

        c_lig_t = F.one_hot(v_lig_in, self.num_classes).float()

        traj = {self.num_diffusion_timesteps - 1: (x_lig_t, c_lig_t, batch_lig)}

        for t_idx in tqdm(reversed(range(self.num_diffusion_timesteps)),
                          desc='sampling', total=self.num_diffusion_timesteps):
            x_lig, c_lig, _ = traj[t_idx]
            t = torch.full((B,), t_idx, dtype=torch.long, device=x_lig.device)

            eps_pred, c_pred = self._denoise(x_lig, c_lig, x_rec, v_rec,
                                              lig_flag, rec_flag, gen_flag,
                                              batch_lig, batch_rec, t)

            # DDPM reverse step for positions
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]
            coeff = beta_t / self.sqrt_one_minus_alphas_cumprod[t_idx]

            x_mean = (1.0 / alpha_t.sqrt()) * (x_lig - coeff * eps_pred)
            if t_idx > 0:
                noise = torch.randn_like(x_lig)
                x_next = x_mean + beta_t.sqrt() * noise
            else:
                x_next = x_mean
            x_next = torch.where(gen_flag.unsqueeze(-1), x_next, x_lig)

            # Atom type: blend prediction with current
            c_next = c_pred.softmax(dim=-1)
            c_next = torch.where(gen_flag.unsqueeze(-1), c_next, c_lig)

            traj[t_idx - 1] = (x_next, c_next, batch_lig)
            traj[t_idx] = tuple(v.cpu() for v in traj[t_idx])

        return traj

# =====================================================================
# EDITABLE SECTION END
# =====================================================================

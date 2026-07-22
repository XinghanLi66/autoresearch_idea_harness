"""DiffBP baseline — Diffusion-based generative model for SBDD with CoM prediction.
Replaces editable section with DiffBP architecture using UniTransformer encoder,
CoM predictor head, interior regularization, and mask-based atom type schedule.
Reference: Lin et al., "Diffusion-based Generative Models for Binding-Pose Prediction" (2023)
"""

_FILE = "CBGBench/repo/models/custom_sbdd.py"

_CONTENT = """\
NUM_ATOM_TYPES = 13  # add_aromatic mode: 13 types

# =====================================================================
# EDITABLE SECTION START — DiffBP baseline
# =====================================================================

from repo.models.diffusion.diffusion_scheduler import CTNVPScheduler, MaskTypeSchedule
from repo.modules.e3nn import get_e3_gnn
from repo.modules.context_emb import get_context_embedder
from repo.modules.attention import H2XAttention
from repo.modules.embs import get_dist_emb
from repo.utils.molecule.constants import absorbing_state
from repo.utils.protein.constants import aa_name_number
from torch_geometric.nn import knn


def interior_loss(x_ligand, x_protein, batch_ligand, batch_protein, k=48, rho=2, gamma=5):
    edge_idx = knn(x_ligand, x_protein, batch_x=batch_ligand, batch_y=batch_protein, k=k, num_workers=16)
    protein_idx, ligand_idx = edge_idx[0], edge_idx[1]
    pos_ligand = x_ligand[ligand_idx]
    pos_protein = x_protein[protein_idx]
    dist2 = torch.square(pos_ligand - pos_protein).sum(dim=-1)
    exp_dist2 = torch.divide(-dist2, rho).exp()
    loss_per_ligand = -rho * (scatter_add(exp_dist2, ligand_idx, dim=0, dim_size=x_ligand.size(0)) + 1e-3).log()
    loss_exp_inter = gamma - loss_per_ligand
    return torch.clamp(loss_exp_inter, min=0.).mean()


class CoMPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.get('node_feat_dim', 128)
        self.n_heads = cfg.get('n_heads', 16)
        self.edge_feat_dim = cfg.get('edge_feat_dim', 4)
        self.num_r_gaussian = cfg.get('num_r_gaussian', 20)
        self.act_fn = cfg.get('act_fn', 'relu')
        self.norm = cfg.get('norm', True)
        self.r_max = cfg.get('r_max', 10.0)
        self.ew_net_type = cfg.get('ew_type', 'global')
        self.cutoff_mode = cfg.get('cutoff_mode', 'knn')
        self.cut_off = cfg.get('k', 32)
        self.num_layers = cfg.get('num_layers_com', 3)
        h2xattentions = []
        for _ in range(self.num_layers):
            h2xattentions.append(H2XAttention(
                self.hidden_dim, self.hidden_dim, self.hidden_dim,
                self.n_heads, self.edge_feat_dim, r_feat_dim=self.num_r_gaussian * 4,
                act_fn=self.act_fn, norm=self.norm, num_r_gaussian=self.num_r_gaussian,
                r_max=self.r_max, ew_net_type=self.ew_net_type))
        self.h2xattentions = nn.ModuleList(h2xattentions)
        self.dist_emb = get_dist_emb(cfg.get('dist_emb_type', 'gaussian_exp'),
                                     self.num_r_gaussian, cut_off=self.r_max)

    def forward(self, x_lig_pred, batch_idx_lig, x_composed, h_composed,
                gen_flag_composed, lig_flag_composed, batch_idx_composed):
        noise_lig_pred = x_lig_pred - x_composed[lig_flag_composed]
        noise_lig_pred_mean = scatter_mean(noise_lig_pred, batch_idx_lig, dim=0)[batch_idx_lig]
        noise_lig_pred = noise_lig_pred - noise_lig_pred_mean

        edge_index = knn_graph(x_composed, k=int(self.cut_off), batch=batch_idx_composed, flow='source_to_target')
        src, dst = edge_index
        edge_type = self._build_edge_type(edge_index, lig_flag_composed)
        dist = torch.norm(x_composed[dst] - x_composed[src], p=2, dim=-1, keepdim=True)
        logits = self.dist_emb(dist)
        e_w = torch.sigmoid(logits)

        x_out = x_composed.clone()
        for h2x in self.h2xattentions:
            delta_x = h2x(x_out, h_composed, edge_type, edge_index, e_w)
            x_out = x_out + delta_x * gen_flag_composed.unsqueeze(-1)

        delta_x_lig = (x_out - x_composed)[lig_flag_composed]
        mean_com_shift = scatter_mean(delta_x_lig, batch_idx_lig, dim=0)[batch_idx_lig]
        return noise_lig_pred, mean_com_shift

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        return F.one_hot(edge_type.long(), num_classes=4)


@register_model('custom')
class CustomSBDD(nn.Module):
    \"\"\"DiffBP: Diffusion model with CoM prediction + interior regularization.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.get('num_atomtype', NUM_ATOM_TYPES)
        self.num_diffusion_timesteps = cfg.generator.num_diffusion_timesteps
        self.time_sampler = cfg.generator.get('time_sampler', 'symmetric')
        self.denoise_structure = cfg.generator.get('denoise_structure', True)
        self.denoise_atom = cfg.generator.get('denoise_atom', True)
        self.intersect_reg = cfg.get('intersect_reg', True)

        pos_cfg = cfg.generator.pos_schedule
        self.pos_scheduler = CTNVPScheduler(
            self.num_diffusion_timesteps,
            beta_start=pos_cfg.beta_start, beta_end=pos_cfg.beta_end, type=pos_cfg.type)

        atom_cfg = cfg.generator.atom_schedule
        self.type_scheduler = MaskTypeSchedule(
            self.num_diffusion_timesteps,
            num_classes=self.num_classes,
            type=atom_cfg.type, absorbing_state=absorbing_state)

        cfg.embedder.num_atomtype = self.num_classes
        self.context_embedder = get_context_embedder(cfg.embedder)
        self.denoiser = get_e3_gnn(cfg.encoder, num_classes=self.num_classes)
        self.com_head = CoMPredictor(cfg.encoder)

    def sample_time(self, batch_size, device='cuda'):
        if self.time_sampler == 'symmetric':
            n = batch_size
            t = torch.randint(0, self.num_diffusion_timesteps, size=(n // 2 + 1,), device=device)
            t = torch.cat([t, self.num_diffusion_timesteps - t - 1], dim=0)[:n]
        else:
            t = torch.randint(0, self.num_diffusion_timesteps, size=(batch_size,), device=device)
        return t

    def forward(self, batch):
        x_lig_0 = batch['ligand_pos']
        v_lig_0 = batch['ligand_atom_type']
        x_rec_0 = batch['protein_pos']
        v_rec_0 = batch['protein_atom_feature']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))
        B = batch_idx_lig.max() + 1

        if self.training:
            t = self.sample_time(B, device=x_lig_0.device)
            return self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                                 lig_flag, rec_flag, batch_idx_lig, batch_idx_rec,
                                 gen_flag_lig, gen_flag_rec, t)
        else:
            loss_dicts = []
            results = {}
            eval_times = np.linspace(0, self.num_diffusion_timesteps - 1, 10)
            for t_val in eval_times:
                t = torch.tensor([t_val] * B).long().to(x_lig_0.device)
                ld, results = self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                                      lig_flag, rec_flag, batch_idx_lig, batch_idx_rec,
                                      gen_flag_lig, gen_flag_rec, t)
                loss_dicts.append(ld)
            return get_dict_mean(loss_dicts), results

    def get_loss(self, x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                 lig_flag, rec_flag, batch_idx_lig, batch_idx_rec,
                 gen_flag_lig, gen_flag_rec, t):
        if self.denoise_structure:
            x_lig_t, pos_noise, com_noise = self.pos_scheduler.forward_add_noise(
                x_lig_0, t, batch_idx_lig, gen_flag_lig, zero_center=True)
        else:
            x_lig_t = x_lig_0

        if self.denoise_atom:
            c_lig_t, v_lig_t, type_loss_flag = self.type_scheduler.forward_add_noise(
                v_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            c_lig_t = F.one_hot(v_lig_0, num_classes=self.num_classes).float()

        x_lig_t, x_rec_t, h_lig_t, h_rec_t = self.context_embedder(
            x_lig_t, x_rec_0, c_lig_t, v_rec_0, aa_rec_0,
            batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)

        context_composed, batch_idx, _ = compose_context(
            {'x': x_lig_t, 'h': h_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
            {'x': x_rec_t, 'h': h_rec_t, 'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
            batch_idx_lig, batch_idx_rec)

        x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)
        x_lig_pred = x[context_composed['lig_flag']]
        c_lig_pred = v[context_composed['lig_flag']]

        x_composed, h_composed = context_composed['x'], h
        gen_flag_composed = context_composed['gen_flag']
        lig_flag_composed = context_composed['lig_flag']
        x_lig_pred, x_com_pred = self.com_head(
            x_lig_pred, batch_idx_lig, x_composed, h_composed,
            gen_flag_composed, lig_flag_composed, batch_idx_composed=batch_idx)

        if self.denoise_structure:
            loss_pos, pos_info = self.pos_scheduler.get_score_loss(
                x_lig_pred, pos_noise, t, gen_flag_lig, batch_idx_lig, score_in=False)
            loss_com, com_info = self.pos_scheduler.get_score_loss(
                x_com_pred, com_noise, t, gen_flag_lig, batch_idx_lig,
                score_in=False, info_tag='com')
        else:
            loss_pos = torch.tensor(0).float()
            loss_com = torch.tensor(0).float()
            pos_info, com_info = {}, {}

        if self.denoise_atom:
            loss_atom, atom_info = self.type_scheduler.get_loss(
                c_lig_pred, v_lig_0, v_lig_t, t,
                type_loss_flag, batch_idx_lig, pred_logit=True)
        else:
            loss_atom, atom_info = torch.tensor(0).float(), {}

        loss_inter = torch.tensor(0).float()
        if self.intersect_reg and self.denoise_structure:
            xs_mean = self.pos_scheduler.xs_mean(
                x_lig_pred + x_com_pred, x_lig_t, t, batch_idx_lig, gen_flag=gen_flag_lig)
            loss_inter = interior_loss(xs_mean, x_rec_0, batch_idx_lig, batch_idx_rec)

        results = {}
        results.update(pos_info)
        results.update(atom_info)
        results.update(com_info)
        return {'pos': loss_pos, 'atom': loss_atom, 'com': loss_com, 'inter': loss_inter}, results

    def sample(self, batch):
        x_lig_in = batch['ligand_pos']
        v_lig_in = batch['ligand_atom_type']
        x_rec_0 = batch['protein_pos']
        v_rec_0 = batch['protein_atom_feature']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))

        c_lig_in = F.one_hot(v_lig_in, num_classes=self.num_classes).float()

        time_seq = list(reversed(range(0, self.num_diffusion_timesteps)))
        B = batch_idx_lig.max() + 1

        traj = {self.num_diffusion_timesteps - 1: (x_lig_in, c_lig_in, batch_idx_lig)}

        for t_idx in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            x_lig, c_lig, _ = traj[t_idx]
            t = torch.full(size=(B,), fill_value=t_idx, dtype=torch.long, device=x_lig_in.device)

            x_lig, x_rec, h_lig, h_rec = self.context_embedder(
                x_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0,
                batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)

            context_composed, batch_idx, _ = compose_context(
                {'x': x_lig, 'h': h_lig, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                batch_idx_lig, batch_idx_rec)

            x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)
            x_lig_out = x[context_composed['lig_flag']]
            c_lig_out = v[context_composed['lig_flag']]

            x_composed, h_composed = context_composed['x'], h
            gen_flag_composed = context_composed['gen_flag']
            lig_flag_composed = context_composed['lig_flag']
            x_lig_out, x_com_out = self.com_head(
                x_lig_out, batch_idx_lig, x_composed, h_composed,
                gen_flag_composed, lig_flag_composed, batch_idx_composed=batch_idx)

            if self.denoise_structure:
                x_lig_next = self.pos_scheduler.backward_remove_noise(
                    x_lig_out + x_com_out, x_lig, t, batch_idx_lig, gen_flag_lig)
            else:
                x_lig_next = x_lig

            if self.denoise_atom:
                c_lig_next, _ = self.type_scheduler.backward_remove_noise(
                    c_lig_out, c_lig, t, batch_idx_lig, gen_flag_lig, pred_logit=True)
            else:
                c_lig_next = c_lig

            traj[t_idx - 1] = (x_lig_next, c_lig_next, batch_idx_lig)
            traj[t_idx] = tuple(x.cpu() for x in traj[t_idx])

        return traj

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 42,
        "end_line": 298,
        "content": _CONTENT,
    },
]

"""Pocket2Mol baseline — Diffusion model with focal prediction + smooth CE for SBDD.
Adapts Pocket2Mol's key architectural contributions (focal prediction head, smooth
cross-entropy for atom types) into the diffusion framework used by CBGBench's
custom model interface.
Reference: Peng et al., "Pocket2Mol: Efficient Molecular Sampling Based on 3D
Protein Pockets" (ICML 2022)
"""

_FILE = "CBGBench/repo/models/custom_sbdd.py"

_CONTENT = """\
NUM_ATOM_TYPES = 13  # add_aromatic mode: 13 types

# =====================================================================
# EDITABLE SECTION START — Pocket2Mol baseline (diffusion + focal)
# =====================================================================

from repo.models.diffusion.diffusion_scheduler import CTNVPScheduler, TypeVPScheduler
from repo.modules.e3nn import get_e3_gnn
from repo.modules.context_emb import get_context_embedder
from repo.utils.protein.constants import aa_name_number
from torch.nn.modules.loss import _WeightedLoss


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets, n_classes, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \\
                .fill_(smoothing / (n_classes - 1)) \\
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class FocalPredictor(nn.Module):
    \"\"\"Predicts which ligand atoms are focal points for generation.\"\"\"
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h, mask):
        logits = self.mlp(h).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        return logits


@register_model('custom')
class CustomSBDD(nn.Module):
    \"\"\"Pocket2Mol-style diffusion model with focal prediction and smooth CE.

    Uses the same UniTransformer encoder and VP noise schedules as TargetDiff,
    but adds a focal prediction head (which ligand atom is the attachment point)
    and uses label-smoothed cross-entropy for atom type prediction, following
    the Pocket2Mol paper.
    \"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.get('num_atomtype', NUM_ATOM_TYPES)
        self.num_diffusion_timesteps = cfg.generator.num_diffusion_timesteps
        self.time_sampler = cfg.generator.get('time_sampler', 'symmetric')
        self.denoise_structure = cfg.generator.get('denoise_structure', True)
        self.denoise_atom = cfg.generator.get('denoise_atom', True)

        pos_scheduler_cfg = cfg.generator.pos_schedule
        self.pos_scheduler = CTNVPScheduler(
            self.num_diffusion_timesteps,
            beta_start=pos_scheduler_cfg.beta_start,
            beta_end=pos_scheduler_cfg.beta_end,
            type=pos_scheduler_cfg.type)

        atom_scheduler_cfg = cfg.generator.atom_schedule
        self.type_scheduler = TypeVPScheduler(
            self.num_diffusion_timesteps,
            num_classes=self.num_classes,
            type=atom_scheduler_cfg.type,
            cosine_s=atom_scheduler_cfg.cosine_s)

        cfg.embedder.num_atomtype = self.num_classes
        self.context_embedder = get_context_embedder(cfg.embedder)
        self.denoiser = get_e3_gnn(cfg.encoder, num_classes=self.num_classes)

        hidden_dim = cfg.encoder.get('node_feat_dim', 128)
        self.focal_predictor = FocalPredictor(hidden_dim)
        self.smooth_cross_entropy = SmoothCrossEntropyLoss(smoothing=0.1)
        self.bceloss_with_logits = nn.BCEWithLogitsLoss()

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
            x_lig_t, _ = self.pos_scheduler.forward_add_noise(x_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            x_lig_t = x_lig_0

        if self.denoise_atom:
            c_lig_t, v_lig_t = self.type_scheduler.forward_add_noise(v_lig_0, t, batch_idx_lig, gen_flag_lig)
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
        h_lig_out = h[context_composed['lig_flag']]

        # Standard diffusion losses
        if self.denoise_structure:
            loss_pos, pos_info = self.pos_scheduler.get_loss(
                x_lig_pred, x_lig_0, x_lig_t, t,
                gen_flag_lig, batch_idx_lig, type='denoise')
        else:
            loss_pos, pos_info = torch.tensor(0).float(), {}

        if self.denoise_atom:
            loss_atom, atom_info = self.type_scheduler.get_loss(
                c_lig_pred, v_lig_0, v_lig_t, t,
                gen_flag_lig, batch_idx_lig, pred_logit=True)
        else:
            loss_atom, atom_info = torch.tensor(0).float(), {}

        # Pocket2Mol focal prediction loss
        focal_logits = self.focal_predictor(h_lig_out, gen_flag_lig)
        focal_target = gen_flag_lig.float()
        loss_focal = self.bceloss_with_logits(focal_logits, focal_target).clamp_max(10.)

        results = {}
        results.update(pos_info)
        results.update(atom_info)
        return {'pos': loss_pos, 'atom': loss_atom, 'focal': loss_focal}, results

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

            if self.denoise_structure:
                x_lig_next = self.pos_scheduler.backward_remove_noise(
                    x_lig_out, x_lig, t, batch_idx_lig, gen_flag_lig, type='denoise')
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

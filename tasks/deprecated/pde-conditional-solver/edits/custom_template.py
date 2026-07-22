import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from layers.Basic import MLP
from layers.Embedding import timestep_embedding, unified_pos_embedding


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Custom'
        self.args = args
        ## embedding
        if args.unified_pos and args.geotype != 'unstructured':
            self.pos = unified_pos_embedding(args.shapelist, args.ref)
            self.preprocess = MLP(args.fun_dim + args.ref ** len(args.shapelist), args.n_hidden * 2,
                                  args.n_hidden, n_layers=0, res=False, act=args.act)
        else:
            self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                                  n_layers=0, res=False, act=args.act)

        # Time conditioning: converts scalar timestep T to hidden-dim embedding
        if args.time_input:
            self.time_fc = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.SiLU(),
                                         nn.Linear(args.n_hidden, args.n_hidden))

        # TODO: Define your custom model architecture here.
        # This model predicts solutions at arbitrary timesteps via time conditioning T.
        # The time embedding is added to spatial features before the main processing.
        # args.shapelist = [101, 31] for the plasticity grid.

        self.placeholder = nn.Parameter((1 / args.n_hidden) * torch.rand(args.n_hidden, dtype=torch.float))
        self.output_layer = nn.Linear(args.n_hidden, args.out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, T=None, geo=None):
        if self.args.unified_pos and self.args.geotype != 'unstructured':
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        # Add time conditioning
        if T is not None and self.args.time_input:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        # TODO: Implement your custom forward pass here.
        # Input fx has shape (B, N, n_hidden) where N = 101*31 = 3131.
        # Time information is already added to fx via the embedding above.
        # Output should have shape (B, N, out_dim) where out_dim=4.

        out = self.output_layer(fx)
        return out


# =====================================================================
# FIXED: Parameter budget check — do not modify below this line
# =====================================================================
_orig_init = Model.__init__

def _patched_init(self, args):
    _orig_init(self, args)
    h = args.n_hidden
    nl = args.n_layers
    mr = getattr(args, 'mlp_ratio', 2)
    od = args.out_dim
    sn = getattr(args, 'slice_num', 32)
    phys_attn = 3 * h * h + h * h + h + sn * h * 2 + sn * h + sn * 2
    mlp_b = h * h * mr + h * mr + h * mr * h + h
    block = h * 2 + phys_attn + h * 2 + mlp_b
    last_extra = h * 2 + h * od + od
    if args.unified_pos and args.geotype != 'unstructured':
        in_dim = args.fun_dim + args.ref ** len(args.shapelist)
    else:
        in_dim = args.fun_dim + args.space_dim
    pre = in_dim * h * 2 + h * 2 + h * 2 * h + h
    time_fc = (h * h + h + h * h + h) if getattr(args, 'time_input', False) else 0
    _budget = int((pre + time_fc + nl * block + last_extra + h + 2000) * 1.05)
    _total = sum(p.numel() for p in self.parameters())
    print(f"Total params: {_total:,} (budget: {_budget:,})")

Model.__init__ = _patched_init

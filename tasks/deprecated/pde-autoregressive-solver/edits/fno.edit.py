"""FNO baseline — rigorous codebase edit ops.
Replaces entire Custom.py with FNO model implementation.
"""

_FILE = "Neural-Solver-Library/models/Custom.py"

_CONTENT = """\
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from layers.Basic import MLP
from layers.Embedding import timestep_embedding, unified_pos_embedding
from layers.FNO_Layers import SpectralConv1d, SpectralConv2d, SpectralConv3d

BlockList = [None, SpectralConv1d, SpectralConv2d, SpectralConv3d]
ConvList = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]


class Model(nn.Module):
    def __init__(self, args, s1=96, s2=96):
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
        if args.time_input:
            self.time_fc = nn.Sequential(nn.Linear(args.n_hidden, args.n_hidden), nn.SiLU(),
                                         nn.Linear(args.n_hidden, args.n_hidden))
        self.padding = [(16 - size % 16) % 16 for size in args.shapelist]
        self.conv0 = BlockList[len(self.padding)](args.n_hidden, args.n_hidden,
                                                  *[args.modes for _ in range(len(self.padding))])
        self.conv1 = BlockList[len(self.padding)](args.n_hidden, args.n_hidden,
                                                  *[args.modes for _ in range(len(self.padding))])
        self.conv2 = BlockList[len(self.padding)](args.n_hidden, args.n_hidden,
                                                  *[args.modes for _ in range(len(self.padding))])
        self.conv3 = BlockList[len(self.padding)](args.n_hidden, args.n_hidden,
                                                  *[args.modes for _ in range(len(self.padding))])
        self.w0 = ConvList[len(self.padding)](args.n_hidden, args.n_hidden, 1)
        self.w1 = ConvList[len(self.padding)](args.n_hidden, args.n_hidden, 1)
        self.w2 = ConvList[len(self.padding)](args.n_hidden, args.n_hidden, 1)
        self.w3 = ConvList[len(self.padding)](args.n_hidden, args.n_hidden, 1)
        self.fc1 = nn.Linear(args.n_hidden, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, args.out_dim)

    def forward(self, x, fx, T=None, geo=None):
        B, N, _ = x.shape
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb
        x = fx.permute(0, 2, 1).reshape(B, self.args.n_hidden, *self.args.shapelist)
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 1:
                x = F.pad(x, [0, self.padding[0]])
            elif len(self.args.shapelist) == 2:
                x = F.pad(x, [0, self.padding[1], 0, self.padding[0]])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 1:
                x = x[..., :-self.padding[0]]
            elif len(self.args.shapelist) == 2:
                x = x[..., :-self.padding[0], :-self.padding[1]]
        x = x.reshape(B, self.args.n_hidden, -1).permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 1,
        "end_line": 65,
        "content": _CONTENT,
    },
]

"""Graph-temporal forecasting model with editable spatial message passing.

Fixed: temporal backbone (dilated causal conv), adjacency loading, output projection.
Editable: SpatialLayer -- the graph message passing component (lines 72--130).
"""
import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


# ==============================================================================
# FIXED -- Configuration
# ==============================================================================

@dataclass
class CustomConfig(BasicTSModelConfig):
    """Configuration for the graph-temporal forecasting model."""
    input_len: int = field(default=12, metadata={"help": "Input sequence length."})
    output_len: int = field(default=12, metadata={"help": "Output sequence length."})
    num_features: int = field(default=207, metadata={"help": "Number of spatial nodes."})
    hidden_dim: int = field(default=32, metadata={"help": "Hidden dimension size."})
    num_st_blocks: int = field(default=2, metadata={"help": "Number of ST blocks."})
    temporal_kernel: int = field(default=3, metadata={"help": "Temporal conv kernel size."})
    dropout: float = field(default=0.3, metadata={"help": "Dropout rate."})
    adj_path: str = field(default="", metadata={"help": "Path to adjacency matrix."})


# ==============================================================================
# FIXED -- Adjacency matrix utilities
# ==============================================================================

def load_adj_matrix(adj_path: str, num_nodes: int) -> torch.Tensor:
    """Load adjacency matrix from pickle or generate identity if unavailable."""
    if adj_path and os.path.exists(adj_path):
        with open(adj_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        # Handle different pickle formats: (ids, id2ind, adj), (adj,), or raw array
        if isinstance(data, (list, tuple)):
            adj_mx = data[-1]  # adjacency is always last element
        else:
            adj_mx = data
        import numpy as _np
        adj_mx = _np.array(adj_mx, dtype=_np.float32)
        adj = torch.tensor(adj_mx, dtype=torch.float32)
    else:
        adj = torch.eye(num_nodes)
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = adj.sum(dim=1).clamp(min=1e-8)
    deg_inv_sqrt = deg.pow(-0.5)
    adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    return adj_norm


# ==============================================================================
# FIXED -- Temporal Convolution Block
# ==============================================================================

class TemporalConv(nn.Module):
    """Gated temporal convolution with dilated causal convolutions."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        # Causal padding to preserve temporal dimension
        self.causal_pad = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv1d(in_dim, out_dim, kernel_size,
                                     dilation=dilation, padding=0)
        self.gate_conv = nn.Conv1d(in_dim, out_dim, kernel_size,
                                   dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D, T] -> [B, N, D_out, T]"""
        B, N, D, T = x.shape
        x = x.reshape(B * N, D, T)
        # Left-pad for causal convolution (preserves temporal dim)
        x = F.pad(x, (self.causal_pad, 0))
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        _, D2, T2 = h.shape
        return h.reshape(B, N, D2, T2)


# ==============================================================================
# EDITABLE -- Spatial Message Passing Layer (lines 94--127)
# ==============================================================================

class SpatialLayer(nn.Module):
    """Graph message passing layer for spatial aggregation.

    This is the editable component. Design a novel spatial aggregation
    mechanism that operates on graph-structured traffic data.

    Interface contract:
        __init__(self, in_dim, out_dim, dropout=0.0):
            in_dim:  input feature dimension per node
            out_dim: output feature dimension per node
            dropout: dropout rate

        forward(self, x, adj):
            x:   [B, N, D]  -- node features (B=batch, N=nodes, D=features)
            adj: [N, N]     -- normalized adjacency matrix
            returns: [B, N, D_out] -- updated node features

    The adjacency matrix adj is a symmetric-normalized weighted matrix
    derived from sensor distances. Nodes without connections have adj[i,j]=0.
    Self-loops are included (diagonal is nonzero).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], adj: [N, N] -> returns [B, N, out_dim]
        # Default: simple linear projection (no graph aggregation)
        # TODO: Implement graph message passing
        out = self.linear(x)
        out = self.dropout(out)
        return out


# ==============================================================================
# FIXED -- Spatial-Temporal Block
# ==============================================================================

class STBlock(nn.Module):
    """One spatial-temporal block: TemporalConv -> SpatialLayer -> TemporalConv."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.tcn1 = TemporalConv(in_dim, hidden_dim, kernel_size, dilation)
        self.spatial = SpatialLayer(hidden_dim, hidden_dim, dropout)
        self.tcn2 = TemporalConv(hidden_dim, out_dim, kernel_size, dilation)
        self.residual = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D, T] -> [B, N, D_out, T_out]"""
        # First temporal convolution
        h = self.tcn1(x)                   # [B, N, H, T1]
        B, N, H, T1 = h.shape

        # Spatial message passing (apply at each time step)
        h_list = []
        for t in range(T1):
            h_t = self.spatial(h[:, :, :, t], adj)   # [B, N, H]
            h_list.append(h_t)
        h = torch.stack(h_list, dim=-1)    # [B, N, H, T1]

        # Second temporal convolution
        h = self.tcn2(h)                   # [B, N, D_out, T2]
        _, _, D2, T2 = h.shape

        # Residual connection (align temporal dim)
        res = x[:, :, :, -T2:]             # [B, N, D, T2]
        res = res.reshape(B * N, -1, T2)
        res = self.residual(res).reshape(B, N, D2, T2)

        out = self.dropout(h + res)
        out = self.norm(out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return out


# ==============================================================================
# FIXED -- Full Model
# ==============================================================================

class Custom(nn.Module):
    """Graph-temporal traffic forecasting model.

    Architecture: Input projection -> [STBlock x K] -> Output projection
    Each STBlock: TemporalConv -> SpatialLayer (editable) -> TemporalConv

    The model loads the pre-computed adjacency matrix at initialization.
    The SpatialLayer class is the only editable component.
    """

    def __init__(self, config: CustomConfig):
        super().__init__()
        N = config.num_features
        D = config.hidden_dim
        K = config.num_st_blocks
        T = config.input_len
        T_out = config.output_len
        ks = config.temporal_kernel
        dp = config.dropout

        # Load adjacency matrix
        self.register_buffer("adj", load_adj_matrix(config.adj_path, N))

        # Input projection: [B, T, N] -> [B, N, D, T]
        self.input_proj = nn.Linear(1, D)

        # ST blocks with increasing dilation
        self.st_blocks = nn.ModuleList()
        for i in range(K):
            dilation = 2 ** i
            in_d = D
            self.st_blocks.append(STBlock(in_d, D, D, ks, dilation, dp))

        # Output projection
        # After K blocks of temporal convolution, time dim shrinks.
        # We use adaptive pooling + linear to produce output_len predictions.
        self.output_fc = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, T_out),
        )

        # ── Parameter Budget Check ──
        # Budget = 1.05x total model params with largest baseline spatial layer (GWNet).
        # GWNet SpatialLayer per block: E1(N*10) + E2(N*10) + fc(2*K_hop*D*D + D)
        #   where K_hop=3, embed_dim=10.
        # Spatial params vary; fixed parts (temporal convs, projections) are constant.
        _spatial_params = sum(
            p.numel() for block in self.st_blocks
            for p in block.spatial.parameters()
        )
        _total_params = sum(p.numel() for p in self.parameters())
        _fixed_params = _total_params - _spatial_params
        _K_hop = 3
        _embed_dim = 10
        _gwnet_spatial_per_block = 2 * N * _embed_dim + 2 * _K_hop * D * D + D
        _gwnet_spatial_total = K * _gwnet_spatial_per_block
        _param_budget = int((_fixed_params + _gwnet_spatial_total) * 1.05)
        print(f"Model parameters: {_total_params:,} (budget: {_param_budget:,})", flush=True)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, T, N] -- historical traffic data
            inputs_timestamps: [B, T, 2] -- time features (unused here, available for extensions)
        Returns:
            prediction: [B, T_out, N]
        """
        B, T, N = inputs.shape

        # [B, T, N] -> [B, N, T, 1] -> [B, N, D, T] via input projection
        x = inputs.permute(0, 2, 1).unsqueeze(-1)  # [B, N, T, 1]
        x = self.input_proj(x)                      # [B, N, T, D]
        x = x.permute(0, 1, 3, 2)                   # [B, N, D, T]

        # Apply ST blocks
        for block in self.st_blocks:
            x = block(x, self.adj)

        # [B, N, D, T_out] -> average over remaining time -> [B, N, D]
        x = x.mean(dim=-1)

        # Output projection: [B, N, D] -> [B, N, T_out] -> [B, T_out, N]
        out = self.output_fc(x)                      # [B, N, T_out]
        return out.permute(0, 2, 1)                  # [B, T_out, N]

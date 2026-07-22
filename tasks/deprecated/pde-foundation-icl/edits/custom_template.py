"""Custom model for multi-physics PDE prediction via in-context learning.

This model receives paired demonstrations of PDE evolution (input state -> output state)
and uses them to predict the output for a new query input, without any fine-tuning.
This is in-context operator learning: the model learns to infer the underlying PDE
dynamics from the demonstrations provided at inference time.

The model is trained jointly on 3 heterogeneous fluid dynamics datasets:
  - NS2D: Incompressible Navier-Stokes (PDEArena), 128x128, channels=[vx, vy, scalar]
  - COMPRESSIBLE2D: Compressible NS high-viscosity (PDEBench), 128x128, channels=[density, vx, vy, pressure]
  - EULER2D: Compressible NS low-viscosity (PDEBench), 128x128, channels=[density, vx, vy, pressure]

Each dataset has 7 channels total (some masked via c_mask): [density, vx, vy, pressure, vorticity, scalar, node_type].

Interface:
  forward(x_tuple) where x_tuple = (init, end, c_mask)
    - init: (B, pairs, C, H, W) — input states of demonstration + query pairs
    - end:  (B, pairs, C, H, W) — output states of demonstration pairs (query output = prediction target)
    - c_mask: (B, C) — channel mask indicating which channels are valid (1=valid, 0=masked)
  returns: (B, pairs, C, H, W) — predicted output states for all pairs (only query pair predictions used)

The model must handle variable numbers of demonstration pairs (up to demo_num).
Demonstrations are presented as (init[i], end[i]) pairs; the query is the last pair
where end is unknown and must be predicted.
"""
import torch
import torch.nn as nn
import math


# ================================================================
# FIXED UTILITIES — do not modify
# ================================================================
def patchify(x, patch_num):
    """Convert image to patches: (B, C, H, W) -> (B, P*P, C*ph*pw)"""
    bs, c, hp, wp = x.shape
    h = hp // patch_num
    w = wp // patch_num
    p = patch_num
    patches = x.view(bs, c, p, h, p, w).permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape((bs, p * p, c * h * w))
    return patches


def depatchify(patches, patch_num, c, h, w):
    """Convert patches back to image: (B, P*P, C*ph*pw) -> (B, C, H, W)"""
    bs = patches.shape[0]
    p = patch_num
    patches = patches.view(bs, p, p, c, h, w).permute(0, 3, 1, 4, 2, 5)
    x = patches.reshape((bs, c, p * h, p * w))
    return x


def build_alternating_block_lowtri_mask(block_num, block_size1, block_size2):
    """Build lower-triangular block causal mask for in-context learning.

    Creates a mask that allows each demonstration pair to attend to all previous pairs
    but prevents attending to future pairs (causal structure). Within each pair,
    the input can attend to the output but not vice versa.

    Args:
        block_num: number of demonstration pairs
        block_size1: number of tokens in input part (patch_num^2)
        block_size2: number of tokens in output part (patch_num^2)
    Returns:
        mask of shape (block_num*(block_size1+block_size2), block_num*(block_size1+block_size2))
    """
    base_mask_inner = torch.ones((block_size1 + block_size2, block_size1 + block_size2))
    base_mask_inner_tail = torch.ones((block_size1 + block_size2, block_size1 + block_size2))
    base_mask_inner_tail[:block_size1, block_size1:] = 0

    expanded_mask = torch.zeros(block_num, block_num, block_size1 + block_size2, block_size1 + block_size2)
    for i in range(block_num):
        for j in range(i):
            expanded_mask[i, j] = base_mask_inner
        expanded_mask[i, i] = base_mask_inner_tail

    permuted_mask = expanded_mask.permute(0, 2, 1, 3)
    final_mask = permuted_mask.reshape(
        block_num * (block_size1 + block_size2),
        block_num * (block_size1 + block_size2),
    )
    return final_mask


# ================================================================
# EDITABLE — agent modifies this section
# ================================================================
class CustomModel(nn.Module):
    """In-context operator learning model for multi-physics PDE prediction.

    The model takes demonstration pairs (input_state, output_state) showing how a PDE
    evolves, plus a query input state, and predicts the query output state.

    Key design choices to consider:
    - How to tokenize 2D fields (patch size, resolution trade-offs)
    - How to encode spatial position information
    - How to encode which demonstration pair each token belongs to
    - How to enforce causal structure (query cannot see answer)
    - What transformer architecture to use (depth, width, attention type)

    Config attributes (cfg dict):
        cfg["transformer"]["dim_channel"]: int = 7  (number of input channels)
        cfg["transformer"]["dim_token"]: int = 1024  (token embedding dimension)
        cfg["transformer"]["nhead"]: int = 8
        cfg["transformer"]["dim_feedforward"]: int = 2048
        cfg["transformer"]["num_layers"]: int = 10
        cfg["transformer"]["dropout"]: float = 0.0
        cfg["demo_num"]: int = 10  (max number of demonstration pairs)
        cfg["patch_num_in"]: int = 8  (number of patches per spatial dim)
        cfg["patch_num_out"]: int = 8
        cfg["patch_resolution"]: int = 16  (pixels per patch per dim = 128/8)
        cfg["use_patch_pos_encoding"]: bool = True
        cfg["use_func_pos_encoding"]: bool = True
    """

    def __init__(self, cfg):
        super(CustomModel, self).__init__()
        self.cfg = cfg
        assert cfg["patch_num_in"] == cfg["patch_num_out"]

        p = cfg["patch_num_in"]
        d = cfg["transformer"]["dim_token"]
        c = cfg["transformer"]["dim_channel"]
        pr = cfg["patch_resolution"]

        # Linear projection: flatten each patch and project to token dim
        self.pre_proj = nn.Linear(c * pr ** 2, d)
        # Output projection: token dim back to patch pixel space
        self.post_proj = nn.Linear(d, c * pr ** 2)

        # TODO: Define your model architecture here.
        # Consider:
        # 1. Positional encodings for spatial patch positions
        # 2. Positional encodings for demonstration pair identity
        # 3. Transformer backbone with appropriate attention masking
        # 4. The causal structure: query output tokens should not attend to themselves

        # Placeholder: simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["transformer"]["nhead"],
            dim_feedforward=cfg["transformer"]["dim_feedforward"],
            dropout=cfg["transformer"]["dropout"],
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg["transformer"]["num_layers"],
        )

    def forward(self, x_tuple):
        """
        Args:
            x_tuple: tuple of (init, end, c_mask)
                init: (B, pairs, C, H, W) — input states
                end:  (B, pairs, C, H, W) — output states (target for query pair)
                c_mask: (B, C) — channel validity mask
        Returns:
            output: (B, pairs, C, H, W) — predicted output states
        """
        init, end, c_mask = x_tuple

        # Stack init and end into interleaved sequence: [init_0, end_0, init_1, end_1, ...]
        # Shape: (B, pairs, 2, C, H, W)
        x = torch.cat((init[:, :, None, :, :, :], end[:, :, None, :, :, :]), dim=2)

        p = self.cfg["patch_num_in"]
        d = self.cfg["transformer"]["dim_token"]
        bs, pairs, _, c, h, w = x.shape

        # Flatten batch dims and patchify
        feature = x.view(-1, *x.shape[-3:])  # (B*pairs*2, C, H, W)
        c_img, ph, pw = feature.shape[-3:]
        h_patch = ph // p
        w_patch = pw // p
        feature = patchify(feature, patch_num=p)  # (B*pairs*2, P*P, C*ph*pw)

        # Project patches to token dimension
        feature = self.pre_proj(feature)  # (B*pairs*2, P*P, d)
        feature = feature.view(bs, -1, p * p, d)  # (B, pairs*2, P*P, d)

        # TODO: Add positional encodings and reshape for transformer
        # TODO: Apply attention mask for causal in-context learning
        # TODO: Process through transformer

        # Flatten spatial and pair dims for transformer
        feature = feature.view(bs, -1, d)  # (B, pairs*2*P*P, d)
        feature = self.transformer(feature)  # (B, pairs*2*P*P, d)

        # Reshape back
        feature = feature.view(bs, pairs, 2, p * p, d)
        # Extract input-side predictions (position 0 of each pair)
        feature = feature[:, :, 0, :, :]  # (B, pairs, P*P, d)

        # Project back to pixel space
        feature = self.post_proj(feature)  # (B, pairs, P*P, C*ph*pw)

        # Depatchify
        feature = feature.view(bs * pairs, *feature.shape[-2:])
        feature = depatchify(feature, patch_num=p, c=c_img, h=h_patch, w=w_patch)
        feature = feature.view(bs, pairs, *feature.shape[-3:])  # (B, pairs, C, H, W)

        return feature
# ================================================================
# FIXED — end of editable section
# ================================================================


# ── FIXED: Parameter budget check ────────────────────────────────
_orig_custom_init = CustomModel.__init__

def _patched_custom_init(self, cfg):
    _orig_custom_init(self, cfg)
    _total = sum(p_.numel() for p_ in self.parameters())
    # The authoritative parameter budget is enforced by tasks/.../budget_check.py,
    # which computes 1.05x the largest instantiated baseline. Keep an informative
    # print here, but avoid hard-coding a stale baseline-specific upper bound.
    print(f"Total params: {_total:,} (task budget enforced by budget_check.py)")

CustomModel.__init__ = _patched_custom_init

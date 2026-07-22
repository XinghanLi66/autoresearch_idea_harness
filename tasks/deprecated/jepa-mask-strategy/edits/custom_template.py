"""
Tiny-ImageNet I-JEPA Self-Supervised Training Script (Self-Contained)

Trains a Vision Transformer (ViT) encoder using a Joint-Embedding Predictive
Architecture (I-JEPA, Assran et al. 2023, https://arxiv.org/abs/2301.08243)
with an EMA-updated target encoder, a small predictor transformer, and a
post-hoc frozen-encoder linear probe on Tiny-ImageNet-200 (64x64, 200 classes).

The 64x64 / patch=4 grid yields 16x16 = 256 patches per image, which is
slightly larger than the paper's 14x14 = 196 patches at 224x224 / patch=16.
This is the canonical regime for I-JEPA's multi-block recipe
(enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.20), npred=4): on the
old CIFAR-10 8x8 = 64 grid the multi-block context collapsed to ~5 patches
and the SSL signal degenerated.

The agent edits ONLY the CustomMaskSampler class (and the CONFIG_OVERRIDES
dict). Everything else -- ViT, EMA, predictor, linear probe, training loop
-- is fixed infrastructure to make the comparison apples-to-apples.

CustomMaskSampler interface
---------------------------
The collator instantiates ONE shared CustomMaskSampler (constructed with
``grid_size=(H_p, W_p)``) and calls ``sampler.sample(generator)`` ONCE per
image, where ``generator`` is a per-batch ``torch.Generator``. The method
must return:

    context_indices : LongTensor of shape [N_ctx]
        flat patch indices (in [0, H_p * W_p)) visible to the context encoder
    target_indices_list : list of LongTensor, length M (per-image targets)
        each entry is a LongTensor [N_tgt_m] of flat patch indices to predict

Different images in a batch may produce different N_ctx / N_tgt_m; the
collator truncates to the per-batch min so all tensors stack cleanly.

Reference: facebookresearch/ijepa src/masks/multiblock.py
"""

import sys; sys.path = [p for p in sys.path if not __import__('os').path.isfile(__import__('os').path.join(p, 'logging.py'))]
import os
import math
import time
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


# ── Custom Mask Sampler ────────────────────────────────────────────────────
# EDITABLE REGION START
class CustomMaskSampler:
    """Mask sampler for I-JEPA-style context/target patch masking on a 2D grid.

    The collator constructs one sampler per process via
    ``CustomMaskSampler(grid_size=(H_p, W_p))`` and calls ``sample(generator)``
    once per image. ``generator`` is a freshly-seeded torch.Generator unique
    to the (worker, batch, image) triple, so different images get different
    masks while remaining reproducible.

    Returns:
        context_indices: LongTensor[N_ctx], flat indices in [0, H_p*W_p)
        target_indices_list: list of LongTensor[N_tgt_m], one per target block

    Default implementation: trivial baseline that uses ALL patches as context
    and the FULL set of patches as a single target. This trains the predictor
    to be the identity on top of the (EMA-target) encoder -- a sanity-check
    lower bound.
    """

    def __init__(self, grid_size):
        self.H, self.W = grid_size
        self.N = self.H * self.W

    def sample(self, generator):
        all_idx = torch.arange(self.N, dtype=torch.long)
        return all_idx, [all_idx]


# CONFIG_OVERRIDES: per-method training-hyperparameter overrides.
# Allowed keys (whitelist enforced in main()):
#   pred_depth: int   -- predictor transformer depth (default 4)
#   pred_dim: int     -- predictor embedding dim (default 192)
CONFIG_OVERRIDES = {}
# EDITABLE REGION END


# ── ViT Backbone ────────────────────────────────────────────────────────────
# Minimal ViT (https://arxiv.org/abs/2010.11929) sized for Tiny-ImageNet
# (64x64, patch_size=4 -> 16x16=256 patches). Three configs: tiny, small,
# base. The 256-patch grid is paper-canonical for I-JEPA's multi-block
# recipe (paper uses 14x14=196 at 224 / patch=16).

VIT_CONFIGS = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0),
    "small": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0),
    "base": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0),
}


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    """Standard 2-D sin-cos positional embeddings (MAE-style)."""
    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=0)  # [2, H, W]
    grid = grid.reshape(2, -1)  # [2, H*W]
    assert embed_dim % 4 == 0
    half = embed_dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)
    out_y = grid[0][:, None] * omega[None, :]  # [H*W, half/2]
    out_x = grid[1][:, None] * omega[None, :]
    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)
    pos = torch.cat([pos_y, pos_x], dim=1)  # [H*W, embed_dim]
    return pos


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, D, H_p, W_p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT that embeds full image, then optionally selects context patches.

    forward(x, context_idx=None):
      x:           [B, 3, H, W]
      context_idx: LongTensor [B, N_ctx] of patch indices to keep (or None for all)
    Returns features [B, N_out, D] (N_out = N_ctx if given else N).
    """

    def __init__(self, img_size=64, patch_size=4, embed_dim=192, depth=12,
                 num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.grid_h = self.patch_embed.grid_h
        self.grid_w = self.patch_embed.grid_w
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        pos = get_2d_sincos_pos_embed(embed_dim, self.grid_h, self.grid_w)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, context_idx=None):
        x = self.patch_embed(x) + self.pos_embed  # [B, N, D]
        if context_idx is not None:
            # gather context patches: context_idx is [B, N_ctx]
            B, N_ctx = context_idx.shape
            D = x.shape[-1]
            idx_exp = context_idx.unsqueeze(-1).expand(B, N_ctx, D)
            x = torch.gather(x, dim=1, index=idx_exp)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def encode_full(self, x):
        """Encode full image without masking (for target encoder + linear probe)."""
        return self.forward(x, context_idx=None)


class Predictor(nn.Module):
    """Small transformer that maps context tokens + target queries -> target features.

    Following ijepa/src/models/vision_transformer_predictor.py: project
    context tokens to a smaller predictor_dim, append a learned mask token at
    each target position (with target positional embedding), run a stack of
    transformer blocks, and project back to encoder_dim at the target slots
    only.
    """

    def __init__(self, encoder_dim, predictor_dim, depth, num_heads,
                 num_patches, grid_h, grid_w):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.in_proj = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        pos = get_2d_sincos_pos_embed(predictor_dim, grid_h, grid_w)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList(
            [Block(predictor_dim, num_heads, mlp_ratio=4.0) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, ctx, ctx_idx, tgt_idx):
        """
        ctx:     [B, N_ctx, encoder_dim]   context-encoder output
        ctx_idx: [B, N_ctx]                flat patch indices of ctx tokens
        tgt_idx: [B, N_tgt]                flat patch indices of target tokens
        Returns: [B, N_tgt, encoder_dim]   predicted target features
        """
        B, N_ctx, _ = ctx.shape
        _, N_tgt = tgt_idx.shape
        ctx = self.in_proj(ctx)
        # Add positional embedding to context tokens at their original locs
        D = ctx.shape[-1]
        pos_ctx = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=ctx_idx.unsqueeze(-1).expand(B, N_ctx, D),
        )
        ctx = ctx + pos_ctx
        # Build target query tokens: mask_token + positional embedding
        pos_tgt = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=tgt_idx.unsqueeze(-1).expand(B, N_tgt, D),
        )
        tgt_q = self.mask_token.expand(B, N_tgt, D) + pos_tgt
        x = torch.cat([ctx, tgt_q], dim=1)  # [B, N_ctx+N_tgt, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, N_ctx:, :]  # take only target slots
        x = self.out_proj(x)
        return x


# ── Linear Probe ────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


# ── Mask Collator (fixed) ──────────────────────────────────────────────────
# Wraps CustomMaskSampler. Calls sampler.sample(g) per image; truncates to
# per-batch minimum length so tensors stack. Mirrors the truncation logic in
# ijepa/src/masks/multiblock.py:__call__ but defers block-shape decisions
# to the user-supplied sampler.

class MaskCollator:
    def __init__(self, sampler, base_collate=None):
        self.sampler = sampler
        self.base_collate = base_collate or torch.utils.data.default_collate
        self._counter = 0

    def __call__(self, batch):
        collated = self.base_collate(batch)
        B = len(batch)
        ctxs, tgts_per_image = [], []
        n_targets = None
        for i in range(B):
            self._counter += 1
            g = torch.Generator()
            mask_seed = (torch.initial_seed() + self._counter) % (2**63 - 1)
            g.manual_seed(mask_seed)
            ctx, tgts = self.sampler.sample(g)
            if not isinstance(ctx, torch.Tensor):
                ctx = torch.as_tensor(ctx, dtype=torch.long)
            ctx = ctx.to(torch.long).flatten()
            tgts = [t if isinstance(t, torch.Tensor) else torch.as_tensor(t) for t in tgts]
            tgts = [t.to(torch.long).flatten() for t in tgts]
            if n_targets is None:
                n_targets = len(tgts)
            assert len(tgts) == n_targets, "sampler must return same #targets per image"
            ctxs.append(ctx)
            tgts_per_image.append(tgts)
        # truncate to per-batch min
        min_ctx = min(len(c) for c in ctxs)
        min_ctx = max(min_ctx, 1)
        ctxs = torch.stack([c[:min_ctx] for c in ctxs], dim=0)  # [B, N_ctx]
        # per-target-block min across batch
        target_stacks = []
        for j in range(n_targets):
            min_tgt = min(len(tgts_per_image[i][j]) for i in range(B))
            min_tgt = max(min_tgt, 1)
            tj = torch.stack([tgts_per_image[i][j][:min_tgt] for i in range(B)], dim=0)
            target_stacks.append(tj)  # [B, N_tgt_j]
        return collated, ctxs, target_stacks


# Per-process global seed offset; set in main() before DataLoader creation.
_GLOBAL_SEED = 0


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = (_GLOBAL_SEED + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ── Data ────────────────────────────────────────────────────────────────────

# ImageNet statistics (Tiny-ImageNet is an ImageNet subset; using ImageNet
# mean/std matches the I-JEPA paper which trains on full ImageNet).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms():
    # Paper uses RandomResizedCrop + RandomHorizontalFlip + ColorJitter
    # (Assran et al. 2023, Sec 4.1; configs/in1k_vith14_ep300.yaml uses the
    # default image_jepa transform from src/transforms.py). RandomResizedCrop
    # scale=(0.3, 1.0) follows the paper's image_jepa transform defaults.
    return transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.3, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── EMA helper ──────────────────────────────────────────────────────────────

@torch.no_grad()
def ema_update(target_model, online_model, m):
    for p_t, p_o in zip(target_model.parameters(), online_model.parameters()):
        p_t.data.mul_(m).add_((1.0 - m) * p_o.detach().data)


# ── Training ────────────────────────────────────────────────────────────────

def evaluate_linear_probe(target_encoder, linear_probe, val_loader, device, use_amp=True):
    target_encoder.eval()
    linear_probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                feats = target_encoder.encode_full(x)  # [B, N, D]
            feats = feats.float().mean(dim=1)
            logits = linear_probe(feats)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def _extract_features(target_encoder, loader, device, use_amp=True):
    """Run target_encoder once over a dataloader, return mean-pooled features + labels.

    Used for the post-hoc frozen-encoder linear probe fit. Eliminates the
    redundant forward passes per probe-fit epoch (we encode once, then train
    the linear head on the cached features).
    """
    target_encoder.eval()
    feats_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            f = target_encoder.encode_full(x)  # [B, N, D]
        f = f.float().mean(dim=1).cpu()
        feats_list.append(f)
        labels_list.append(y.cpu())
    return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


def fit_linear_probe_posthoc(target_encoder, linear_probe, train_loader_eval,
                             val_loader, device, num_classes, use_amp=True,
                             probe_epochs=50, probe_lr=0.1, probe_batch=512):
    """Post-hoc frozen-encoder linear probe.

    Freezes target_encoder, pre-computes mean-pooled features on train+val
    (with eval-time transforms — no flip / jitter / RandomResizedCrop), then
    trains a fresh nn.Linear from scratch with SGD lr=0.1, momentum=0.9,
    weight_decay=0, cosine LR schedule for ``probe_epochs`` epochs.
    Updates ``linear_probe.classifier`` weights in-place and returns the
    final val accuracy.
    """
    feat_dim = linear_probe.classifier.in_features
    # Re-init the linear head from scratch so we don't reuse the stale
    # online-probe weights.
    linear_probe.classifier = nn.Linear(feat_dim, num_classes).to(device)

    train_feats, train_labels = _extract_features(target_encoder, train_loader_eval, device, use_amp)
    val_feats, val_labels = _extract_features(target_encoder, val_loader, device, use_amp)
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)
    print(f"Post-hoc probe: train_feats={tuple(train_feats.shape)} "
          f"val_feats={tuple(val_feats.shape)}", flush=True)

    optimizer = optim.SGD(linear_probe.parameters(), lr=probe_lr,
                          momentum=0.9, weight_decay=0.0)

    n = train_feats.size(0)
    iters_per_epoch = max(1, (n + probe_batch - 1) // probe_batch)
    total_iters = probe_epochs * iters_per_epoch

    def cos_lr(it):
        return 0.5 * probe_lr * (1.0 + math.cos(math.pi * it / max(total_iters, 1)))

    step = 0
    linear_probe.train()
    for ep in range(probe_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, probe_batch):
            idx = perm[i:i + probe_batch]
            xb = train_feats[idx]
            yb = train_labels[idx]
            cur_lr = cos_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
            logits = linear_probe(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
        if ep % 10 == 0 or ep == probe_epochs - 1:
            linear_probe.eval()
            with torch.no_grad():
                preds = linear_probe(val_feats).argmax(dim=1)
                cur_val = 100.0 * (preds == val_labels).float().mean().item()
            linear_probe.train()
            print(f"Probe-fit: epoch={ep} loss={loss.item():.4f} "
                  f"val_acc={cur_val:.2f} lr={cur_lr:.4f}", flush=True)

    linear_probe.eval()
    with torch.no_grad():
        preds = linear_probe(val_feats).argmax(dim=1)
        final_val = 100.0 * (preds == val_labels).float().mean().item()
    return final_val


def main():
    seed = int(os.environ.get("SEED", 42))
    arch = os.environ.get("ARCH", "tiny")
    data_dir = os.environ.get("EBJEPA_DSETS", "/data/eb_jepa")

    # Hyperparameters (paper: Assran et al. 2023 Sec 4.1 / configs/in1k_vith14_ep300.yaml)
    img_size = 64
    patch_size = 4  # 16x16 = 256 patches per Tiny-ImageNet image
    grid_h = grid_w = img_size // patch_size

    epochs = 100
    batch_size = 256
    lr = 1e-3
    start_lr = 2e-4
    final_lr = 1e-6
    weight_decay = 0.04
    final_weight_decay = 0.4
    warmup_epochs = 10
    ema_start = 0.996
    ema_end = 1.0
    pred_depth = 4
    pred_dim = 192

    # CONFIG_OVERRIDES (whitelist)
    for k, v in CONFIG_OVERRIDES.items():
        if k == "pred_depth": pred_depth = int(v)
        elif k == "pred_dim": pred_dim = int(v)

    use_amp = True
    num_workers = 4

    global _GLOBAL_SEED
    _GLOBAL_SEED = seed * 100003
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Sampler + collator
    sampler = CustomMaskSampler(grid_size=(grid_h, grid_w))
    collator = MaskCollator(sampler)

    # Data: Tiny-ImageNet-200, 64x64 RGB, 200 classes, 100k train + 10k val.
    # Pre-downloaded and pre-organized by the container build into
    # /data/eb_jepa/tiny-imagenet-200/{train,val}/<wnid>/images/*.JPEG so that
    # torchvision.datasets.ImageFolder discovers all 200 classes correctly.
    tin_root = os.path.join(data_dir, "tiny-imagenet-200")
    train_set = ImageFolder(root=os.path.join(tin_root, "train"),
                            transform=get_train_transforms())
    val_set = ImageFolder(root=os.path.join(tin_root, "val"),
                          transform=get_eval_transforms())
    num_classes = len(train_set.classes)
    assert num_classes == 200, f"expected 200 Tiny-ImageNet classes, got {num_classes}"
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collator,
                              worker_init_fn=seed_worker,
                              generator=make_generator(seed))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=make_generator(seed + 1))
    print(f"Tiny-ImageNet-200: train={len(train_set)} val={len(val_set)} "
          f"classes={num_classes} batch={batch_size} "
          f"grid={grid_h}x{grid_w}={grid_h*grid_w} patches",
          flush=True)

    # Models
    cfg = VIT_CONFIGS[arch]
    encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, **cfg).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor = Predictor(
        encoder_dim=cfg["embed_dim"], predictor_dim=pred_dim,
        depth=pred_depth, num_heads=max(pred_dim // 64, 1),
        num_patches=encoder.num_patches, grid_h=grid_h, grid_w=grid_w,
    ).to(device)
    linear_probe = LinearProbe(cfg["embed_dim"], num_classes).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"Model: ViT-{arch} encoder={n_enc:,} predictor={n_pred:,} "
          f"pred_depth={pred_depth} pred_dim={pred_dim}", flush=True)

    # Optimizer (AdamW for encoder + predictor only).
    # Linear probe is fit POST-HOC against the frozen target encoder after
    # JEPA training completes (see fit_linear_probe_posthoc); training-time
    # online probing produces stale features because the EMA target drifts.
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )
    scaler = GradScaler(device.type, enabled=use_amp)

    # Schedules
    iters_per_epoch = len(train_loader)
    total_iters = max(epochs * iters_per_epoch, 1)
    warmup_iters = warmup_epochs * iters_per_epoch

    def lr_at(it):
        if it < warmup_iters:
            return start_lr + (lr - start_lr) * (it / max(warmup_iters, 1))
        prog = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return final_lr + 0.5 * (lr - final_lr) * (1 + math.cos(math.pi * prog))

    def wd_at(it):
        prog = it / total_iters
        return weight_decay + (final_weight_decay - weight_decay) * prog

    def ema_at(it):
        return ema_start + (ema_end - ema_start) * (it / total_iters)

    print(f"Training: epochs={epochs} iters/epoch={iters_per_epoch} "
          f"total_iters={total_iters}", flush=True)

    global_step = 0
    start_time = time.time()
    log_every_epochs = 5
    cur_lr = lr  # in case epoch loop is empty
    for epoch in range(epochs):
        encoder.train(); predictor.train()
        loss_sum = 0.0
        for batch_data, ctx_idx, target_idx_list in train_loader:
            (imgs, labels) = batch_data
            imgs = imgs.to(device, non_blocking=True)
            ctx_idx = ctx_idx.to(device, non_blocking=True)
            target_idx_list = [t.to(device, non_blocking=True) for t in target_idx_list]

            # Update LR / WD
            cur_lr = lr_at(global_step)
            cur_wd = wd_at(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
                pg["weight_decay"] = cur_wd

            with autocast(device.type, enabled=use_amp, dtype=torch.bfloat16):
                # Target features: full image through target encoder, gather
                # targets, then LayerNorm (canonical I-JEPA, see
                # facebookresearch/ijepa src/train.py: F.layer_norm(h, (h.size(-1),))).
                with torch.no_grad():
                    h_full = target_encoder.encode_full(imgs)  # [B, N, D]
                    h_full = F.layer_norm(h_full, (h_full.size(-1),))
                    targets = []
                    for t_idx in target_idx_list:
                        D = h_full.size(-1)
                        idx_exp = t_idx.unsqueeze(-1).expand(-1, -1, D)
                        targets.append(torch.gather(h_full, dim=1, index=idx_exp))

                # Context features: masked image through online encoder
                z_ctx = encoder(imgs, context_idx=ctx_idx)  # [B, N_ctx, D]

                # Predict each target block separately and compare the raw
                # predictor output against the LayerNorm'd target features,
                # matching canonical I-JEPA training.
                pred_loss = 0.0
                for t_idx, t_feat in zip(target_idx_list, targets):
                    z_pred = predictor(z_ctx, ctx_idx, t_idx)
                    pred_loss = pred_loss + F.smooth_l1_loss(z_pred, t_feat)
                pred_loss = pred_loss / max(len(target_idx_list), 1)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(pred_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema_update(target_encoder, encoder, ema_at(global_step))

            loss_sum += pred_loss.item()
            global_step += 1

        avg_loss = loss_sum / max(iters_per_epoch, 1)

        if epoch % log_every_epochs == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            # Note: val_acc here is only meaningful AFTER the post-hoc probe
            # has been fit. During JEPA training we report jepa_loss as the
            # primary signal (TRAIN_METRICS is purely a progress log).
            print(f"TRAIN_METRICS: epoch={epoch} | jepa_loss={avg_loss:.4f} "
                  f"| lr={cur_lr:.2e} | time={elapsed:.1f}s", flush=True)

    # ── Post-hoc frozen-encoder linear probe ────────────────────────────
    # JEPA training is done. Freeze target_encoder, build an EVAL-transform
    # train loader (no flip / jitter / RandomResizedCrop), fit a fresh
    # nn.Linear from scratch on cached features (~50 epochs, SGD lr=0.1
    # cosine), and report THAT accuracy.
    for p in target_encoder.parameters():
        p.requires_grad = False
    target_encoder.eval()
    train_set_eval = ImageFolder(root=os.path.join(tin_root, "train"),
                                 transform=get_eval_transforms())
    train_loader_eval = DataLoader(train_set_eval, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers,
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=make_generator(seed + 2))
    val_acc = fit_linear_probe_posthoc(
        target_encoder, linear_probe, train_loader_eval, val_loader,
        device, num_classes, use_amp=use_amp,
    )
    print(f"TEST_METRICS: val_acc={val_acc:.2f}", flush=True)
    print(f"Training completed. Final val_acc={val_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()

"""I-JEPA multi-block mask sampler -- paper-faithful baseline.

Reference: Assran et al. 2023, "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture" (https://arxiv.org/abs/2301.08243).

Implementation ported verbatim from
  facebookresearch/ijepa/src/masks/multiblock.py:MaskCollator
  (commit main, methods _sample_block_size + _sample_block_mask)

Hyperparameters from
  facebookresearch/ijepa/configs/in1k_vith14_ep300.yaml:
    enc_mask_scale  = (0.85, 1.0)   # one large context block
    pred_mask_scale = (0.15, 0.2)   # four small target blocks
    aspect_ratio    = (0.75, 1.5)
    nenc            = 1
    npred           = 4
    allow_overlap   = false
    min_keep        = 10            # paper-canonical; safe at 16x16=256

We use Tiny-ImageNet (64x64, patch_size=4 -> 16x16 = 256 patches) instead
of CIFAR-10 (8x8 = 64 patches) so multi-block runs in its native regime:
target blocks at scale 0.15-0.2 yield ~38-51 patches and the context block
at scale 0.85-1.0 has ample room (218-256 patches before target carving).
This is paper-faithful and matches the 14x14 = 196 patch grid I-JEPA used
for ImageNet-1k.
"""

_FILE = "eb_jepa/custom_mask.py"

_MULTIBLOCK = """\
class CustomMaskSampler:
    \"\"\"I-JEPA multi-block sampler (Assran et al. 2023).

    One large context block (scale 0.85-1.0, aspect 1.0) with 4 small
    target blocks (scale 0.15-0.2, aspect 0.75-1.5) carved out. The
    context's acceptable region is the *complement* of all target blocks
    (allow_overlap=False), so context never sees what it must predict.
    Indices returned for context are the patches still inside the
    encoder block AFTER target patches are excluded.
    \"\"\"

    def __init__(self, grid_size,
                 enc_mask_scale=(0.85, 1.0),
                 pred_mask_scale=(0.15, 0.2),
                 aspect_ratio=(0.75, 1.5),
                 nenc=1, npred=4, min_keep=10):
        self.H, self.W = grid_size
        self.N = self.H * self.W
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.H * self.W * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        ar = min_ar + _rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * ar)))
        w = int(round(math.sqrt(max_keep / ar)))
        while h >= self.H: h -= 1
        while w >= self.W: w -= 1
        h = max(h, 1); w = max(w, 1)
        return h, w

    def _sample_block_mask(self, generator, b_size, acceptable_regions=None):
        h, w = b_size
        tries = 0
        og_timeout = 20
        timeout = og_timeout
        while True:
            top = int(torch.randint(0, max(self.H - h, 1), (1,), generator=generator).item())
            left = int(torch.randint(0, max(self.W - w, 1), (1,), generator=generator).item())
            mask = torch.zeros((self.H, self.W), dtype=torch.int32)
            mask[top:top + h, left:left + w] = 1
            mask_complement = torch.ones((self.H, self.W), dtype=torch.int32)
            mask_complement[top:top + h, left:left + w] = 0
            if acceptable_regions is not None:
                N = max(int(len(acceptable_regions) - tries), 0)
                m = mask.clone()
                for k in range(N):
                    m = m * acceptable_regions[k]
                idx = torch.nonzero(m.flatten()).squeeze(-1)
            else:
                idx = torch.nonzero(mask.flatten()).squeeze(-1)
            if idx.numel() > self.min_keep:
                return idx, mask_complement
            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = og_timeout
            if tries > 5:
                # Fallback: accept whatever we have if >=1
                if idx.numel() >= 1:
                    return idx, mask_complement
                # Resample with same generator advancing

    def sample(self, generator):
        # 1) decide block sizes (shared seed across image -- we use the
        #    per-image generator, which already differs across images
        #    via _GLOBAL_SEED + _counter in the collator).
        p_size = self._sample_block_size(generator, self.pred_mask_scale,
                                          self.aspect_ratio)
        e_size = self._sample_block_size(generator, self.enc_mask_scale,
                                          (1.0, 1.0))
        # 2) sample target blocks; collect their complements as acceptable regions
        targets = []
        complements = []
        for _ in range(self.npred):
            t_idx, t_comp = self._sample_block_mask(generator, p_size)
            targets.append(t_idx)
            complements.append(t_comp)
        # 3) sample one context block constrained to NOT overlap targets
        ctx_idx, _ = self._sample_block_mask(generator, e_size,
                                              acceptable_regions=complements)
        return ctx_idx, targets


# Paper uses pred_depth=12, pred_emb_dim=384 for ViT-Huge on ImageNet.
# We keep CIFAR-scale defaults but slightly bump predictor capacity since
# multi-block is the paper's primary recipe and benefits most from it.
CONFIG_OVERRIDES = {"pred_depth": 6, "pred_dim": 192}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 59,
        "end_line": 91,
        "content": _MULTIBLOCK,
    },
]

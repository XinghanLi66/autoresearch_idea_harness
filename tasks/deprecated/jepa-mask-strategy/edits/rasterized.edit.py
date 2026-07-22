"""I-JEPA Table 6 'rasterized' baseline (paper-faithful, fixed-context variant).

Reference: Assran et al. 2023, Sec 4.4 / Table 6
(https://arxiv.org/abs/2301.08243). Paper text:

    "in rasterized masking, the image is split into four large quadrants,
     and the goal is to use one quadrant as a context to predict the
     other three quadrants"

Table 6 row: ``Targets = Quadrant, Freq=3, Context = Complement,
Avg.Ratio=0.25``.

Per the task spec we adapt 'rasterized' to a CONTIGUOUS HORIZONTAL STRIP
target (4 rows of patches at a random vertical offset) on the 16x16
Tiny-ImageNet grid. This preserves the *spatial-rasterized* nature of
the paper's quadrant baseline (a contiguous, axis-aligned stripe of
patches) while keeping the target count comparable to ``multiblock``
(4 rows x 16 cols = 64 patches, vs multiblock 4 x [0.15-0.20] x 256
= ~150-200 before per-batch truncation).

For an apples-to-apples ablation we **fix the encoder context block to
scale (0.85, 1.0) across all four baselines** so that only the target
sampling varies. Encoder block sampling is the same routine used by
``multiblock`` (verbatim from
``facebookresearch/ijepa/src/masks/multiblock.py``), with allow_overlap
= False so the context never sees what it must predict.

Hyperparameters
---------------
- enc_mask_scale = (0.85, 1.0)
- strip_rows     = 4 rows of patches  (~0.25 of 16-row grid -- matches
                                        paper's per-target-quadrant ratio)
"""

_FILE = "eb_jepa/custom_mask.py"

_RASTERIZED = """\
class CustomMaskSampler:
    \"\"\"I-JEPA Table 6 'rasterized' baseline (horizontal strip variant).

    Targets = a contiguous horizontal strip (``strip_rows`` rows of
    patches) at a random vertical offset; context = encoder Block(0.85-1.0)
    constrained to NON-target patches (allow_overlap=False).

    Reference: Assran et al. 2023, Sec 4.4 / Table 6,
    https://arxiv.org/abs/2301.08243.
    \"\"\"

    def __init__(self, grid_size,
                 enc_mask_scale=(0.85, 1.0),
                 strip_rows=4,
                 min_keep=10):
        self.H, self.W = grid_size
        self.N = self.H * self.W
        self.enc_mask_scale = enc_mask_scale
        self.strip_rows = max(1, min(strip_rows, self.H - 1))
        self.min_keep = min_keep

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        # Verbatim from facebookresearch/ijepa/src/masks/multiblock.py
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
        # Verbatim from facebookresearch/ijepa/src/masks/multiblock.py
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
                if idx.numel() >= 1:
                    return idx, mask_complement

    def sample(self, generator):
        # 1) pick a random vertical offset for the horizontal strip
        max_top = max(self.H - self.strip_rows, 1)
        top = int(torch.randint(0, max_top, (1,), generator=generator).item())
        target_mask_2d = torch.zeros((self.H, self.W), dtype=torch.int32)
        target_mask_2d[top:top + self.strip_rows, :] = 1
        target_idx = torch.nonzero(target_mask_2d.flatten()).squeeze(-1)
        target_complement = 1 - target_mask_2d

        # 2) sample one encoder context block constrained to NOT overlap strip
        e_size = self._sample_block_size(generator, self.enc_mask_scale,
                                          (1.0, 1.0))
        ctx_idx, _ = self._sample_block_mask(generator, e_size,
                                              acceptable_regions=[target_complement])
        return ctx_idx, [target_idx]


CONFIG_OVERRIDES = {"pred_depth": 4, "pred_dim": 192}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 59,
        "end_line": 91,
        "content": _RASTERIZED,
    },
]

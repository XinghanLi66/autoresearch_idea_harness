"""I-JEPA Table 6 'random' baseline (paper-faithful, fixed-context variant).

Reference: Assran et al. 2023, Sec 4.4 / Table 6
(https://arxiv.org/abs/2301.08243). Paper text describing the row:

    "In random masking, the target is a set of random patches and the
     context is the image complement."

Table 6 row: ``Targets = Random(0.6), Freq=1, Context = Complement,
Avg.Ratio=0.4``. The official ijepa repo
(``facebookresearch/ijepa/src/masks/``) does NOT ship a separate
``random.py`` -- only ``multiblock.py`` -- so we implement the random
sampler here from the paper description.

For an apples-to-apples ablation on Tiny-ImageNet we **fix the encoder
context block to scale (0.85, 1.0) across all four baselines**
(multiblock / block / rasterized / random) so that only the *target*
sampling varies. Encoder block sampling is the same routine used by
``multiblock`` (verbatim from
``facebookresearch/ijepa/src/masks/multiblock.py`` -- _sample_block_size +
_sample_block_mask). The random target patches are drawn UNIFORMLY at
random (independent of spatial structure) as a single 'target block'
with the patches that fall WITHIN the encoder context block excluded
afterwards (allow_overlap=False), so the context never sees what it
must predict.

Hyperparameters
---------------
- enc_mask_scale       = (0.85, 1.0)
- target_count_per_img = ~75 patches  (matches multiblock 4 x [0.15-0.20]
                                        x 256 = ~38-51 each => ~150-200
                                        total before truncation; per-batch
                                        truncation will then bring this
                                        to a similar effective count)
- target_ratio         = 0.30         # 0.30 * 256 ~= 77 patches; close to
                                       # paper's Random(0.6)*0.4 effective
                                       # 'target inside context complement'
                                       # ratio scaled to our setup
"""

_FILE = "eb_jepa/custom_mask.py"

_RANDOM = """\
class CustomMaskSampler:
    \"\"\"I-JEPA Table 6 'random' baseline with fixed encoder context.

    Targets = uniformly random subset of patches (Random(target_ratio));
    context = encoder Block(0.85-1.0) restricted to NON-target patches
    (allow_overlap=False), so context never sees what it must predict.

    Reference: Assran et al. 2023 Sec 4.4 / Table 6,
    https://arxiv.org/abs/2301.08243.
    \"\"\"

    def __init__(self, grid_size,
                 enc_mask_scale=(0.85, 1.0),
                 target_ratio=0.30,
                 min_keep=10):
        self.H, self.W = grid_size
        self.N = self.H * self.W
        self.enc_mask_scale = enc_mask_scale
        self.target_ratio = target_ratio
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
        # 1) sample uniformly random target patch indices (paper Sec 4.4)
        n_target = max(int(round(self.N * self.target_ratio)), self.min_keep)
        noise = torch.rand(self.N, generator=generator)
        ids_shuffle = torch.argsort(noise)
        target_idx = ids_shuffle[:n_target].clone()

        # 2) build target-complement mask for the encoder constraint
        target_complement = torch.ones((self.H, self.W), dtype=torch.int32)
        target_complement.view(-1)[target_idx] = 0

        # 3) sample one encoder context block constrained to NOT overlap target
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
        "content": _RANDOM,
    },
]

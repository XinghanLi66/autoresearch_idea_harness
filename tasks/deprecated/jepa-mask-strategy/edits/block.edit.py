"""I-JEPA Table 6 'block' baseline (paper-faithful, fixed-context variant).

Reference: Assran et al. 2023, Sec 4.4 / Table 6, "Ablating masking strategy"
(https://arxiv.org/abs/2301.08243). The paper's Table 6 row for ``block`` is
``Targets = Block(0.6), Freq=1, Context = Complement, Avg.Ratio=0.4``,
i.e. one big target block of scale 0.6 with context = its complement.

For an apples-to-apples ablation on Tiny-ImageNet we **fix the encoder
context block to scale (0.85, 1.0) across all four baselines**
(multiblock / block / rasterized / random) so that only the *target*
sampling varies. This removes the confound where varying both the target
shape AND the context coverage changes the difficulty of the prediction
task simultaneously. Encoder block sampling is the same routine used by
``multiblock`` (verbatim from
``facebookresearch/ijepa/src/masks/multiblock.py`` -- _sample_block_size +
_sample_block_mask, with allow_overlap=False).

Hyperparameters
---------------
- enc_mask_scale  = (0.85, 1.0)   # paper recipe for multi-block context
- pred_mask_scale = (0.5, 0.75)   # one big target block (Table 6 used 0.6;
                                  # we widen slightly to 0.5-0.75 per task spec)
- aspect_ratio    = (0.75, 1.5)   # paper aspect range for blocks
- nenc            = 1
- npred           = 1             # SINGLE target block (key contrast vs multiblock)
- allow_overlap   = False         # context never sees what it must predict
- min_keep        = 10            # paper-canonical
"""

_FILE = "eb_jepa/custom_mask.py"

_BLOCK = """\
class CustomMaskSampler:
    \"\"\"I-JEPA Table 6 'block' baseline with fixed encoder context.

    One big target block (scale 0.5-0.75, aspect 0.75-1.5) and a single
    encoder context block (scale 0.85-1.0) constrained to the COMPLEMENT
    of the target (allow_overlap=False), so context never sees what it
    must predict. Block-sampling helpers ported verbatim from
    facebookresearch/ijepa/src/masks/multiblock.py.
    \"\"\"

    def __init__(self, grid_size,
                 enc_mask_scale=(0.85, 1.0),
                 pred_mask_scale=(0.5, 0.75),
                 aspect_ratio=(0.75, 1.5),
                 npred=1, min_keep=10):
        self.H, self.W = grid_size
        self.N = self.H * self.W
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
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
                if idx.numel() >= 1:
                    return idx, mask_complement

    def sample(self, generator):
        # 1) decide block sizes
        p_size = self._sample_block_size(generator, self.pred_mask_scale,
                                          self.aspect_ratio)
        e_size = self._sample_block_size(generator, self.enc_mask_scale,
                                          (1.0, 1.0))
        # 2) sample one target block; collect its complement as acceptable region
        targets = []
        complements = []
        for _ in range(self.npred):
            t_idx, t_comp = self._sample_block_mask(generator, p_size)
            targets.append(t_idx)
            complements.append(t_comp)
        # 3) sample one context block constrained to NOT overlap target
        ctx_idx, _ = self._sample_block_mask(generator, e_size,
                                              acceptable_regions=complements)
        return ctx_idx, targets


# Paper recipe for ablation table uses pred_depth=12, pred_emb_dim=384 on
# ViT-B/16. We keep the modest CIFAR-scale predictor since this baseline
# is meant to be compared head-to-head with multiblock under identical
# training infrastructure.
CONFIG_OVERRIDES = {"pred_depth": 4, "pred_dim": 192}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 59,
        "end_line": 91,
        "content": _BLOCK,
    },
]

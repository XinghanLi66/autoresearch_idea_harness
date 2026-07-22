# Diffusion Model: Classifier-Free Guidance Optimization

## Objective
Improve text-to-image generation quality by designing a better classifier-free guidance (CFG) method. Your improvement should generalize across different Stable Diffusion model variants.

## Background
Classifier-free guidance (CFG) is a fundamental technique in diffusion models for text-guided generation. The standard CFG formula is:

```
noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
```

where `noise_uc` is the unconditional noise prediction and `noise_c` is the conditional (text-guided) noise prediction.

However, standard CFG has limitations:
- Requires high guidance scales (typically 7.5-12.5)
- Can cause mode collapse and saturation
- Results in curved, unnatural sampling trajectories
- Poor invertibility

## Task
Your goal is to improve the CFG mechanism to achieve better text-image alignment (measured by CLIP score) while maintaining or improving sample quality. You must implement your improvement in **two files**:

1. **`latent_diffusion.py`** — `BaseDDIMCFGpp` class for SD v1.5
2. **`latent_sdxl.py`** — `BaseDDIMCFGpp` class for SDXL

The evaluation will test your method on both models (SD v1.5, SDXL).

## Editable Regions

### SD v1.5 (`latent_diffusion.py`, lines 621-679)
- Class `BaseDDIMCFGpp(StableDiffusion)` with `sample()` method
- Key API: `self.get_text_embed()`, `self.initialize_latent()`, `self.predict_noise()`, `self.alpha(t)`

### SDXL (`latent_sdxl.py`, lines 713-755)
- Class `BaseDDIMCFGpp(SDXL)` with `reverse_process()` method
- Key API: `self.initialize_latent(size=...)`, `self.predict_noise()`, `self.scheduler.alphas_cumprod[t]`

## Evaluation
- **Metric**: CLIP score (cosine similarity between generated image and text prompt), averaged across 2 models
- **Models**: SD v1.5, SDXL
- **Test prompts**: 100 diverse COCO-style prompts
- **Seeds**: Multi-seed evaluation

## Baselines
- **cfg**: Standard classifier-free guidance (uses noise_pred for renoising)
- **cfgpp**: CFG++ method (uses unconditional noise for renoising, keeping trajectory on manifold)
- **zeroinit**: CFG++ with zero-initialization (skips first K=2 steps)

Your implementation should aim to achieve higher average CLIP scores than all baselines.

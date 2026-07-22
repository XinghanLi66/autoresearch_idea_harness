# Diffusion Model: Sampler Efficiency Optimization

## Objective
Design an efficient sampling algorithm for text-to-image diffusion models that achieves high generation quality with minimal sampling steps (NFE).

## Background
Diffusion models generate images by iteratively denoising from random noise. Different sampling methods have different trade-offs:

- **DDIM**: First-order ODE solver, deterministic, fast but may need more steps for quality
- **Euler**: Simple first-order method, baseline performance
- **DPM++ 2M**: Second-order multistep method, more efficient
- **DPM++ 2S**: Second-order singlestep method, higher quality per step

The core sampling loop follows this pattern:
```python
for step, t in enumerate(timesteps):
    # 1. Predict noise
    noise_pred = model(zt, t, text_embedding)

    # 2. Estimate clean image (Tweedie's formula)
    z0t = (zt - sigma_t * noise_pred) / alpha_t

    # 3. Update to next step (THIS IS THE KEY DIFFERENCE)
    zt_next = update_rule(zt, z0t, noise_pred, t, t_next)
```

Different samplers use different `update_rule` strategies.

## Task
Your goal is to design an improved sampling update rule that achieves better image-text alignment (CLIP score) with a fixed budget of NFE=20 steps. You must implement your improvement in **two files**:

1. **`latent_diffusion.py`** — `BaseDDIMCFGpp` class for SD v1.5
2. **`latent_sdxl.py`** — `BaseDDIMCFGpp` class for SDXL

## Editable Regions

### SD v1.5 (`latent_diffusion.py`, lines 621-679)
- Class `BaseDDIMCFGpp(StableDiffusion)` with `sample()` method
- Key API: `self.get_text_embed()`, `self.initialize_latent()`, `self.predict_noise()`, `self.alpha(t)`

### SDXL (`latent_sdxl.py`, lines 713-755)
- Class `BaseDDIMCFGpp(SDXL)` with `reverse_process()` method
- Key API: `self.initialize_latent(size=...)`, `self.predict_noise()`, `self.scheduler.alphas_cumprod[t]`

## Evaluation
- **Metric**: CLIP score (cosine similarity between generated image and text prompt)
- **Fixed budget**: NFE=20 steps
- **Test prompts**: 100 diverse COCO-style prompts
- **Seeds**: Multi-seed evaluation

## Baselines
- **ddim**: Standard DDIM sampler (first-order)
- **dpm2m**: DPM++ 2M sampler (second-order multistep)
- **dpm2s**: DPM++ 2S sampler (second-order singlestep)

Your implementation should aim to achieve higher CLIP scores than all baselines with the same NFE=20 budget.

# Custom Sampler for Diffusion Bridge Models

## Objective
Design and implement a novel, superior sampling algorithm for Diffusion Bridge Models. Your implementation must be written inside the `sample_custom_bridge` function in `ddbm/karras_diffusion.py`. The evaluation pipeline will dynamically call this function to generate target images from source conditions.

## Background
Diffusion Bridge Models enable high-quality image-to-image (I2I) translation by creating stochastic or deterministic paths between two arbitrary distributions (e.g., from a sketch to a realistic image). The codebase provides references to three foundational approaches:
- **DDBM (Denoising Diffusion Bridge Models)**: Simulates the bridge using a continuous Fokker-Planck/SDE formulation.
- **DBIM (Diffusion Bridge Implicit Models)**: An accelerated method that analytically decouples the trajectory into explicit coefficients (`coeff_x0_hat`, `coeff_xT`, `coeff_xs`) to jump across large time steps efficiently.
- **ECSI (Endpoint-Conditioned Stochastic Interpolants, Zhang et al. 2024)**: A Euler discretization of the reverse bridge SDE using a `z_hat` (noise) reparameterization with explicit stochasticity control (`ε_t = η·(γ γ̇ − (α̇/α)γ²)`), falling back to DBIM on the final two steps for endpoint sharpness.

Your goal is to design a sampling kernel that synthesizes these strengths or introduces a completely novel mathematical transition step.

## Codebase
This task evaluates your sampler on the **Edges2Handbags** image-to-image translation dataset. 
- **Metric**: **FID (Fréchet Inception Distance)**. A lower FID indicates higher generation quality and better diversity. 
- **Efficiency**: The total Number of Function Evaluations (NFE) will also be tracked. Your sampler should maintain competitive inference speed.

Your `sample_custom_bridge` is integrated into the testing pipeline via the benchmark's execution script. 

## Interface Contract
You are permitted to write your novel logic inside the following function. 

```python
@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    **kwargs
):
    # x: initial state tensor (e.g., source image with noise)
    # ts: time schedule tensor (decreasing from t_max to 0)
    # eta: scale for stochasticity
    
    # ... YOUR CUSTOM SAMPLING LOGIC HERE ...

    # MUST return exactly these 6 variables in this order:
    return x, path, nfe, pred_x0, ts, first_noise
```

**Constraints**:
- You must NOT modify the function signature (name, arguments, or return structure). The outer `sample.py` loop strictly expects a tuple of `(final_image, sampling_path, num_function_evals, predicted_x0_list, time_schedule, initial_noise)`.
- You must NOT alter how external hyper-parameters (like `guidance_scale` or `corrupt_scale`) are parsed from environment variables.
- **The only hard rule on NFE**: you may call `denoiser(...)` at most `len(ts)` times total. The pipeline wraps the callable with a counter; the `(len(ts)+1)`-th call raises `RuntimeError: NFE_BUDGET_EXCEEDED` and the run is rejected. How you allocate those calls and schedule stochasticity is entirely your choice.

## Reference Baseline FIDs (NFE=5)
| baseline | edges2handbags | Imagenet (center-inpaint) |
|---|---|---|
| `dbim`            | 5.180 | 6.070 |
| `ddbm` (50 NFE)   | 11.139 | 10.556 |
| `ecsi`            | **4.234** | N/A |
| `dbimho`          | 5.528 | **5.528** |

Your goal: beat the **best-per-env** baselines — `ecsi`=4.234 on edges2handbags and `dbimho`=5.528 on Imagenet — while staying within the NFE=5 budget.

## Hints
- **Baseline Expectation**: Your custom sampler MUST achieve a lower FID score than the standard `DBIM` baseline.
- **Stretch Goal**: Aim to match or surpass the state-of-the-art `ECSI` baseline in both sample quality and conditional diversity.
- **SDE/ODE Modulation & Trajectory**: Since the marginal distributions (the schedules for $x_0$, $x_T$, and noise) are strictly fixed by the underlying VP schedule, **do not arbitrarily alter the foundational closed-form coefficients**. Instead, focus on how to better modulate the ratio of SDE (stochastic) to ODE (deterministic) components across the timestep sequence.
- **Stochasticity scheduling**: How should the stochasticity parameter (`eta` or `epsilon_t`) behave across the trajectory to balance exploration and artifact removal?

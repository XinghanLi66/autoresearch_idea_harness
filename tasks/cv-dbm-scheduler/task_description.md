# Task: Time Scheduler for Diffusion Bridge Models (NFE=5)

## Background
In diffusion bridge sampling, the time schedule controls the discretization step sizes. Currently, the `uniform` (linear) and `karras` schedules are the most widely used baselines in the field.

## Objective
Design a novel time schedule optimized specifically for extremely low-step sampling (**NFE = 5**). Your goal is to achieve a better generation quality (lower FID) than the standard baselines.

⚠️ **CRITICAL CONSTRAINT**: 
To maintain compatibility with our evaluation interface, your code **MUST** be written inside the function named `get_sigmas_uniform`. Please ignore the function name—do **NOT** implement a basic linear/uniform schedule. Use this exact function slot to implement your new, advanced mathematical curve.

## Implementation
```python
import torch

def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    """
    Requirements:
    1. Length: Must return a 1D PyTorch tensor of exactly length `n + 1`.
    2. Monotonic: The sequence must strictly decrease from `t_max` to `t_min`.
    3. Terminal Value: The final element (index `n`) must exactly equal `t_min`.
    4. Device: Move the tensor to the requested `device`.
    """
    # For this task, n will typically be 5 (NFE=5).
    # Implement your novel schedule formulation here...
    
    # Example return:
    # return sigmas.to(device)
```

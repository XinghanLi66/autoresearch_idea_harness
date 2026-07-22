# Foundational PDE Model: In-Context Operator Architecture Design

## Research Question
Design a model architecture for in-context operator learning that can predict PDE dynamics across multiple heterogeneous physics domains from a few demonstration examples, without any fine-tuning or task-specific adaptation.

## Background
Traditional neural PDE solvers (FNO, Transformers, etc.) are trained on a single PDE type and cannot generalize to new physics. Recent foundational PDE models aim to learn a single model that works across diverse PDE families. **In-context operator learning** is a particularly promising paradigm: the model receives a few (input, output) demonstration pairs showing how a PDE evolves, then predicts the output for a new query input — analogous to few-shot learning in NLP.

The key challenge is designing an architecture that can:
1. Efficiently process 2D PDE fields (128x128 resolution, multiple channels)
2. Distinguish between different demonstration pairs and their roles (input vs. output)
3. Enforce causal structure so the query prediction cannot "peek" at the answer
4. Generalize across fundamentally different physics (incompressible vs. compressible flow)

## Task
Modify the `CustomModel` class in `VICON/src/custom_model.py` (lines 89-205) to implement your model architecture. The model receives demonstration pairs showing PDE evolution and must predict the output state for a query input.

## Interface
```python
class CustomModel(nn.Module):
    def __init__(self, cfg):
        # cfg contains model hyperparameters (see docstring for details)
        ...

    def forward(self, x_tuple):
        # x_tuple = (init, end, c_mask)
        #   init: (B, pairs, C=7, H=128, W=128) — input states
        #   end:  (B, pairs, C=7, H=128, W=128) — output states
        #   c_mask: (B, C=7) — channel validity mask
        # Returns: (B, pairs, C, H, W) — predicted outputs
        ...
```

The model is called with `pairs` demonstration pairs. Each pair consists of an input state and output state. The last pair is the query whose output must be predicted. During training, all pair outputs are supervised; during evaluation, only the query prediction matters.

Available utility functions (in the FIXED section):
- `patchify(x, patch_num)`: Convert images to patches
- `depatchify(patches, patch_num, c, h, w)`: Convert patches back to images
- `build_alternating_block_lowtri_mask(block_num, block_size1, block_size2)`: Build causal attention mask

## Evaluation
The model is trained jointly on 3 fluid dynamics benchmarks and evaluated on each:
- **NS2D**: Incompressible Navier-Stokes (PDEArena), channels=[vx, vy, scalar]
- **COMPRESSIBLE2D**: Compressible NS with high viscosity (PDEBench), channels=[density, vx, vy, pressure]
- **EULER2D**: Compressible NS with near-zero viscosity (PDEBench), channels=[density, vx, vy, pressure]

Metrics (lower is better):
- `last_step_rel_err`: Relative L2 error at the last prediction step
- `all_avg_rel_err`: Average relative L2 error across all prediction steps


# Task: Unconditional 3D Structure Generation

## Research Question
Design a generative model for unconditional 3D structure generation that works across both molecular (QM9, GEOM-DRUG) and crystalline (MP-20) domains. The model must learn to generate realistic 3D atomic structures from scratch, without any conditioning information.

## Background
3D structure generation is a fundamental problem in computational chemistry and materials science. The goal is to learn a generative model p(x) over 3D atomic structures, where each structure consists of atom types and 3D coordinates. Key challenges include:
- **SE(3) equivariance**: The generative process should respect rotational and translational symmetry of physical space.
- **Variable-size structures**: Molecules and crystals have different numbers of atoms.
- **Dual domains**: Molecules use Cartesian coordinates in free space, while crystals use fractional coordinates within a periodic lattice.

Existing approaches include:
- **EDM**: Equivariant Diffusion Models using EGNN-based score networks with DDPM denoising.
- **GeoBFN**: Geometric Bayesian Flow Networks with continuous normalizing flows on SE(3).
- **GeoLDM**: Geometric Latent Diffusion Models with equivariant VAE + latent space diffusion.

## What to Implement
Implement the `StructureGenerator` class in `custom_mol3d.py`. You must implement:
1. `__init__(self, config)`: Set up your model architecture and any required components.
2. `compute_loss(self, batch) -> loss`: Compute training loss given a batch of structures.
3. `sample(self, n_samples, atom_counts, device, **kwargs) -> dict`: Generate new structures.

## Batch Format

### Molecules (QM9, GEOM-DRUG):
```python
batch = {
    'positions': (B, max_atoms, 3),      # Cartesian coordinates, zero-padded
    'atom_types': (B, max_atoms),        # integer atom type indices, 0=padding
    'num_atoms': (B,),                    # actual number of atoms per sample
    'mask': (B, max_atoms),              # 1 for real atoms, 0 for padding
    'dataset_type': 'molecule',
}
```

### Crystals (MP-20):
```python
batch = {
    'frac_coords': (B, max_atoms, 3),   # fractional coordinates [0,1)
    'atom_types': (B, max_atoms),        # integer atom type indices, 0=padding
    'num_atoms': (B,),                    # actual number of atoms per sample
    'mask': (B, max_atoms),              # 1 for real atoms, 0 for padding
    'lattice': (B, 3, 3),               # lattice matrix
    'dataset_type': 'crystal',
}
```

## Evaluation
The model is tested on three benchmarks:

### Molecules (QM9, GEOM-DRUG):
- **atom_stability**: Fraction of atoms with valid valence (based on bond distance analysis).
- **mol_stability**: Fraction of molecules where all atoms are stable.
- **validity**: Fraction of molecules that pass RDKit sanitization.
- **uniqueness**: Fraction of valid molecules with unique SMILES representations.

### Crystals (MP-20):
- **validity**: Fraction of crystals passing SMACT composition + structure distance checks.
- **cov_recall**: Coverage recall — fraction of test set structures matched by generated set.
- **cov_precision**: Coverage precision — fraction of generated structures matching test set.
- **prop_wdist_density**: Wasserstein distance of density distributions.
- **prop_wdist_num_elems**: Wasserstein distance of element count distributions.

## Editable Region
Lines 42-320 of `custom_mol3d.py` are editable (the section between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers). You may define any helper classes, layers, or functions within this region. The region must contain a `StructureGenerator` class with the specified interface (`__init__`, `compute_loss`, `sample`).

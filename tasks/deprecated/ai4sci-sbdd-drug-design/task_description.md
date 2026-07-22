# Task: Structure-Based Drug Design

## Research Question
Design a generative model for pocket-conditioned 3D molecule generation that works
across three drug design scenarios: de novo design, linker design, and fragment growing.
The model must learn to generate realistic drug-like molecules that bind to protein
pockets, using the CrossDocked2020 dataset.

## Background
Structure-based drug design (SBDD) aims to generate molecules that bind to a target
protein pocket with high affinity. The key challenges include:
- **3D geometry**: Generated molecules must have valid 3D conformations.
- **Protein conditioning**: The model must be aware of the pocket structure.
- **Chemical validity**: Generated molecules should satisfy valence rules and be
  drug-like (good QED, SA scores).
- **Binding affinity**: Molecules should dock well to the target pocket (low Vina score).

Three design scenarios test different aspects of generalization:
- **De novo**: Generate the entire molecule from scratch given a protein pocket.
- **Linker**: Given two molecular fragments bound to a pocket, generate the linker
  atoms connecting them.
- **Fragment**: Given a molecular fragment, grow additional atoms to complete the molecule.

Existing approaches include:
- **TargetDiff**: Diffusion model using UniTransformer for joint position + atom denoising.
- **DiffBP**: Diffusion model with CoM prediction and interior regularization.
- **Pocket2Mol**: Autoregressive model using GVP for sequential atom/bond generation.

## What to Implement
Implement the `CustomSBDD` class in `custom_sbdd.py`. You must implement:
1. `__init__(self, cfg)`: Set up your model architecture.
2. `forward(self, batch)`: Training forward pass returning `(loss_dict, results)`.
3. `sample(self, batch)`: Generate molecules returning trajectory dict.

## Batch Format (PyG HeteroData, flattened)
```python
batch = {
    'ligand_pos':           [N_lig, 3],    # ligand atom 3D positions
    'ligand_atom_type':     [N_lig],       # atom type indices (0-12, add_aromatic)
    'ligand_lig_flag':      [N_lig],       # True for all ligand atoms
    'ligand_gen_flag':      [N_lig],       # True for atoms to generate
    'ligand_element_batch': [N_lig],       # batch index per ligand atom
    'protein_pos':          [N_rec, 3],    # protein atom positions
    'protein_atom_feature': [N_rec],       # protein atom features
    'protein_aa_type':      [N_rec],       # amino acid type (0-19)
    'protein_lig_flag':     [N_rec],       # False for protein atoms
    'protein_element_batch':[N_rec],       # batch index per protein atom
}
```

For linker/frag tasks: `ligand_gen_flag` distinguishes context atoms (False) from
atoms to be generated (True). In de novo mode, `gen_flag == lig_flag` (all True).

## Model Interface
- `forward(batch)` returns `(loss_dict: Dict[str, Tensor], results: Dict)`.
  - `loss_dict` keys are weighted by `config.train.loss_weights` (e.g., pos=1.0, atom=100.0).
- `sample(batch)` returns trajectory dict `{timestep: (pos, atom_type_probs, batch_idx)}`.
  - Final generated molecule = `trajectory[0]` (last denoising step).
  - `pos`: [N_lig, 3] atom positions; `atom_type_probs`: [N_lig, 13] type logits/probs.

## Evaluation
The model is tested on 3 scenarios (denovo, linker, frag). For each:
- **qed**: Drug-likeness score (0-1, higher is better)
- **sa**: Synthetic accessibility (1-10, lower is better)
- **validity**: Fraction of generated molecules passing RDKit sanitization (higher is better)
- **vina_score**: Vina score-only binding affinity (kcal/mol, lower is better)
- **vina_dock**: Vina re-docked binding affinity (kcal/mol, lower is better)

## Editable Region
Lines 42-310 of `custom_sbdd.py` (between `EDITABLE SECTION START` and `EDITABLE SECTION END`
markers). You may define helper classes, layers, or functions within this region.
The region must contain a `CustomSBDD` class decorated with `@register_model('custom')`
implementing the `forward` and `sample` interface.

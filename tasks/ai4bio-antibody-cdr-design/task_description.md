# Task: Antibody CDR Sequence-Structure Co-Design

## Research Question
Design a novel generative architecture for antibody complementarity-determining region (CDR) loop sequence-structure co-design. The model must jointly generate CDR amino acid sequences and 3D CA backbone coordinates conditioned on the antigen-antibody framework context.

## Background
Antibodies are Y-shaped immune proteins that bind antigens through six hypervariable loops called CDRs (H1-H3, L1-L3). Designing CDR loops that bind specific epitopes is central to therapeutic antibody engineering. This requires jointly optimizing:
- **Sequence**: amino acid identity at each CDR position (determines binding specificity)
- **Structure**: 3D backbone conformation (determines shape complementarity)
- **Epitope conditioning**: the generated CDR must form contacts with the target epitope

State-of-the-art approaches use diffusion models (DiffAb), equivariant GNNs (dyMEAN, MEAN), flow matching, and iterative refinement to generate CDR loops. The task evaluates generalization across three biologically motivated data splits.

## What to Implement
Modify the `CustomCDRModel` class in `custom_cdr.py` (lines 178-301). You must implement:

1. `__init__(self, ...)`: Define your model architecture.
2. `forward(self, batch)`: Training forward pass. Receives a list of sample dicts, returns a dict of named losses (e.g., `{'seq': tensor, 'coord': tensor}`).
3. `sample(self, batch)`: Inference. Receives a list of sample dicts, returns a list of prediction dicts with keys: `complex_id`, `cdr_type`, `pred_sequence`, `true_sequence`, `pred_coords`, `true_coords`, `ppl`.

You may define helper classes/modules within the editable region. The starter code is a simple MLP that runs but performs poorly.

## Data Format
Each sample is a dict with:
- `heavy_seq` / `light_seq`: amino acid sequences (str)
- `heavy_coords` / `light_coords`: CA coordinates `[L, 3]` (Tensor)
- `ag_coords`: antigen CA coordinates `[L_ag, 3]` (Tensor)
- `ag_surface`: antigen surface chemical features `[128, 6]` (Tensor)
- `cdr_info`: dict mapping CDR label (e.g., "H3") to `{'indices': ndarray, 'seq': str, 'coords': ndarray [N,3], 'chain': str}`

## Evaluation
The model is tested on 3 CHIMERA-Bench splits (epitope_group, antigen_fold, temporal). Following CHIMERA-Bench Track 1 and the original DiffAb/MEAN/dyMEAN papers, **AAR, RMSD, and TM-score are reported on CDR-H3 only** — the most variable and challenging loop. The model must still generate all 6 CDRs (H1-H3, L1-L3); only H3 is scored. Metrics computed:
- **Sequence**: AAR (amino acid recovery), CAAR (contact AAR), PPL
- **Structure**: RMSD (Kabsch-aligned CA), TM-score
- **Binding**: Fnat, iRMSD, DockQ
- **Epitope**: Epitope F1
- **Designability**: n_liabilities
- **Composites**: CHIMERA-S (structural), CHIMERA-B (binding)

Higher AAR, TM-score, Fnat, DockQ, EpitopeF1, CHIMERA-S, CHIMERA-B is better. Lower RMSD is better.

## Editable Region
Lines 178-301 of `custom_cdr.py`.

# Task: Protein Sequence Encoder Design for Function Prediction

## Research Question
Design a novel protein sequence encoder architecture that learns effective representations for predicting protein function from amino acid sequences. The goal is a general-purpose encoder that transfers well across diverse protein property prediction tasks.

## Background
Protein function prediction from sequence is a fundamental problem in computational biology. Given a protein's amino acid sequence (up to ~1000 residues), the model must learn structural and functional patterns to predict properties like enzymatic activity, fluorescence, and solubility.

Current approaches fall into three categories:
- **Sequential models**: CNN and RNN-based encoders that treat the sequence as a 1D signal (e.g., convolutional filters over one-hot amino acid encodings).
- **Attention models**: Transformer-based encoders that capture long-range residue interactions via self-attention over learned token embeddings.
- **Graph models**: GNN-based encoders that construct residue contact graphs and perform message passing (e.g., GCN on k-nearest neighbor sequence graphs).

Key challenges include:
- **Long sequences**: Proteins can have hundreds to thousands of residues; efficient encoding of long-range dependencies is critical.
- **Sparse vocabulary**: Only 20 standard amino acids, but local context and global structure both matter.
- **Cross-task generalization**: The same encoder must work for regression (continuous property values) and classification tasks.

## What to Implement
Implement the `ProteinEncoder` class in `custom_protein.py` (lines 107-177). You must implement:
1. `__init__(self, vocab_size, onehot_dim, max_seq_len)`: Initialize your encoder architecture.
2. `forward(self, batch: ProteinBatch) -> Tensor [B, output_dim]`: Encode a batch of protein sequences into fixed-size representations.

Your encoder must set `self.output_dim` (an integer) so the downstream prediction head knows the representation size.

## Input Format (ProteinBatch)
```python
class ProteinBatch:
    token_ids: Tensor     # [B, max_len] integer amino acid indices (0=pad, 1-20=amino acids)
    onehot: Tensor        # [B, max_len, 20] one-hot encoded sequences
    mask: Tensor          # [B, max_len] attention mask (1=real residue, 0=padding)
    seq_lengths: Tensor   # [B] actual sequence lengths
    targets: Tensor       # [B] regression/classification targets
```

You may use any combination of these input formats. The `token_ids` are suitable for embedding layers, `onehot` for convolutional or linear layers, and `mask` for attention-based models.

## Amino Acid Vocabulary
20 standard amino acids (ACDEFGHIKLMNPQRSTVWY), index 0 is padding. Maximum sequence length is 1000 (truncated/padded).

## Evaluation
The encoder is evaluated on 3 protein property prediction benchmarks from the Therapeutics Data Commons (TDC):

### Regression (metric: Spearman correlation, higher is better):
- **Beta-lactamase**: Enzyme activity prediction (~4,158 proteins, TAPE benchmark)
- **Fluorescence**: GFP fluorescence intensity (~21,446 proteins, TAPE benchmark)

### Binary classification (metrics: PR-AUC and F1, higher is better):
- **Solubility**: Protein solubility prediction (binary: soluble vs insoluble; ~62,478 proteins, PEER/DeepProtein benchmark). Following DeepProtein, this task is treated as a binary classification problem (`config['binary']=True`) and evaluated with PR-AUC / F1 / accuracy / ROC-AUC.

The training loop automatically uses `BCEWithLogitsLoss` for Solubility and `MSELoss` for the
two regression tasks; you only need to design the encoder. All benchmarks use pre-defined
train/valid/test splits.

## Editable Region
Lines 107-177 of `custom_protein.py` are editable (between `EDITABLE SECTION START` and `EDITABLE SECTION END` markers). You may define helper classes, layers, or functions within this region. The region must contain a `ProteinEncoder` class with the specified interface.

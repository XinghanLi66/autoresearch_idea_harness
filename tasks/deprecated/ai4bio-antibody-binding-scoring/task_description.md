# Task: Antibody Binding Affinity Zero-Shot Scoring

## Research Question
Design a **zero-shot** scoring function over a frozen protein language model (ESM-2 3B, MLM head — `facebook/esm2_t36_3B_UR50D`, the exact checkpoint used by AbBiBench's ESM-2 baseline, see `vendor/external_packages/AbBiBench/models/ESM-2/get_model_log_likelihood.py:130-131`) that assigns higher likelihood to antibody variants with higher experimental binding affinity. No supervised training on binding labels is permitted — the function may only use the wild-type antibody-antigen complex, a candidate mutant heavy/light chain, and forward passes through the frozen ESM-2 model.

## Background
AbBiBench [Tang et al., 2025 — https://openreview.net/forum?id=UN26B1826Y, https://github.com/MSBMI-SAFE/AbBiBench] evaluates antibody scoring models purely in the zero-shot setting: for each antibody–antigen complex dataset, it computes the Spearman rank correlation between a model's per-variant log-likelihood (or likelihood proxy) and experimentally measured binding scores. No model in the AbBiBench leaderboard is trained on binding labels — the benchmark specifically measures whether pretrained protein/antibody models already encode binding-relevant signal.

Published AbBiBench Spearman numbers (full-set, ESM-2 3B, mean log-likelihood over the full complex — the simplest zero-shot protocol of Meier et al., 2021) on the splits used here, taken directly from the AbBiBench README leaderboard table at https://github.com/MSBMI-SAFE/AbBiBench (row "ESM2"):
- **3gbn_h1 (influenza H1)**: 0.23 (ESM-2 mean PLL, full set ~1.9k rows)
- **4fqi_h1 (influenza H1, different complex)**: -0.02 (ESM-2 mean PLL, **full** ~65k rows; our task subsamples to 5000 rows with `np.random.RandomState(seed=42).choice(...)`, which can shift the Spearman by a few hundredths — the `esm2_pll` baseline run on this task measures the actual 5000-row reference)
- **4d5_her2 (HER2)**: this dataset is outside the published AbBiBench README leaderboard (the README table's `1n8z` column is the bevacizumab/VEGF DMS, not 4d5_her2); the `esm2_pll` baseline run on this task measures the reference Spearman.

Better zero-shot scoring functions exist: masked-marginal scoring at mutated positions [Meier et al., 2021 — https://doi.org/10.1101/2021.07.09.450648] typically beats mean PLL on DMS by focusing on sites that actually changed; wild-type-normalized scores (log P(mutant) − log P(wildtype) computed via masked or unmasked forward passes) can further improve correlation. Structure-conditioned inverse folding models (ESM-IF, ProteinMPNN) reach 0.59–0.65 on 3gbn_h1 and 4fqi_h1 by conditioning on the antigen structure — but those are out of scope here because we restrict to ESM-2 sequence features.

The research question is therefore: **can a sequence-only zero-shot scoring function over frozen ESM-2 3B meaningfully improve over the mean-PLL / masked-marginal baselines?** Candidate directions include antigen-conditioned context windows, wild-type-relative scoring, per-position weighting (e.g. CDR regions or positions near the paratope), ensembling multiple scoring modes, or using ESM-2 hidden-state geometry (e.g. cosine distance of per-residue embeddings between mutant and wild-type).

## What to Implement
Implement the `ScoringFunction` class in `custom_abscore.py`. You must implement:
1. `__init__(self, esm_model, esm_tokenizer, device)`: Initialize; store references to the frozen ESM-2 MLM model and tokenizer. **No trainable parameters beyond constants/look-ups.** The grading harness checks that your scoring function consumes no binding labels.
2. `score_batch(self, wt_heavy: str, wt_light: str, wt_antigen: str, mut_heavy_seqs: List[str], mut_light_seqs: List[str]) -> List[float]`: Return one scalar likelihood-proxy per variant; higher should correspond to higher binding. Allowed: any number of forward passes through `self.esm_model` / `self.esm_tokenizer`. Not allowed: access to `binding_score` (the harness never passes labels into this function).
3. No loss/optimizer is trained — there is no `compute_loss`. The test loop simply calls `score_batch` over all CSV rows and reports Spearman correlation.

## Input
Per-dataset the harness provides:
- `wt_heavy`, `wt_light`, `wt_antigen`: wild-type sequences extracted from the AbBiBench PDB (parsed via Biopython).
- A CSV of mutant variants with columns `heavy_chain_seq` (or `mut_heavy_chain_seq`), `light_chain_seq` (optional; falls back to WT), and `binding_score` (hidden from your scoring function; only used by the harness to compute Spearman).

## Evaluation
Three AbBiBench splits, all zero-shot (no train/val/test split — every row is a test row because we never train):

| Label | Dataset | Antigen | Rows (approx.) |
|-------|---------|---------|----------------|
| influenza | 3gbn_h1 | Influenza hemagglutinin H1 | ~1.9k |
| sars | 4fqi_h1 | Influenza H1 (4fqi HA stem broadly-neutralizing antibody complex) — **note**: the task split label is `sars` for historical task-draft reasons, but the underlying dataset is influenza H1 (4fqi), NOT SARS-CoV-2. The label is preserved for backward compatibility with leaderboard/score-spec naming. | **subsampled to 5000** (full 65k is compute-infeasible for per-position scoring within 24h) |
| her2 | 4d5_her2 | HER2 (trastuzumab Fab) | ~2.1k |

**Metric**: `spearman_<label>` = Spearman rank correlation between your scores and the experimental `binding_score`. Higher is better. No MSE term (your scores are on an arbitrary scale — only ranks matter).

**Reference points (computed on the same CSVs with ESM-2 3B, seed 42)**:
- `mean_pll` baseline (Meier et al., 2021 — mean token log-prob of `heavy+light+antigen` under one unmasked forward pass, the exact AbBiBench ESM-2 protocol implemented in `vendor/external_packages/AbBiBench/models/ESM-2/get_model_log_likelihood.py:62-98`): the AbBiBench README leaderboard (https://github.com/MSBMI-SAFE/AbBiBench, "ESM2" row) reports Spearman 0.23 on 3gbn_h1 and -0.02 on 4fqi_h1 (full 4fqi_h1 set, NOT subsampled). Our pre-run on 4d5_her2 (not in the AbBiBench README leaderboard) under the same protocol is weakly negative.
- `masked_marginal` baseline (Meier et al., 2021, Eq. 2 — for each mutated heavy-chain position p, mask position p in the WT complex and read `log P(mut_aa | wt_seq with p masked) − log P(wt_aa | wt_seq with p masked)` from that single masked forward; sum over mutated positions): the canonical "masked marginal" zero-shot DMS score. Cost is one masked forward per WT heavy-chain position (cached per dataset).
- `wildtype_delta` baseline (Meier et al., 2021 §2.2 / Brandes et al., 2023 "ESM1b LLR", a.k.a. "wildtype marginal" — for each mutated heavy-chain position p, do ONE unmasked forward over the mutant complex and read `log P(mut_aa | mutant_ctx) − log P(wt_aa | mutant_ctx)` at position p; sum over mutated positions): an alternative zero-shot scoring that costs one unmasked forward per variant and is mathematically distinct from both `mean_pll` (which averages over all positions) and `masked_marginal` (which uses masked WT context).
- Upper bounds from the AbBiBench paper on the same CSVs (using much larger / structure-conditioned models, NOT available to you): ProteinMPNN 0.59 / 0.61 / 0.32, ESM-IF1 0.59 / 0.65 / n.a., diffab 0.67 / 0.00 / n.a.

**Expected agent target**: improve over the best ESM-2-sequence-only baseline (masked_marginal) on at least 2 of the 3 splits. A strong result is Spearman ~0.35 on influenza, ~0.15 on sars, ~0.10 on her2 — i.e. closing some of the gap to the structure-conditioned upper bound without using structure. Numbers above ~0.5 on influenza/sars with a sequence-only ESM-2 3B zero-shot scorer would be publishable; the task is explicitly designed so that reaching the paper's inverse-folding numbers (0.59+) is not expected from ESM-2 alone.

## Constraints
- **No binding-label training.** The budget-check harness imports your `custom_abscore.py` module and asserts `ScoringFunction` contains no `torch.nn.Parameter` instances that get updated (the module has no training loop — any `requires_grad=True` tensor will be ignored).
- **ESM-2 3B only.** You may not load additional pretrained models (the container has `facebook/esm2_t36_3B_UR50D` cached; any other model load from the hub will fail at runtime because compute nodes have no network).
- **Sequence-only.** You receive raw amino-acid sequences, not 3D structure.
- **Compute budget.** 4fqi_h1 is subsampled to 5000 rows; total per-split budget is 24h on 1 GPU. A method doing ~1-2 forward passes per variant over a 450-residue complex is fine; masked-marginal at ALL positions over the full 65k set would not fit — that's why subsampling plus restricting masked passes to mutated positions is standard practice.

## Editable Region
Lines 133-167 of `custom_abscore.py` (between `EDITABLE SECTION START` and `EDITABLE SECTION END`). The region must contain a `ScoringFunction` class with the specified interface.

## Disclosure: Reference number provenance and baseline protocol fidelity
The published AbBiBench Spearman numbers cited above (3gbn_h1=+0.23, 4fqi_h1=-0.02, 4d5_her2=-0.20 on the same CSVs at ESM-2 3B) come from the AbBiBench upstream `get_model_log_likelihood.py` "mean PLL" implementation which contains a known **off-by-one indexing convention**: see `vendor/external_packages/AbBiBench/models/ESM-2/get_model_log_likelihood.py:85-96`. There, the loop iterates `i, token` over the tokenizer output (which excludes the BOS token), but indexes `probs[0, i, token_idx]` directly — even though `probs` is produced from input that *does* include the BOS token. As a result, each amino acid's log-probability is read from the model's output at the position of the *previous* token (BOS for `i=0`, AA1 for `i=1`, etc.), not at its own position. This is mathematically distinct from the textbook Meier 2021 mean PLL and is propagated through the AbBiBench README leaderboard.

Baselines in this task are deliberately split into two categories with respect to this issue:

- **`esm2_pll`** — strictly reproduces the AbBiBench upstream protocol *including the off-by-one indexing*, so its Spearman numbers match the AbBiBench README leaderboard "ESM2" row exactly. Treat this as the "AbBiBench-reported reference", not as a textbook mean PLL.
- **`masked_marginal`** — paper-faithful Meier et al. 2021 Eq. 2 masked-marginal scoring at mutated positions, with correct (non-shifted) indexing.
- **`delta_mlp`** (a.k.a. `wildtype_delta`) — paper-faithful Meier 2021 §2.2 / Brandes et al. 2023 ESM1b LLR "wildtype marginal" scoring, with correct indexing.

Agents are evaluated on raw Spearman against experimental binding scores; they are not required to match or to deviate from the upstream off-by-one convention — they may freely choose any indexing as long as their `ScoringFunction` is sequence-only zero-shot over frozen ESM-2 3B.

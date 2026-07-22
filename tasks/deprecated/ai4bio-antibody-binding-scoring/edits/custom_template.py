"""
Antibody Binding Affinity Zero-Shot Scoring — Self-contained template.

Protocol (AbBiBench zero-shot, Tang et al., 2025,
https://github.com/MSBMI-SAFE/AbBiBench): for each variant, a scoring
function produces a scalar likelihood-proxy over a frozen ESM-2 MLM.
The harness reports Spearman rank correlation between the scores and
experimentally measured binding_score values. No supervised training
on binding labels is permitted.

Structure:
  Lines 1-132:   FIXED — Imports, data loading, ESM-2 MLM loading, helpers
  Lines 133-167: EDITABLE — ScoringFunction class (starter: mean-PLL of concat)
  Lines 168+:    FIXED — Evaluation loop, CLI
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional

import torch
import torch.nn.functional as F

from scipy.stats import spearmanr
from Bio import PDB
from Bio.SeqUtils import seq1

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# FIXED — Constants and utilities
# =====================================================================

ESM_MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
ESM_EMBED_DIM = 2560  # ESM-2 3B hidden dimension
# ESM-2 3B uses rotary position embeddings (config.position_embedding_type=
# 'rotary'), so it has no hard cap from learned positional tables. The
# canonical AbBiBench ESM-2 script uses no explicit truncation. We raise
# the cap to 2048 so that 4d5_her2 (heavy ~120 + light ~107 + antigen
# ~1015 ~= 1242 residues) is NOT silently truncated at the antigen C-
# terminus the way the previous 1022 cap did. Rotary extrapolation past
# the training context (1024) is mild for one-sequence-per-pass scoring.
MAX_SEQ_LEN = 2048


def extract_sequences_from_pdb(pdb_file, heavy_chain, light_chain, antigen_chains):
    """Extract amino acid sequences from PDB file chains.

    Antigen chains are concatenated in the order given by
    `antigen_chains` (i.e. the order specified in metadata.json), NOT
    PDB iteration order. This matches AbBiBench's notebook conventions
    and makes the antigen sequence deterministic regardless of how the
    PDB was written. (The upstream
    AbBiBench/models/ESM-2/get_model_log_likelihood.py uses PDB-iter
    order; for the antibody-DMS datasets used here both orders coincide
    for single-antigen-chain complexes 4fqi (chain A) and 4d5_her2
    (chain C of 1n8z_bac.pdb), but we still pin to metadata order for
    safety.)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", pdb_file)
    heavy_seq, light_seq = "", ""
    antigen_chain_seqs = {}
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                if residue.get_resname() in PDB.Polypeptide.standard_aa_names:
                    seq.append(seq1(residue.get_resname()))
            chain_seq = "".join(seq)
            if chain.id == heavy_chain:
                heavy_seq = chain_seq
            elif chain.id == light_chain:
                light_seq = chain_seq
            elif chain.id in antigen_chains:
                antigen_chain_seqs[chain.id] = chain_seq
    # Concatenate in metadata-specified order (NOT PDB-iter order).
    antigen_seq = "".join(antigen_chain_seqs.get(c, "") for c in antigen_chains)
    return heavy_seq, light_seq, antigen_seq


@torch.no_grad()
def esm_forward_logprobs(esm_model, esm_tokenizer, sequence: str, device,
                          max_len: int = MAX_SEQ_LEN):
    """Single forward pass, returns per-position log-probabilities over the
    full vocabulary and the tokenized input ids (excluding BOS/EOS).

    Args:
        sequence: amino acid string (will be truncated to max_len).
    Returns:
        logprobs: Tensor [L, V] — log-softmax over vocab per residue position
                  (L = len(tokenized sequence) without BOS/EOS).
        token_ids: LongTensor [L] — tokenized residue ids (without BOS/EOS).
        tokenizer: the tokenizer (for vocab lookup).
    """
    sequence = sequence[:max_len]
    enc = esm_tokenizer(sequence, return_tensors="pt",
                        padding=False, truncation=True,
                        max_length=max_len + 2, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = esm_model(**enc)
    logits = out.logits[0]  # [L_full, V]
    logprobs_full = F.log_softmax(logits, dim=-1)
    # Strip BOS/EOS positions. ESM-2 adds <cls> at 0 and <eos> at -1.
    logprobs = logprobs_full[1:-1]         # [L, V]
    token_ids = enc["input_ids"][0, 1:-1]  # [L]
    return logprobs, token_ids


@torch.no_grad()
def esm_mean_token_logprob(esm_model, esm_tokenizer, sequence: str, device,
                            max_len: int = MAX_SEQ_LEN) -> float:
    """Mean log P(x_i | x_{1..L}) under the MLM, unmasked context.

    This is the exact "mean PLL with full context" protocol used by
    AbBiBench for ESM-2 (see AbBiBench/models/ESM-2/get_model_log_likelihood.py,
    mean over per-position log P without masking).
    """
    logprobs, token_ids = esm_forward_logprobs(
        esm_model, esm_tokenizer, sequence, device, max_len)
    per_pos = logprobs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [L]
    return float(per_pos.mean().item())


# =====================================================================
# EDITABLE SECTION START — ScoringFunction (lines 133-167)
# =====================================================================

class ScoringFunction:
    """Zero-shot scoring function over a frozen ESM-2 3B MLM.

    Starter implementation: mean token log-probability over the
    concatenated (heavy + light + antigen) sequence — the baseline
    AbBiBench uses for ESM-2 (Meier et al., 2021). Expected Spearman
    on this task's three splits: ~0.20 / ~0.00 / ~-0.20.

    You should replace this with a better zero-shot scorer (masked
    marginals at mutated positions, WT-delta, per-chain weighting,
    antigen-context ensembling, embedding-distance score, etc.).

    Contract:
      - Must NOT train on binding labels (the harness never passes them).
      - Must NOT load additional pretrained models from the hub.
      - May run any number of forward passes through self.esm_model.
      - Return one scalar per variant; higher = higher predicted affinity.
    """

    def __init__(self, esm_model, esm_tokenizer, device):
        self.esm_model = esm_model
        self.esm_tokenizer = esm_tokenizer
        self.device = device

    def score_batch(self, wt_heavy: str, wt_light: str, wt_antigen: str,
                    mut_heavy_seqs: List[str], mut_light_seqs: List[str]
                    ) -> List[float]:
        """Score a batch of variants. Returns List[float], one per variant."""
        scores = []
        for mh, ml in zip(mut_heavy_seqs, mut_light_seqs):
            complex_seq = mh + ml + wt_antigen
            s = esm_mean_token_logprob(
                self.esm_model, self.esm_tokenizer, complex_seq, self.device)
            scores.append(s)
        return scores

# =====================================================================
# EDITABLE SECTION END
# =====================================================================


# =====================================================================
# FIXED — Evaluation and CLI
# =====================================================================

def evaluate_zero_shot(args):
    """Zero-shot evaluation loop — no training, just score all rows."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata
    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)

    if args.antigen_key not in metadata:
        raise KeyError(
            f"antigen_key={args.antigen_key!r} not in metadata. "
            f"Available: {list(metadata.keys())}")
    meta = metadata[args.antigen_key]
    pdb_path = meta["pdb_path"].replace("./data", args.data_dir)
    heavy_chain_id = meta["heavy_chain"]
    light_chain_id = meta["light_chain"]
    antigen_chain_ids = meta["antigen_chains"]

    # Extract wildtype sequences from PDB
    wt_heavy, wt_light, wt_antigen = extract_sequences_from_pdb(
        pdb_path, heavy_chain_id, light_chain_id, antigen_chain_ids)
    print(f"WT sequences: heavy={len(wt_heavy)}aa, light={len(wt_light)}aa, "
          f"antigen={len(wt_antigen)}aa")
    # Sanity print of antigen prefix and chain order — lets us check
    # alignment against the canonical antigen sequence (e.g. for
    # 4d5_her2 the antigen should start with HER2 ECD residues from
    # UniProt P04626; for 3gbn / 4fqi it should start with influenza
    # hemagglutinin H1).
    print(f"WT antigen chains (metadata order): {antigen_chain_ids}")
    print(f"WT antigen seq[:60]: {wt_antigen[:60]}")
    if wt_heavy:
        print(f"WT heavy seq[:60]:   {wt_heavy[:60]}")

    # Resolve CSV
    csv_paths = [p.replace("./data", args.data_dir)
                 for p in meta["affinity_data"]]
    target_csv = None
    for p in csv_paths:
        if args.dataset_label.lower() in os.path.basename(p).lower():
            target_csv = p
            break
    if target_csv is None:
        target_csv = csv_paths[0]
    print(f"Loading data from: {target_csv}")
    df = pd.read_csv(target_csv)
    print(f"Loaded {len(df)} measurements")

    hc_col = "heavy_chain_seq" if "heavy_chain_seq" in df.columns else "mut_heavy_chain_seq"
    lc_col = "light_chain_seq" if "light_chain_seq" in df.columns else None

    heavy_seqs: List[str] = []
    light_seqs: List[str] = []
    targets: List[float] = []
    for _, row in df.iterrows():
        hc = row.get(hc_col, wt_heavy)
        if not isinstance(hc, str) or len(hc) == 0:
            hc = wt_heavy
        lc = wt_light
        if lc_col and lc_col in df.columns:
            lc_val = row.get(lc_col, wt_light)
            if isinstance(lc_val, str) and len(lc_val) > 0:
                lc = lc_val
        bs = row.get("binding_score", None)
        if bs is None or pd.isna(bs):
            continue
        heavy_seqs.append(hc)
        light_seqs.append(lc)
        targets.append(float(bs))

    print(f"Valid rows: {len(targets)}")

    # Optional subsample (used for 4fqi_h1 which has ~65k rows — too many
    # for masked-marginal style scoring inside the per-split 24h budget).
    if args.max_rows > 0 and len(targets) > args.max_rows:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(targets), size=args.max_rows, replace=False)
        idx.sort()
        heavy_seqs = [heavy_seqs[i] for i in idx]
        light_seqs = [light_seqs[i] for i in idx]
        targets = [targets[i] for i in idx]
        print(f"Subsampled to {len(targets)} rows (seed={args.seed})")

    # Load frozen ESM-2 MLM
    print(f"Loading ESM-2 model: {ESM_MODEL_NAME}")
    esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm_model = AutoModelForMaskedLM.from_pretrained(ESM_MODEL_NAME)
    esm_model.eval()
    esm_model.to(device)
    for p in esm_model.parameters():
        p.requires_grad = False
    print(f"ESM-2 loaded ({sum(p.numel() for p in esm_model.parameters()):,} params, frozen)")

    # Build scoring function
    scorer = ScoringFunction(esm_model, esm_tokenizer, device)

    # Score in mini-batches (the model forward is per-sequence so we just
    # iterate; mini-batching is offered as a hint for the agent).
    all_scores: List[float] = []
    bs = max(1, args.batch_size)
    n = len(heavy_seqs)
    t0 = None
    import time
    t0 = time.time()
    for start in range(0, n, bs):
        end = min(start + bs, n)
        part_scores = scorer.score_batch(
            wt_heavy, wt_light, wt_antigen,
            heavy_seqs[start:end], light_seqs[start:end])
        all_scores.extend(part_scores)
        if (start // bs) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, end) * (n - end)
            print(f"TRAIN_METRICS scored={end}/{n} elapsed={elapsed:.1f}s "
                  f"eta={eta:.1f}s", flush=True)

    scores_arr = np.array(all_scores, dtype=np.float64)
    targets_arr = np.array(targets, dtype=np.float64)

    valid = np.isfinite(scores_arr) & np.isfinite(targets_arr)
    if valid.sum() < 5 or len(np.unique(scores_arr[valid])) < 2:
        spear = 0.0
    else:
        spear, _ = spearmanr(scores_arr[valid], targets_arr[valid])
        if np.isnan(spear):
            spear = 0.0

    print(f"TEST_METRICS spearman={spear:.6f} n={int(valid.sum())}",
          flush=True)

    # Optional dump of per-variant scores for debugging.
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(
        args.output_dir, f"scores_{args.antigen_key}.csv")
    pd.DataFrame({
        "binding_score": targets_arr,
        "predicted_score": scores_arr,
    }).to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Antibody Binding Affinity Zero-Shot Scoring")
    parser.add_argument("--antigen-key", type=str, required=True,
                        help="Key in metadata.json (e.g. 3gbn, 4fqi, 4d5_her2)")
    parser.add_argument("--dataset-label", type=str, required=True,
                        help="Dataset label for CSV selection (e.g. 3gbn_h1)")
    parser.add_argument("--data-dir", type=str, default="/data",
                        help="Root data directory")
    parser.add_argument("--metadata-file", type=str,
                        default="/data/metadata.json",
                        help="Path to metadata.json")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Hint for the scorer; ignored by the harness")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Subsample to this many rows (0=no subsample)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    evaluate_zero_shot(args)


if __name__ == "__main__":
    main()

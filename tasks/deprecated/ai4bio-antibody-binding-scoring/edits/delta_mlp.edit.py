"""wildtype_delta baseline — per-position mutant-context WT-relative scoring.

This is the "ESM-PSSM" / "wildtype-marginal" zero-shot scoring from
Brandes et al., 2023 (Genome-wide prediction of disease variant effects
with a deep protein language model, Nature Genetics,
https://doi.org/10.1038/s41588-023-01465-0, "ESM1b LLR" formulation,
also used as the unmasked-context "wildtype marginal" in Meier et al.,
2021 §2.2, https://doi.org/10.1101/2021.07.09.450648):

For each mutated heavy-chain position p:
    s_p = log P(mut_aa_p | mutant complex sequence, NO masking)
        - log P(wt_aa_p  | mutant complex sequence, NO masking)
score(variant) = sum over mutated p of s_p

Both terms are read from a SINGLE unmasked forward pass over the mutant
complex (one forward per variant). This is mathematically distinct from
both:
  * `esm2_pll`: averages log P over ALL positions of one unmasked
    forward — dominated by the (~99% of) unchanged residues, so the
    WT-vs-mut signal is heavily diluted.
  * `masked_marginal`: builds a per-position MASKED forward over the
    WT sequence and reads logP(mut)-logP(wt) from the WT-masked
    distribution (Meier 2021 Eq. 2).

The "wildtype-marginal" / "ESM-PSSM" variant kept here uses a single
UNMASKED forward over the mutant — same forward cost as `esm2_pll` but
restricted to the mutated positions and WT-relative, so it is the
natural cheap third baseline alongside the more expensive masked-
marginal.
"""

_FILE = "AbBiBench/custom_abscore.py"

_CONTENT = """\

class ScoringFunction:
    \"\"\"Wildtype-marginal scoring (Meier et al., 2021 §2.2 / Brandes
    et al., 2023): one unmasked forward over the mutant complex,
    sum over mutated heavy-chain positions of
        log P(mut_aa | mutant_ctx) - log P(wt_aa | mutant_ctx).
    Mathematically distinct from `esm2_pll` (which averages over ALL
    positions) and `masked_marginal` (which masks the WT context).\"\"\"

    def __init__(self, esm_model, esm_tokenizer, device):
        self.esm_model = esm_model
        self.esm_tokenizer = esm_tokenizer
        self.device = device

    @torch.no_grad()
    def _per_pos_logprobs(self, sequence: str):
        \"\"\"Return per-residue log-probs (after stripping BOS/EOS) and
        the tokenizer-vocab map.\"\"\"
        logprobs, _ = esm_forward_logprobs(
            self.esm_model, self.esm_tokenizer, sequence, self.device,
            max_len=MAX_SEQ_LEN)
        return logprobs  # [L, V]

    def score_batch(self, wt_heavy: str, wt_light: str, wt_antigen: str,
                    mut_heavy_seqs, mut_light_seqs):
        vocab = self.esm_tokenizer.get_vocab()
        scores = []
        for mh, ml in zip(mut_heavy_seqs, mut_light_seqs):
            mut_concat = (mh + ml + wt_antigen)[:MAX_SEQ_LEN]
            logp = self._per_pos_logprobs(mut_concat)  # [L, V]
            if len(mh) != len(wt_heavy):
                # Length mismatch — cannot align WT vs mut residues by
                # index. Fall back to mean-PLL so the row still gets a
                # finite score.
                s = float(logp.gather(-1,
                    self.esm_tokenizer(mut_concat, return_tensors='pt',
                        add_special_tokens=True, truncation=True,
                        max_length=MAX_SEQ_LEN+2)['input_ids'][0,1:-1]
                        .to(self.device).unsqueeze(-1)).squeeze(-1).mean())
                scores.append(s)
                continue
            total = 0.0
            n_used = 0
            n_heavy_in_logp = min(len(wt_heavy), logp.shape[0])
            for p in range(n_heavy_in_logp):
                if mh[p] == wt_heavy[p]:
                    continue
                wt_id = vocab.get(wt_heavy[p], None)
                mu_id = vocab.get(mh[p], None)
                if wt_id is None or mu_id is None:
                    continue
                # Both terms read from the SAME mutant-context unmasked
                # forward at position p. This is the "wildtype-marginal"
                # of Meier 2021 §2.2.
                total += float(logp[p, mu_id] - logp[p, wt_id])
                n_used += 1
            if n_used == 0:
                scores.append(0.0)
            else:
                scores.append(total)
        return scores

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 133,
        "end_line": 167,
        "content": _CONTENT,
    },
]

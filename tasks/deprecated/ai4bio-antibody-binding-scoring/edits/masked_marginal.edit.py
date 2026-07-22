"""masked_marginal baseline — Meier et al., 2021 masked-marginal scoring.

Strict implementation of Meier et al., 2021, Eq. 2
(https://doi.org/10.1101/2021.07.09.450648):

    score(variant) = sum over mutated positions p of
        [ log P(mut_aa_p | wt_seq with position p masked)
         - log P(wt_aa_p  | wt_seq with position p masked) ]

Both probabilities are read from the SAME masked forward pass — the WT
sequence with position p replaced by <mask>. This means every variant's
score reuses the per-position WT masked logprobs, so we cache them once
per dataset (one masked forward per WT heavy-chain position) and then
score each variant in O(K) lookups where K = number of mutated heavy-
chain positions.

Mutations are taken on the heavy chain only (AbBiBench's heavy-chain DMS
convention, see `vendor/external_packages/AbBiBench/models/ESM-2/get_model_log_likelihood.py`
which scores the concatenated complex; we restrict the masked-marginal
sum to heavy-chain residues, the chain that varies across the DMS).

Cost: one masked forward per WT heavy-chain position, total ~|wt_heavy|
forwards per dataset (~120 for IgG VH). For 4fqi_h1 with up to 5000
variants × ~120 WT positions, this is ~120 forwards total (cached),
which is far cheaper than O(K) forwards per variant.
"""

_FILE = "AbBiBench/custom_abscore.py"

_CONTENT = """\

class ScoringFunction:
    \"\"\"Masked-marginal scoring (Meier et al., 2021, Eq. 2).

    Caches per-position masked WT logprobs once per (wt_heavy, wt_light,
    wt_antigen) tuple, then scores each variant by a sum of
    logP(mut_aa)-logP(wt_aa) at mutated heavy-chain positions, all read
    from the cached WT-masked logprobs.
    \"\"\"

    def __init__(self, esm_model, esm_tokenizer, device):
        self.esm_model = esm_model
        self.esm_tokenizer = esm_tokenizer
        self.device = device
        self._wt_masked_cache = {}  # key: (wt_heavy, wt_light, wt_antigen) -> dict[int, Tensor[V]]
        self._wt_concat_cache = {}  # key: same -> wt full complex string (truncated)

    @torch.no_grad()
    def _build_wt_masked_logprobs(self, wt_heavy, wt_light, wt_antigen):
        key = (wt_heavy, wt_light, wt_antigen)
        if key in self._wt_masked_cache:
            return self._wt_masked_cache[key], self._wt_concat_cache[key]
        wt_concat = (wt_heavy + wt_light + wt_antigen)[:MAX_SEQ_LEN]
        # Tokenize once to get base ids; we will mask one heavy position
        # at a time and rerun the model.
        enc = self.esm_tokenizer(wt_concat, return_tensors='pt',
                                 add_special_tokens=True, truncation=True,
                                 max_length=MAX_SEQ_LEN + 2)
        base_ids = enc['input_ids'][0].to(self.device)
        attn = enc['attention_mask'][0].to(self.device)
        mask_id = self.esm_tokenizer.mask_token_id
        cache = {}
        # Heavy chain occupies positions 0 .. len(wt_heavy)-1 of wt_concat,
        # i.e. tokens 1 .. len(wt_heavy) of base_ids (BOS shifts +1).
        n_heavy = min(len(wt_heavy), MAX_SEQ_LEN)
        for p in range(n_heavy):
            token_idx = p + 1  # +1 for BOS
            if token_idx >= base_ids.numel() - 1:
                break
            ids = base_ids.clone()
            ids[token_idx] = mask_id
            out = self.esm_model(input_ids=ids.unsqueeze(0),
                                 attention_mask=attn.unsqueeze(0))
            logp = torch.log_softmax(out.logits[0, token_idx], dim=-1).cpu()
            cache[p] = logp
        self._wt_masked_cache[key] = cache
        self._wt_concat_cache[key] = wt_concat
        return cache, wt_concat

    def score_batch(self, wt_heavy: str, wt_light: str, wt_antigen: str,
                    mut_heavy_seqs, mut_light_seqs):
        wt_logp, _ = self._build_wt_masked_logprobs(
            wt_heavy, wt_light, wt_antigen)
        vocab = self.esm_tokenizer.get_vocab()
        scores = []
        for mh, ml in zip(mut_heavy_seqs, mut_light_seqs):
            if len(mh) != len(wt_heavy):
                # Length mismatch — cannot align positions for masked
                # marginal. Fall back to mean-PLL of mutant complex so
                # the row gets a finite score (will simply be ranked
                # together with other length-mismatched rows).
                s = esm_mean_token_logprob(
                    self.esm_model, self.esm_tokenizer,
                    mh + ml + wt_antigen, self.device)
                scores.append(s)
                continue
            total = 0.0
            n_used = 0
            for p in range(len(wt_heavy)):
                if mh[p] == wt_heavy[p]:
                    continue
                if p not in wt_logp:
                    continue  # past truncation cap
                wt_id = vocab.get(wt_heavy[p], None)
                mu_id = vocab.get(mh[p], None)
                if wt_id is None or mu_id is None:
                    continue
                lp = wt_logp[p]
                total += float(lp[mu_id] - lp[wt_id])
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

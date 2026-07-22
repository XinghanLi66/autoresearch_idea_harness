"""Multipack with Document Splitting baseline.

Combines the multipack sampler's length-aware batch construction with
document splitting to eliminate token waste from cropping. Instead of
discarding tokens when a document doesn't fit, it splits the document
at the row boundary and carries the remainder to a subsequent row
(re-prepending a BOS token to maintain the BOS-aligned invariant).

This approach is inspired by:
  1. The multipack distributed sampler (imoneoi/multipack_sampler) which
     optimizes packing efficiency by length-aware assignment.
  2. Dataset Decomposition (Kaddour et al., NeurIPS 2024) which uses
     variable-length sequences with length-aware scheduling.
  3. The "greedy concatenation with splitting" approach used in GPT-3
     and Megatron-LM pretraining, where documents are concatenated and
     split at context boundaries rather than cropped.

Key insight: splitting preserves ALL tokens (0% crop rate) at the cost
of some documents losing long-range coherence across the split boundary.
The BOS token re-insertion ensures each row fragment is properly delimited.

Reference:
  imoneoi (2023). multipack_sampler: Multipack distributed sampler for
    fast padding-free training of LLMs. github.com/imoneoi/multipack_sampler.
  Kaddour, J. et al. (2024). Dataset Decomposition: Faster LLM Training
    with Variable Sequence Length Curriculum. NeurIPS 2024.
"""

_FILE = "nanochat/nanochat/custom_dataloader.py"

_MULTIPACK_SPLIT = """\
def pack_rows(B, row_capacity, doc_buffer, refill_buffer, row_buffer):
    \"\"\"Multipack with document splitting: zero-crop packing.

    Length-aware packing that splits documents across row boundaries instead
    of cropping them. Remainders are re-queued with a BOS token prepended.
    Achieves 0% token waste at the cost of splitting some documents.

    References:
      - imoneoi/multipack_sampler (2023)
      - Kaddour et al. (NeurIPS 2024), Dataset Decomposition
    \"\"\"
    import torch
    buffer_size = 1000
    tokens_used = 0
    tokens_cropped = 0  # should be 0 with splitting

    # Track remaining capacity and position per row
    remaining = [row_capacity] * B
    positions = [0] * B

    # Phase 1: Length-aware assignment -- sort buffer, assign longest docs
    # to emptiest rows (similar to multipack's load-balancing)
    while len(doc_buffer) < buffer_size:
        refill_buffer()

    # Sort docs by length descending for better packing decisions
    doc_buffer.sort(key=len, reverse=True)

    # Assign docs to rows using a length-aware greedy strategy:
    # For each doc (longest first), place it in the row with the most space
    placed_indices = []
    for i in range(len(doc_buffer)):
        doc = doc_buffer[i]
        doc_len = len(doc)

        # Find row with most remaining space
        best_row = max(range(B), key=lambda r: remaining[r])

        if remaining[best_row] <= 0:
            break  # all rows full

        if doc_len <= remaining[best_row]:
            # Document fits entirely
            pos = positions[best_row]
            row_buffer[best_row, pos:pos + doc_len] = torch.tensor(
                doc, dtype=torch.long
            )
            positions[best_row] += doc_len
            remaining[best_row] -= doc_len
            tokens_used += doc_len
            placed_indices.append(i)
        else:
            # Document too long: SPLIT instead of crop
            space = remaining[best_row]
            if space > 0:
                pos = positions[best_row]
                row_buffer[best_row, pos:pos + space] = torch.tensor(
                    doc[:space], dtype=torch.long
                )
                positions[best_row] += space
                remaining[best_row] = 0
                tokens_used += space

                # Re-queue remainder with BOS prepended (BOS = first token of
                # original doc, which was prepended during tokenization)
                bos_token = doc[0]  # BOS is always the first token
                remainder = [bos_token] + doc[space:]
                doc_buffer[i] = remainder  # replace in-place, don't mark as placed
            else:
                break

    # Remove placed docs from buffer (reverse order to preserve indices)
    for idx in sorted(placed_indices, reverse=True):
        doc_buffer.pop(idx)

    # Phase 2: Fill any remaining gaps
    for row_idx in range(B):
        while remaining[row_idx] > 0:
            while len(doc_buffer) < 1:
                refill_buffer()

            space = remaining[row_idx]

            # Try to find a doc that fits
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(doc_buffer):
                dl = len(doc)
                if dl <= space and dl > best_len:
                    best_idx = i
                    best_len = dl

            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                doc_len = len(doc)
                pos = positions[row_idx]
                row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(
                    doc, dtype=torch.long
                )
                positions[row_idx] += doc_len
                remaining[row_idx] -= doc_len
                tokens_used += doc_len
            else:
                # Split the shortest doc to fill remaining
                shortest_idx = min(
                    range(len(doc_buffer)), key=lambda j: len(doc_buffer[j])
                )
                doc = doc_buffer.pop(shortest_idx)
                pos = positions[row_idx]
                row_buffer[row_idx, pos:pos + space] = torch.tensor(
                    doc[:space], dtype=torch.long
                )
                tokens_used += space
                positions[row_idx] += space
                remaining[row_idx] = 0

                # Re-queue remainder with BOS
                if len(doc) > space:
                    bos_token = doc[0]
                    remainder = [bos_token] + doc[space:]
                    doc_buffer.insert(0, remainder)
                # tokens_cropped stays 0 -- we split, not crop

    packing_stats = {
        'tokens_used': tokens_used,
        'tokens_cropped': tokens_cropped,
        'tokens_total': B * row_capacity,
    }
    return packing_stats
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 69,
        "end_line": 140,
        "content": _MULTIPACK_SPLIT,
    },
]

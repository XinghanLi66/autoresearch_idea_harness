"""First-Fit Decreasing (FFD) packing baseline.

Classic bin-packing heuristic adapted for LLM sequence packing.
Sorts documents by length (longest first), then places each document
into the first row with sufficient remaining capacity. When no document
fits any row, the shortest document is cropped to fill the emptiest row.

This is the standard offline bin-packing heuristic with asymptotic ratio
11/9 * OPT + 6/9 (Johnson 1973). It reduces fragmentation compared to
greedy/online methods by considering all rows jointly and prioritizing
placement of large items first.

Reference:
  Johnson, D.S. (1973). Near-optimal bin packing algorithms.
  Coffman, E.G., Garey, M.R., Johnson, D.S. (1996).
    Approximation algorithms for bin packing: a survey.
"""

_FILE = "nanochat/nanochat/custom_dataloader.py"

_FFD = """\
def pack_rows(B, row_capacity, doc_buffer, refill_buffer, row_buffer):
    \"\"\"First-Fit Decreasing (FFD) packing.

    Classic bin-packing: sort documents longest-first, place each into the
    first row where it fits. Jointly considers all B rows to reduce
    fragmentation.

    Reference: Johnson (1973), Near-optimal bin packing algorithms.
    \"\"\"
    import torch
    buffer_size = 1000
    tokens_used = 0
    tokens_cropped = 0

    # Track remaining capacity and write position per row
    remaining = [row_capacity] * B
    positions = [0] * B

    # We process in rounds: fill buffer, sort, place as many as possible
    max_rounds = B * 20  # safety limit
    for _ in range(max_rounds):
        if all(r == 0 for r in remaining):
            break

        # Ensure buffer is well-stocked
        while len(doc_buffer) < buffer_size:
            refill_buffer()

        # Sort buffer descending by length (FFD core step)
        doc_buffer.sort(key=len, reverse=True)

        placed_any = False
        i = 0
        while i < len(doc_buffer):
            doc = doc_buffer[i]
            doc_len = len(doc)

            # First-fit: find first row where doc fits entirely
            placed = False
            for row_idx in range(B):
                if doc_len <= remaining[row_idx]:
                    pos = positions[row_idx]
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(
                        doc, dtype=torch.long
                    )
                    positions[row_idx] += doc_len
                    remaining[row_idx] -= doc_len
                    tokens_used += doc_len
                    doc_buffer.pop(i)
                    placed = True
                    placed_any = True
                    break
            if not placed:
                i += 1

        if not placed_any:
            # No whole document fits any row -- crop to fill remaining spaces
            for row_idx in range(B):
                if remaining[row_idx] > 0:
                    while len(doc_buffer) < 1:
                        refill_buffer()
                    # Pick shortest doc to minimize wasted tokens
                    shortest_idx = min(
                        range(len(doc_buffer)), key=lambda j: len(doc_buffer[j])
                    )
                    doc = doc_buffer.pop(shortest_idx)
                    space = remaining[row_idx]
                    pos = positions[row_idx]
                    row_buffer[row_idx, pos:pos + space] = torch.tensor(
                        doc[:space], dtype=torch.long
                    )
                    tokens_cropped += len(doc) - space
                    tokens_used += space
                    positions[row_idx] += space
                    remaining[row_idx] = 0
            break

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
        "content": _FFD,
    },
]

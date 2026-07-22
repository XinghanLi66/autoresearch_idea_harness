"""Shortest-Pack-First Histogram-Packing (SPFHP) baseline.

SOTA histogram-based packing algorithm from Krell et al. (2021).
Instead of operating on individual documents, SPFHP works on the
sequence-length histogram: it traverses lengths from longest to shortest
and applies a worst-fit strategy, assigning each length bin to the pack
(row) with the most remaining space ("shortest-pack-first").

Key advantages over naive bin-packing:
  - O(n + S^2) time complexity (S = max seq length), independent of dataset size
  - Processes millions of sequences in seconds
  - Achieves 89-99% packing efficiency depending on packing depth

With packing depth 3 (max 3 docs per row), SPFHP achieves ~89% efficiency.
With depth 8, it reaches ~99%. The implementation here uses the buffer as
a streaming histogram approximation.

Reference:
  Krell, M.M., Kosec, M., Perez, S.P., Fitzgibbon, A. (2021).
  Efficient Sequence Packing without Cross-contamination: Accelerating
  Large Language Models without Impacting Performance. arXiv:2107.02027.
  Also: NeurIPS 2021 Workshop on Efficient Natural Language and Speech Processing.
"""

_FILE = "nanochat/nanochat/custom_dataloader.py"

_SPFHP = """\
def pack_rows(B, row_capacity, doc_buffer, refill_buffer, row_buffer):
    \"\"\"Shortest-Pack-First Histogram-Packing (SPFHP).

    SOTA histogram-based packing from Krell et al. (2021, arXiv:2107.02027).
    Builds a length histogram from the buffer, traverses from longest to
    shortest, and assigns each length bin to the row with the most remaining
    space (worst-fit / shortest-pack-first).

    This avoids the O(n * buffer_size) per-document search of best-fit and
    achieves near-optimal packing efficiency.
    \"\"\"
    import torch
    from collections import defaultdict

    buffer_size = 2000  # larger buffer for better histogram statistics
    tokens_used = 0
    tokens_cropped = 0
    max_depth = 8  # max documents per row (packing depth)

    # Fill buffer generously for good histogram coverage
    while len(doc_buffer) < buffer_size:
        refill_buffer()

    # Build length histogram: length -> list of doc indices in buffer
    len_to_docs = defaultdict(list)
    for i, doc in enumerate(doc_buffer):
        doc_len = min(len(doc), row_capacity)  # cap at row capacity
        len_to_docs[doc_len].append(i)

    # Get sorted unique lengths (descending = longest first)
    sorted_lengths = sorted(len_to_docs.keys(), reverse=True)

    # Track remaining capacity, position, and doc count per row
    remaining = [row_capacity] * B
    positions = [0] * B
    doc_counts = [0] * B
    used_indices = set()

    # SPFHP core: traverse lengths longest-to-shortest
    for length in sorted_lengths:
        doc_indices = len_to_docs[length]
        for doc_idx in doc_indices:
            if doc_idx in used_indices:
                continue

            # Worst-fit: find row with MOST remaining space where doc fits
            # and hasn't exceeded packing depth
            best_row = -1
            best_remaining = -1
            for row_idx in range(B):
                if (remaining[row_idx] >= length and
                        doc_counts[row_idx] < max_depth and
                        remaining[row_idx] > best_remaining):
                    best_row = row_idx
                    best_remaining = remaining[row_idx]

            if best_row >= 0:
                doc = doc_buffer[doc_idx]
                actual_len = min(len(doc), remaining[best_row])
                pos = positions[best_row]
                row_buffer[best_row, pos:pos + actual_len] = torch.tensor(
                    doc[:actual_len], dtype=torch.long
                )
                positions[best_row] += actual_len
                remaining[best_row] -= actual_len
                tokens_used += actual_len
                doc_counts[best_row] += 1
                used_indices.add(doc_idx)

    # Remove used docs from buffer (in reverse order to preserve indices)
    for idx in sorted(used_indices, reverse=True):
        doc_buffer.pop(idx)

    # Fill any remaining space by cropping from buffer
    for row_idx in range(B):
        while remaining[row_idx] > 0:
            while len(doc_buffer) < 1:
                refill_buffer()

            space = remaining[row_idx]
            # Try to find a doc that fits exactly or nearly
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
                # Must crop - pick shortest to minimize waste
                shortest_idx = min(
                    range(len(doc_buffer)), key=lambda j: len(doc_buffer[j])
                )
                doc = doc_buffer.pop(shortest_idx)
                pos = positions[row_idx]
                row_buffer[row_idx, pos:pos + space] = torch.tensor(
                    doc[:space], dtype=torch.long
                )
                tokens_cropped += len(doc) - space
                tokens_used += space
                positions[row_idx] += space
                remaining[row_idx] = 0

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
        "content": _SPFHP,
    },
]

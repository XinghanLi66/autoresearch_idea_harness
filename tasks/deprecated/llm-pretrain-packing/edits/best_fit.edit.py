"""Vanilla Best-Fit baseline — rigorous codebase edit ops.

No-op: the template already implements BOS-aligned best-fit packing with
cropping, so this baseline leaves the editable region unchanged.

This is the default algorithm: for each row, find the largest document that
fits, repeat until no document fits, then crop the shortest to fill. Achieves
100% utilization but ~35% token crop rate.
"""

_FILE = "nanochat/nanochat/custom_dataloader.py"

_BEST_FIT = """\
def pack_rows(B, row_capacity, doc_buffer, refill_buffer, row_buffer):
    \"\"\"Vanilla best-fit packing with cropping.

    For each row: find largest doc that fits entirely, repeat, then crop
    shortest doc to fill remaining space. 100% utilization, ~35% crop rate.
    \"\"\"
    import torch
    buffer_size = 1000
    tokens_used = 0
    tokens_cropped = 0

    for row_idx in range(B):
        pos = 0
        while pos < row_capacity:
            # Ensure buffer has documents
            while len(doc_buffer) < buffer_size:
                refill_buffer()

            remaining = row_capacity - pos

            # Find largest doc that fits entirely
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = i
                    best_len = doc_len

            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                doc_len = len(doc)
                row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                pos += doc_len
                tokens_used += doc_len
            else:
                # No doc fits - crop shortest in buffer to fill remaining
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                tokens_cropped += len(doc) - remaining
                row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                pos += remaining
                tokens_used += remaining

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
        "content": _BEST_FIT,
    },
]

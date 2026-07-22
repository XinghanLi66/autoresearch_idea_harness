"""Custom Distributed Dataloader for Pretraining — Sequence Packing.

Based on nanochat's BOS-aligned best-fit packing dataloader.
The packing algorithm is extracted into `pack_rows()` for modification.

Original: ~35% of tokens are cropped at T=2048. Improve packing efficiency
and/or training signal quality by modifying the packing strategy.
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files


def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


# ============================================================================
# EDITABLE REGION START — Packing Algorithm
# ============================================================================
def pack_rows(B, row_capacity, doc_buffer, refill_buffer, row_buffer):
    """Pack B rows of fixed length `row_capacity` from documents in `doc_buffer`.

    This is the core packing algorithm. Each row starts with a BOS token (already
    prepended to each document during tokenization). The goal is to arrange
    variable-length tokenized documents into fixed-size rows while minimizing
    token waste from cropping.

    Args:
        B: Number of rows (batch size).
        row_capacity: Number of tokens per row (T + 1, to get T input and T target tokens).
        doc_buffer: List of tokenized documents (list of lists of ints). Documents are
            consumed (popped) from this buffer. Call `refill_buffer()` to add more.
        refill_buffer: Callable that adds more tokenized documents to `doc_buffer`.
        row_buffer: Pre-allocated torch tensor of shape (B, row_capacity) to fill.

    Returns:
        packing_stats: dict with optional statistics, e.g.:
            - 'tokens_used': total tokens placed from complete documents
            - 'tokens_cropped': total tokens discarded due to cropping
            - 'tokens_total': total token capacity across all rows (B * row_capacity)

    Current algorithm (BOS-aligned best-fit with cropping):
        For each row:
        1. Search buffer for the LARGEST document that fits entirely in remaining space
        2. Place it and repeat until no document fits
        3. When no document fits, crop the SHORTEST document to fill remaining space exactly
        This achieves 100% utilization (no padding) but ~35% of tokens are cropped.
    """
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
                # No doc fits - crop shortest in buffer to fill remaining and minimize waste
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
# ============================================================================
# EDITABLE REGION END
# ============================================================================


def custom_tokenizing_distributed_data_loader_with_state_bos(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    Custom BOS-aligned dataloader using the editable pack_rows() function.

    This dataloader delegates the core packing logic to pack_rows(), allowing
    experimentation with different packing strategies while keeping the rest
    of the data pipeline (tokenization, DDP sharding, GPU transfer) intact.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        packing_stats = pack_rows(B, row_capacity, doc_buffer, refill_buffer, row_buffer)

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {
            "pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch,
            "packing_stats": packing_stats,
        }

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def custom_tokenizing_distributed_data_loader_bos(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in custom_tokenizing_distributed_data_loader_with_state_bos(*args, **kwargs):
        yield inputs, targets

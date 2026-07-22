# LLM Pretraining: Sequence Packing Strategy

## Research Question
Design an improved sequence packing algorithm for LLM pretraining data. The packing algorithm arranges variable-length tokenized documents into fixed-size training sequences, directly affecting both data efficiency and training signal quality.

## Background
Modern LLM pretraining uses BOS-aligned packing: each training row starts with a BOS token, and multiple documents are packed into a single fixed-length sequence. The current best-fit algorithm searches a buffer of tokenized documents for the largest document that fits the remaining space, repeating until no document fits, then crops a document to fill the remainder exactly.

This achieves 100% utilization (no padding tokens), but approximately **35% of all tokens are discarded** due to cropping at sequence length T=2048. Cropped tokens are wasted compute -- they were tokenized but never trained on. Additionally, cropped documents lose their ending context, potentially degrading the training signal.

Sequence packing is related to classical bin packing. Relevant design choices include how documents are assigned to rows, when long documents are split instead of cropped, and whether packing decisions are made per row or jointly across multiple rows.

## What You Can Modify
The `pack_rows()` function (lines 69-140) in `custom_dataloader.py`:
- Document selection strategy (which document to place next)
- Cropping policy (when and how to crop documents)
- Buffer management (how to organize the document buffer)
- Row construction order (how to fill multiple rows jointly)
- Document splitting (splitting long documents across rows vs. cropping)
- Any combination of the above

**Note**: The function signature `pack_rows(B, row_capacity, doc_buffer, refill_buffer, row_buffer)` must be preserved. The function must fill `row_buffer` (shape `[B, row_capacity]`) with tokens and return a `packing_stats` dict. Each row should start with a BOS token (already prepended to documents during tokenization).

## Evaluation
- **Primary metric**: Validation BPB (bits per byte, lower is better) -- measures model quality
- **Secondary metric**: Packing utilization = tokens_used / (tokens_used + tokens_cropped) -- measures data efficiency
- **Model**: nanochat depth-4 (small model for fast iteration)
- **Dataset**: ClimbMix (parquet format, pre-tokenized)
- **Sequence length**: T=2048

# LLM Pretraining: Attention Mechanism Optimization

## Research Question
Design an improved self-attention mechanism for GPT-2 language model pretraining. Your modifications should reduce validation loss compared to the standard multi-head attention with learned absolute position embeddings.

## What You Can Modify
The `CausalSelfAttention` class (lines 34-70 in `custom_pretrain.py`), including:
- Position encoding scheme (the default uses learned absolute position embeddings via `wpe`)
- Query/Key/Value computation and projection
- Attention score computation and masking
- Any attention-related hyperparameters

**Note**: If your attention mechanism implements its own position encoding (replacing the learned `wpe`), set `self.use_pos_emb = False` in `__init__` — the model will then skip adding position embeddings in the forward pass.

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better), plus perplexity (WikiText-2, LAMBADA) and downstream accuracy (ARC-Easy, HellaSwag, PIQA, WinoGrande)
- **Model**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Dataset**: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N Chinchilla-optimal)
- **Training**: 13535 iterations, BSZ=64, GA=8, 2-GPU DDP
- **Hardware**: H200 GPU


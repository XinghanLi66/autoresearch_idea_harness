# LLM Test-Time Training / Adaptation

## Research Question
Design a test-time training (TTT) strategy that improves GPT-2 Medium language modeling performance by adapting model weights or adding learned memory mechanisms at inference time. The model uses real GPT-2 Medium weights from HuggingFace, and your TTT adapter runs on the evaluation context to reduce validation loss.

## Background
Standard language models use fixed weights at inference time. Test-time training (TTT) adapts the model at inference to better fit the local data distribution. Recent advances include:
- **TTT layers** (Sun et al., 2024): Replace attention with a self-supervised inner loop that updates hidden states
- **Titan** (Behrouz et al., 2025): Neural long-term memory with surprise-based gating that learns to memorize at test time
- **LoRA adaptation**: Lightweight low-rank updates to pretrained weights using eval context
- **Parameter golf**: Competitions showing TTT can reduce BPB by 0.01-0.03 on language modeling

The key challenge is designing an adaptation strategy that is both effective (reduces loss) and efficient (runs within a reasonable time budget).

## What You Can Modify
The `TTTAdapter` class in `custom_ttt_eval.py` (lines 276-346):
- `__init__()`: Initialize adapter hyperparameters and state
- `setup(model, config)`: Set up TTT-specific state after the pretrained model is loaded
- `adapt_and_evaluate(model, eval_chunks, ctx)`: The core TTT logic

You can implement any of:
- **Weight adaptation**: Fine-tune a subset of model parameters (e.g., LoRA, bias-only, layer norm) on eval context using self-supervised loss
- **Learned memory**: Add memory modules (Titan-style) that store and retrieve information from context
- **Hybrid approaches**: Combine adaptation mechanisms with memory-augmented inference
- **Novel TTT objectives**: Design auxiliary losses beyond standard next-token prediction for adaptation

## Architecture
The task loads real GPT-2 Medium weights (downloaded from HuggingFace) and converts them to nanoGPT format. No pretraining is done -- the task is purely about TTT adaptation quality.

The script (`custom_ttt_eval.py`):
1. Loads GPT-2 Medium weights from `/data/gpt2-medium` (pre-downloaded HuggingFace snapshot)
2. Converts HF weights to nanoGPT format (handles Conv1D -> Linear transposition)
3. Runs your `TTTAdapter` for adaptation + evaluation
4. Reports validation loss and benchmark PPL

## Interface
```python
class TTTAdapter:
    def __init__(self):
        # Set hyperparameters (learning rate, steps, etc.)
        pass

    def setup(self, model, config):
        # Called once after loading the pretrained model with the GPTConfig
        pass

    def adapt_and_evaluate(self, model, eval_chunks, ctx):
        # model: pretrained GPT (nn.Module, eval mode)
        # eval_chunks: list of (x, y) pairs, shape (1, 1024) each
        # ctx: torch.amp.autocast context for mixed precision
        # Returns: float (average cross-entropy loss)
        pass
```

**Constraints**:
- Additional parameters must be < 5% of the base model
- Always work on a copy of the model (use `copy.deepcopy`)
- Must return a valid float loss value
- `copy`, `math`, `torch`, `torch.nn`, `F`, `np`, `os`, `time` are available

## Evaluation
- **Metric**: Validation loss (cross-entropy, lower is better) after TTT adaptation
- **Model size**: GPT-2 Medium (24L/16H/1024D, ~355M params)
- **Weights**: Real GPT-2 Medium from HuggingFace (no pretraining needed)
- **Dataset**: FineWeb (GPT-2 tokenizer) for evaluation data

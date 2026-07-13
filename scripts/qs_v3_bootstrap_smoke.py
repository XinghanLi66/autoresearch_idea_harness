#!/usr/bin/env python3
"""Run a self-contained V3 training bootstrap smoke inside a QS pod.

The smoke intentionally avoids large model/data dependencies. It validates the
current V3 researcher-CoT training path with synthetic message-format data and a
tiny randomly initialized Llama model plus LoRA on the visible GPU. A later smoke
can swap in real Qwen weights once the model/cache location on /mnt/3fs is fixed.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import importlib.metadata as importlib_metadata
import json
import math
import platform
import sys
import types
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from autoresearch_idea_harness.io import load_config, write_json  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _synthetic_rows() -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Richard Sutton, a world-class AI researcher. "
                        "Reason toward one genuinely novel idea."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "It is 1988. Temporal credit assignment is unstable. "
                        "Find a better way to learn from partial predictions."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I would stop treating prediction targets as fixed labels. "
                        "The Core idea is to bootstrap from my own later estimates, "
                        "so each update improves a moving forecast rather than waiting "
                        "for terminal feedback. The Non-trivial crux is controlling "
                        "the bias from bootstrapping while gaining sample efficiency."
                    ),
                },
            ],
            "researcher": "Richard Sutton",
            "source": "qs_bootstrap_smoke",
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Kaiming He, a world-class AI researcher. "
                        "Reason toward one genuinely novel idea."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "It is 2015. Very deep visual networks degrade as depth "
                        "increases even when overfitting is not the main cause."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I would make the network learn the residual function instead "
                        "of the whole transformation. The Core idea is to give every "
                        "block an identity path, so optimization can preserve a good "
                        "solution and only learn the correction. The Non-trivial crux "
                        "is that this changes the optimization geometry without making "
                        "the architecture a shallow ensemble."
                    ),
                },
            ],
            "researcher": "Kaiming He",
            "source": "qs_bootstrap_smoke",
        },
    ]


def _build_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    texts: list[str] = []
    for row in _synthetic_rows():
        for message in row["messages"]:
            texts.extend([message["role"], message["content"]])
    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<s>": 2,
        "</s>": 3,
        "<|im_start|>": 4,
        "<|im_end|>": 5,
        "\n": 6,
    }
    for text in texts:
        for token in text.replace("\n", " ").split():
            token = token.strip()
            if token and token not in vocab:
                vocab[token] = len(vocab)
    raw = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tok.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
    )
    return tok


def _module_presence(names: list[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _module_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _patch_transformers_modeling_layers() -> bool:
    if importlib.util.find_spec("transformers.modeling_layers") is not None:
        return False
    import torch

    module = types.ModuleType("transformers.modeling_layers")
    module.__spec__ = importlib.machinery.ModuleSpec("transformers.modeling_layers", loader=None)

    class GradientCheckpointingLayer(torch.nn.Module):
        pass

    module.GradientCheckpointingLayer = GradientCheckpointingLayer
    sys.modules["transformers.modeling_layers"] = module
    return True


def run_smoke(run_dir: Path, require_cuda: bool) -> dict[str, Any]:
    import torch

    patched_modeling_layers = _patch_transformers_modeling_layers()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    import train_v3_researcher_cot_lora as train_mod

    started = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    train_jsonl = data_dir / "train.jsonl"
    _write_jsonl(train_jsonl, _synthetic_rows())

    cfg = load_config(ROOT / "configs" / "default.yaml")
    qs_cfg = cfg.get("v3_training", {}).get("qs", {})
    modules = _module_presence(["torch", "transformers", "peft", "datasets", "tokenizers"])
    versions = _module_versions(["torch", "transformers", "peft", "datasets", "tokenizers"])
    if not all(modules.values()):
        missing = [name for name, ok in modules.items() if not ok]
        raise RuntimeError(f"missing bootstrap modules: {missing}")

    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for QS bootstrap smoke but is not available")
    device = torch.device("cuda:0" if cuda_available else "cpu")
    gpu = {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "name": torch.cuda.get_device_name(0) if cuda_available else None,
        "capability": torch.cuda.get_device_capability(0) if cuda_available else None,
        "total_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            if cuda_available
            else None
        ),
    }

    tokenizer = _build_tokenizer()
    tok_ds = train_mod.load_and_tokenize(str(train_jsonl), tokenizer, max_length=128)
    sample = tok_ds[0]
    active_labels = sum(1 for value in sample["labels"] if value != -100)
    if active_labels <= 0:
        raise RuntimeError("assistant-turn masking produced zero active labels")

    model_cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(model_cfg).to(device)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    collator = train_mod.PaddingCollator(tokenizer=tokenizer, max_length=128)
    batch = collator([tok_ds[0], tok_ds[1]])
    batch = {key: value.to(device) for key, value in batch.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
    forward_start = time.time()
    output = model(**batch)
    loss = output.loss
    if not math.isfinite(float(loss.detach().cpu())):
        raise RuntimeError(f"non-finite loss: {loss}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    train_step_s = time.time() - forward_start

    adapter_dir = run_dir / "tiny_lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir / "tokenizer")

    result = {
        "status": "ok",
        "repo_root": str(ROOT),
        "run_dir": str(run_dir),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "qs_config": {
            "queue_id": qs_cfg.get("queue_id"),
            "cluster_id": qs_cfg.get("cluster_id"),
            "resource_package_id": qs_cfg.get("resource_package_id"),
            "remote_project_root": qs_cfg.get("remote_project_root"),
        },
        "modules": modules,
        "module_versions": versions,
        "compat": {
            "patched_transformers_modeling_layers": patched_modeling_layers,
            "transformers_modeling_layers_spec": str(importlib.util.find_spec("transformers.modeling_layers")),
        },
        "gpu": gpu,
        "dataset": {
            "train_jsonl": str(train_jsonl),
            "rows": len(tok_ds),
            "first_active_labels": active_labels,
            "first_tokens": len(sample["input_ids"]),
        },
        "training": {
            "model": "tiny_random_llama_lora",
            "loss": round(float(loss.detach().cpu()), 6),
            "step_seconds": round(train_step_s, 4),
            "adapter_dir": str(adapter_dir),
            "peak_gpu_memory_gb": (
                round(torch.cuda.max_memory_allocated() / 1e9, 4) if cuda_available else None
            ),
        },
        "elapsed_seconds": round(time.time() - started, 4),
    }
    write_json(run_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    result = run_smoke(args.run_dir, require_cuda=not args.allow_cpu)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

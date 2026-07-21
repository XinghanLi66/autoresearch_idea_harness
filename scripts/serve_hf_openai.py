#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/chat/completions server on HF transformers.

Stands up a proposer endpoint WITHOUT vLLM (not in the QS GB200 image). Loads a (large) HF model
with device_map="auto" (naive pipeline-MP across the pod's GPUs; tp_plan=None disables tfm auto-TP),
and serves chat completions using the model's own chat template. Meant for the M2 235B proposer on QS.

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (subset of the OpenAI schema).
Run:  python scripts/serve_hf_openai.py --model <dir> --host 0.0.0.0 --port 8000 [--enable-thinking false]
"""
from __future__ import annotations
import argparse, json, time, uuid, threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any


class ChatReq(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    enable_thinking: bool | None = None  # per-request override; else server default


def build_app(model, tok, served_name: str, default_thinking: bool) -> FastAPI:
    app = FastAPI()
    gen_lock = threading.Lock()  # single model, serialize generate() calls

    @app.get("/health")
    def health():
        return {"status": "ok", "model": served_name}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatReq):
        think = default_thinking if req.enable_thinking is None else req.enable_thinking
        try:
            prompt = tok.apply_chat_template(req.messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=think)
        except TypeError:
            prompt = tok.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        in_len = inputs["input_ids"].shape[1]
        do_sample = req.temperature is not None and req.temperature > 0
        with gen_lock, torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=req.max_tokens, do_sample=do_sample,
                temperature=req.temperature if do_sample else None,
                top_p=req.top_p if do_sample else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = out[0][in_len:]
        text = tok.decode(gen, skip_special_tokens=True)
        return JSONResponse({
            "id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion",
            "created": int(time.time()), "model": served_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": int(in_len), "completion_tokens": int(gen.shape[0]),
                      "total_tokens": int(in_len + gen.shape[0])},
        })

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="optional PEFT LoRA adapter dir to load on the base")
    ap.add_argument("--served-name", default="m2-sft")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--enable-thinking", default="true", choices=["true", "false"])
    args = ap.parse_args()

    print(f"[serve] loading {args.model} (device_map=auto, bf16) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", tp_plan=None)
    if args.adapter:
        from peft import PeftModel
        print(f"[serve] attaching LoRA adapter {args.adapter} ...", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()  # fold LoRA in for fast inference
    model.eval()
    print("[serve] model loaded; starting API", flush=True)
    app = build_app(model, tok, args.served_name, args.enable_thinking == "true")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

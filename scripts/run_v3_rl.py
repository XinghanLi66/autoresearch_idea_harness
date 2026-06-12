#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _setup_env() -> None:
    defaults = {
        "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/torch_inductor",
        "TRITON_CACHE_DIR": "/dev/shm/triton_cache",
        "TMPDIR": "/dev/shm/tmp",
        "OMP_NUM_THREADS": "1",
        "VLLM_PLUGINS": "",
        "SYMPY_GROUND_TYPES": "python",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    libcuda = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    if os.path.exists(libcuda):
        existing = os.environ.get("LD_PRELOAD", "")
        os.environ["LD_PRELOAD"] = f"{existing}:{libcuda}" if existing else libcuda
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    for key in ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR"]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


_setup_env()

from omegaconf import OmegaConf


def _load_verl_ppo_base():
    import importlib.resources as res
    import verl.trainer.config as vcfg_pkg

    with res.files(vcfg_pkg).joinpath("_generated_ppo_trainer.yaml").open() as f:
        vcfg = OmegaConf.create(f.read())
    vcfg = OmegaConf.merge(vcfg, OmegaConf.load(str(ROOT.parent / "proposal_rl" / "configs" / "verl_ppo.yaml")))
    return vcfg


def run_v3_rl(args: argparse.Namespace) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_log = output_dir / "rollout_rewards.jsonl"

    os.environ["EMBED_MODEL"] = args.embed_model
    os.environ["V3_RL_ROLLOUT_LOG"] = str(rollout_log)
    os.environ["V3_PRS_WEIGHT"] = str(args.prs_weight)
    os.environ["V3_FORMAT_WEIGHT"] = str(args.format_weight)

    n_gpu = int(os.environ.get("NGPU", args.n_gpus))
    per_device = int(args.per_device_train_batch_size)
    train_batch_size = n_gpu * per_device * int(args.gradient_accumulation_steps)
    rollout_tp = int(args.rollout_tensor_parallel_size or n_gpu)
    if n_gpu % rollout_tp != 0:
        raise ValueError(f"n_gpus={n_gpu} must be divisible by rollout_tensor_parallel_size={rollout_tp}")

    vcfg = _load_verl_ppo_base()
    overrides = OmegaConf.create({
        "data": {
            "train_files": str(Path(args.train_file).resolve()),
            "val_files": str(Path(args.train_file).resolve()),
            "train_batch_size": train_batch_size,
            "max_prompt_length": int(args.max_prompt_length),
            "max_response_length": int(args.max_response_length),
            "return_raw_chat": False,
            "filter_overlong_prompts": True,
            "truncation": "left",
        },
        "actor_rollout_ref": {
            "model": {
                "path": str(Path(args.model_path).resolve()),
                "lora_rank": int(args.lora_r),
                "lora_alpha": int(args.lora_alpha),
                "enable_gradient_checkpointing": True,
                "trust_remote_code": True,
                "override_config": {"attn_implementation": "sdpa"},
            },
            "actor": {
                "optim": {
                    "lr": float(args.learning_rate),
                    "lr_warmup_steps_ratio": float(args.warmup_ratio),
                },
                "ppo_mini_batch_size": max(1, min(train_batch_size, n_gpu * per_device)),
                "ppo_micro_batch_size_per_gpu": per_device,
                "ppo_max_token_len_per_gpu": int(args.ppo_max_token_len),
                "use_kl_loss": True,
                "kl_loss_coef": float(args.kl_coeff),
                "rollout_n": int(args.num_generations),
            },
            "rollout": {
                "n": int(args.num_generations),
                "tensor_model_parallel_size": rollout_tp,
                "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
                "enforce_eager": True,
                "max_num_batched_tokens": int(args.ppo_max_token_len),
                "max_model_len": int(args.max_prompt_length) + int(args.max_response_length),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": per_device,
                "log_prob_max_token_len_per_gpu": int(args.ppo_max_token_len),
                "fsdp_config": {"param_offload": False},
            },
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "kl_ctrl": {"kl_coef": float(args.kl_coeff)},
        },
        "reward": {
            "custom_reward_function": {
                "path": str(ROOT / "scripts" / "v3_verl_reward.py"),
                "name": "compute_score",
                "reward_kwargs": {},
            },
            "num_workers": int(args.reward_workers),
        },
        "trainer": {
            "project_name": "autoresearch_idea_harness",
            "experiment_name": args.run_id,
            "default_local_dir": str(output_dir),
            "total_epochs": int(args.num_train_epochs),
            "total_training_steps": int(args.total_training_steps) if args.total_training_steps else None,
            "save_freq": int(args.save_steps),
            "n_gpus_per_node": n_gpu,
            "logger": ["console"],
            "resume_mode": "auto" if args.resume else "disable",
        },
        "ray_kwargs": {
            "ray_init": {
                "num_cpus": max(n_gpu * 4, 32),
                "_temp_dir": "/dev/shm/ray_tmp",
            },
        },
    })
    vcfg = OmegaConf.merge(vcfg, overrides)
    (output_dir / "verl_config.yaml").write_text(OmegaConf.to_yaml(vcfg))

    # Warm caches before Ray workers start.
    SentenceTransformer(args.embed_model)
    if torch.cuda.is_available():
        torch.ones(1, device="cuda:0")
    shutil.rmtree(os.environ["TORCHINDUCTOR_CACHE_DIR"], ignore_errors=True)
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    from verl.trainer.main_ppo import run_ppo

    run_ppo(vcfg)

    final_dir = output_dir / "final"
    if not (final_dir / "config.json").exists():
        ckpt_dirs = sorted(output_dir.glob("global_step_*"))
        if ckpt_dirs:
            actor_dir = ckpt_dirs[-1] / "actor"
            hf_subdir = actor_dir / "huggingface"
            if not (hf_subdir / "config.json").exists():
                hf_subdir.mkdir(parents=True, exist_ok=True)
                model_path = Path(args.model_path)
                for pattern in ("*.json", "*.model", "tokenizer*"):
                    for src in model_path.glob(pattern):
                        shutil.copy2(src, hf_subdir / src.name)
            import subprocess

            final_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "verl.model_merger",
                    "merge",
                    "--backend",
                    "fsdp",
                    "--local_dir",
                    str(actor_dir),
                    "--target_dir",
                    str(final_dir),
                ],
                cwd=str(ROOT.parent / "proposal_rl"),
                check=False,
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Run V3 PRS GRPO training with verl.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reward-type", default="prs", choices=["prs"])
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--n-gpus", type=int, default=8)
    p.add_argument("--rollout-tensor-parallel-size", type=int, default=8)
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--total-training-steps", type=int, default=80)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2.0e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--kl-coeff", type=float, default=0.02)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=4096)
    p.add_argument("--max-response-length", type=int, default=1800)
    p.add_argument("--ppo-max-token-len", type=int, default=8192)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--save-steps", type=int, default=40)
    p.add_argument("--reward-workers", type=int, default=4)
    p.add_argument("--prs-weight", type=float, default=0.8)
    p.add_argument("--format-weight", type=float, default=0.2)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.reward_type != "prs":
        raise ValueError("V3 RL currently supports PRS only.")
    run_v3_rl(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a compute-soak training queue for QS (keeps idle GB200 busy; jobs wait at priority 0).

Emits runs/qs_training_queue/submit_all.sh — one (prepare + guarded-submit) block per job. Running that
script queues every job on QS at low priority, so the scheduler backfills idle GPUs behind cc000's primary
zoo runs and no compute is wasted. This generator does NOT submit anything itself.

Default matrix = the training-hparam SWEEP (lr variants) over the model-zoo arms. Override with --manifest
<json> ({"jobs":[{run_id,method,base_model,lr,max_steps,seq_len,grad_accum,nproc,workers,save_steps}]}).

NOTE: base_model paths must be staged to /mnt/3fs by cc000 first; adjust paths in the manifest to match.
Run the output with the guard envs set, e.g.:  CONFIRM_SUBMIT_V3_QS_FULL_SFT=1 CONFIRM_SUBMIT_V3_QS_LORA=1 bash runs/qs_training_queue/submit_all.sh
"""
from __future__ import annotations
import argparse, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
M3 = "/mnt/3fs/lxh/agentic-training"
TRAIN = f"{M3}/data/researcher_cot/anchored_v1/train.jsonl"
VAL = f"{M3}/data/researcher_cot/anchored_v1/val.jsonl"

# Default: lr-sweep (hparam swap) over the zoo. base lr + one alt per arm. Priority 0 = backfill.
DEFAULT_JOBS = [
    # full-FT dense (1 epoch ~101 steps, seq 1664)
    {"run_id": "zoo_qwen3_8b_lr5e6",  "method": "full", "base_model": f"{M3}/models/Qwen3-8B",  "lr": 5e-6, "workers": 1},
    {"run_id": "zoo_qwen3_8b_lr1e5",  "method": "full", "base_model": f"{M3}/models/Qwen3-8B",  "lr": 1e-5, "workers": 1},
    {"run_id": "zoo_qwen3_14b_lr3e6", "method": "full", "base_model": f"{M3}/models/Qwen3-14B", "lr": 3e-6, "workers": 1},
    {"run_id": "zoo_qwen3_14b_lr6e6", "method": "full", "base_model": f"{M3}/models/Qwen3-14B", "lr": 6e-6, "workers": 1},
    {"run_id": "zoo_qwen3_32b_lr2e6", "method": "full", "base_model": f"{M3}/models/Qwen3-32B", "lr": 2e-6, "workers": 1},
    {"run_id": "zoo_qwen3_32b_lr4e6", "method": "full", "base_model": f"{M3}/models/Qwen3-32B", "lr": 4e-6, "workers": 1},
    # LoRA MoE (2-3 epochs ok; keep 1-2 epoch here)
    {"run_id": "zoo_qwen3_30b_a3b_lr1e4", "method": "lora", "base_model": f"{M3}/models/Qwen3-30B-A3B",  "lr": 1e-4, "workers": 1},
    {"run_id": "zoo_qwen3_30b_a3b_lr5e5", "method": "lora", "base_model": f"{M3}/models/Qwen3-30B-A3B",  "lr": 5e-5, "workers": 1},
    {"run_id": "zoo_qwen3_235b_lr5e5",    "method": "lora", "base_model": f"{M3}/models/Qwen3-235B-A22B","lr": 5e-5, "workers": 2},
    {"run_id": "zoo_qwen3_235b_lr2e5",    "method": "lora", "base_model": f"{M3}/models/Qwen3-235B-A22B","lr": 2e-5, "workers": 2},
]
DEFAULTS = {"max_steps": 101, "seq_len": 1664, "grad_accum": 4, "nproc": 4, "save_steps": 101}


def block(j: dict) -> str:
    j = {**DEFAULTS, **j}
    if j["method"] == "full":
        prep = "scripts/prepare_v3_qs_researcher_full_sft_run.py"
        outdir = "runs/qs_researcher_full_sft"
        guard = "CONFIRM_SUBMIT_V3_QS_FULL_SFT"
        extra = (f" --grad-accum {j['grad_accum']} --nproc-per-node {j['nproc']} "
                 f"--save-steps {j['save_steps']} --no-resume-check")
    else:
        prep = "scripts/prepare_v3_qs_researcher_lora_train_run.py"
        outdir = "runs/qs_researcher_lora_train"
        guard = "CONFIRM_SUBMIT_V3_QS_LORA"
        extra = ""
    submit_sh = f"{outdir}/{j['run_id']}/qs_create_job.sh"
    rid, lr, w, meth = j["run_id"], j["lr"], j["workers"], j["method"]
    return (
        f'echo "=== {rid} ({meth}, lr={lr}, {w}w) ==="\n'
        f'python {prep} --run-id {rid} --base-model {j["base_model"]} '
        f'--remote-train-jsonl {TRAIN} --remote-val-jsonl {VAL} '
        f'--max-steps {j["max_steps"]} --max-seq-length {j["seq_len"]} --lr {lr}{extra}\n'
        f'{guard}=1 bash {submit_sh}\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="JSON with {jobs:[...]}; default = built-in lr sweep")
    ap.add_argument("--out", default=str(ROOT / "runs/qs_training_queue/submit_all.sh"))
    a = ap.parse_args()
    jobs = json.loads(pathlib.Path(a.manifest).read_text())["jobs"] if a.manifest else DEFAULT_JOBS
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    hdr = ("#!/usr/bin/env bash\n# Auto-generated compute-soak training queue (priority 0). Review before running.\n"
           "# Requires base_model paths staged on /mnt/3fs. Set guard envs to actually submit.\n"
           "set -uo pipefail\ncd \"$(dirname \"$0\")/../..\"\n\n")
    out.write_text(hdr + "\n".join(block(j) for j in jobs))
    out.chmod(0o755)
    print(f"wrote {out} with {len(jobs)} queued jobs (priority 0).")
    print("jobs:", ", ".join(j["run_id"] for j in jobs))
    print("\nto queue them (after weights staged):")
    print(f"  CONFIRM_SUBMIT_V3_QS_FULL_SFT=1 CONFIRM_SUBMIT_V3_QS_LORA=1 bash {out}")


if __name__ == "__main__":
    main()

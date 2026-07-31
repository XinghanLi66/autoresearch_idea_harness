#!/usr/bin/env bash
# CORRECT LoRA upload: ignore the PEFT-generated README (bad base_model path); upload weights, then a
# clean card with a VALID public base_model id. Adds M1 Qwen3-30B-A3B SFT LoRA. Idempotent.
set -uo pipefail
B=/mnt/3fs/lxh/agentic-training
echo "==== LoRA+M1 upload (fixed) $(date -Is) host=$(hostname) ===="
HF_TOKEN=""; for f in /mnt/3fs/lxh/.hf_token "$B/.hf_token"; do [ -f "$f" ] && { HF_TOKEN="$(tr -d '[:space:]' <"$f")"; break; }; done
[ -z "${HF_TOKEN:-}" ] && { echo "ABORT_NO_TOKEN"; exit 7; }
export HF_TOKEN; export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -m pip install -q "huggingface_hub>=0.36" hf_transfer 2>&1 | tail -1 || true
python3 - <<'PY' 2>&1
import os, traceback
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); NS=api.whoami().get("name"); print("NAMESPACE:",NS)
B="/mnt/3fs/lxh/agentic-training"; PFX="idea-proposal-training"
# ignore the bad PEFT README + checkpoints + optimizer; upload adapter weights/config/tokenizer only
IGN=["README.md","checkpoint-*","checkpoint-*/**","optimizer*","*.pt","*.pth","global_step*"]
def card(title,bm_public,lineage,recipe):
    return (f"---\nlibrary_name: peft\nbase_model: {bm_public}\ntags:\n- idea-proposal-training\n- research-agent\n---\n\n"
            f"# {title}\n\nPrivate transfer copy (V3 idea-proposal-training).\n\n"
            f"**Ultimate base (public):** {bm_public}\n\n**Actual training base (lineage):** {lineage}\n\n"
            f"**Apply recipe:**\n{recipe}\n")
# (repo, src, public_base_model_id, lineage, title, recipe)
ARMS=[
 (f"{PFX}-d1rl-lora", f"{B}/runs/researcher_cot/rl/dpo_d1_r1_0528_qwen3_8b/output/checkpoint-290",
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", f"{NS}/{PFX}-d1sft (our D1-SFT full weights)",
  "D1-RL (DPO LoRA on D1-SFT)", f"Load {NS}/{PFX}-d1sft; PeftModel.from_pretrained(base,this_repo); merge_and_unload()."),
 (f"{PFX}-m2sft-lora", f"{B}/runs/qs_researcher_lora_train/m2_qwen3_235b_a22b_lora/output_fullep",
  "Qwen/Qwen3-235B-A22B", "public Qwen/Qwen3-235B-A22B (this IS the SFT adapter)",
  "M2-SFT (Qwen3-235B-A22B MoE, SFT LoRA)", "Load Qwen/Qwen3-235B-A22B (device_map=auto,tp_plan=None); apply adapter; merge_and_unload() -> merged_sft."),
 (f"{PFX}-m2rl-lora", f"{B}/runs/researcher_cot/rl/dpo_m2_qwen3_235b_fullep/output",
  "Qwen/Qwen3-235B-A22B", f"merged_sft = Qwen/Qwen3-235B-A22B + {NS}/{PFX}-m2sft-lora (merged)",
  "M2-RL (Qwen3-235B-A22B MoE, DPO LoRA) full epoch",
  f"(1) merged_sft = merge Qwen/Qwen3-235B-A22B + {NS}/{PFX}-m2sft-lora; (2) apply this adapter to merged_sft; merge_and_unload(). ~53GB adapter."),
 (f"{PFX}-qwen3-8b-rl-lora", f"{B}/runs/researcher_cot/rl/dpo_qwen3_8b",
  "Qwen/Qwen3-8B", "our Qwen3-8B SFT S1 ckpt-102 (runs/qs_researcher_full_sft/s1_qwen3_8b_full/output/checkpoint-102, uploaded separately)",
  "Qwen3-8B-RL (DPO LoRA on S1 SFT)", "Load S1-SFT; PeftModel.from_pretrained+merge_and_unload."),
 (f"{PFX}-qwen3-14b-rl-lora", f"{B}/runs/researcher_cot/rl/dpo_qwen3_14b",
  "Qwen/Qwen3-14B", "our Qwen3-14B SFT S2 ckpt-102 (uploaded separately)",
  "Qwen3-14B-RL (DPO LoRA on S2 SFT)", "Load S2-SFT; PeftModel.from_pretrained+merge_and_unload."),
 (f"{PFX}-qwen3-32b-rl-lora", f"{B}/runs/researcher_cot/rl/dpo_qwen3_32b",
  "Qwen/Qwen3-32B", "our Qwen3-32B SFT S3 ckpt-102 (uploaded separately)",
  "Qwen3-32B-RL (DPO LoRA on S3 SFT)", "Load S3-SFT; PeftModel.from_pretrained+merge_and_unload."),
 (f"{PFX}-m1-qwen3-30b-a3b-sft-lora", f"{B}/runs/qs_researcher_lora_train/m1_qwen3_30b_a3b_lora/output",
  "Qwen/Qwen3-30B-A3B", "public Qwen/Qwen3-30B-A3B (M1 SFT LoRA, not a final eval arm)",
  "M1 Qwen3-30B-A3B (SFT LoRA)", "Load Qwen/Qwen3-30B-A3B; PeftModel.from_pretrained(base,this_repo); merge_and_unload(). final adapter only (checkpoint-* excluded)."),
]
res=[]
for repo_s,src,bm,lin,title,recipe in ARMS:
    repo=f"{NS}/{repo_s}"; print(f"\n>>> {repo}",flush=True)
    try:
        api.create_repo(repo,private=True,repo_type="model",exist_ok=True)
        ci=api.upload_folder(folder_path=src,repo_id=repo,ignore_patterns=IGN,commit_message=f"upload {repo_s} weights")
        api.upload_file(path_or_fileobj=card(title,bm,lin,recipe).encode(),path_in_repo="README.md",repo_id=repo,commit_message="clean model card")
        w=any(f.endswith("adapter_model.safetensors") for f in api.list_repo_files(repo))
        print(f"DONE {repo} sha={getattr(ci,'oid','?')} weights={'YES' if w else 'NO'} url=https://huggingface.co/{repo}",flush=True)
        res.append((repo,getattr(ci,'oid','?'),w))
    except Exception as e:
        print(f"FAIL {repo}: {type(e).__name__}: {e}",flush=True); traceback.print_exc(); res.append((repo,"FAIL",False))
print("\n==== LORA+M1 SUMMARY ====")
for r,s,w in res: print(f"  {r} -> sha={s} weights={w}")
PY
echo "==== done $(date -Is) ===="
sleep 15

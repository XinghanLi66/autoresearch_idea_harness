#!/usr/bin/env bash
# Finish the transfer: 6 LoRA adapters (valid peft cards) + qwen25sft (full). Idempotent (hash-skip).
set -uo pipefail
B=/mnt/3fs/lxh/agentic-training
echo "==== finish upload $(date -Is) host=$(hostname) ===="
HF_TOKEN=""
for f in /mnt/3fs/lxh/.hf_token "$B/.hf_token" /mnt/3fs/lxh/.cache/huggingface/token; do
  [ -f "$f" ] && { HF_TOKEN="$(tr -d '[:space:]' < "$f")"; echo "  token file: $f"; break; }
done
[ -z "${HF_TOKEN:-}" ] && { echo "ABORT_NO_TOKEN"; exit 7; }
export HF_TOKEN; export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -m pip install -q "huggingface_hub>=0.36" hf_transfer 2>&1 | tail -1 || true
python3 - <<'PY' 2>&1
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); NS=api.whoami().get("name"); print("NAMESPACE:",NS)
B="/mnt/3fs/lxh/agentic-training"; PFX="idea-proposal-training"
IGN=["checkpoint-*","checkpoint-*/**","optimizer*","*.pt","*.pth","global_step*"]
ALLOW_FULL=["*.safetensors","*.json","tokenizer*","*.txt","*.jinja","*.model","merges.txt","vocab.json","chat_template.*"]
def pcard(title,bm,base_desc,recipe):
    return (f"---\nlibrary_name: peft\nbase_model: {bm}\ntags:\n- idea-proposal-training\n- research-agent\n---\n\n"
            f"# {title}\n\nPrivate transfer copy (V3 idea-proposal-training).\n\n**Base:** {base_desc}\n\n**Recipe:**\n{recipe}\n")
def fcard(title,base_desc,note):
    return (f"---\nlibrary_name: transformers\ntags:\n- idea-proposal-training\n- research-agent\n---\n\n"
            f"# {title}\n\nPrivate transfer copy (V3 idea-proposal-training).\n\n**Base:** {base_desc}\n\n{note}\n")
ADPT=[
 (f"{PFX}-d1rl-lora", f"{B}/runs/researcher_cot/rl/dpo_d1_r1_0528_qwen3_8b/output/checkpoint-290", f"{NS}/{PFX}-d1sft",
  f"{NS}/{PFX}-d1sft (D1-SFT full)", "D1-RL (DPO LoRA on D1-SFT)",
  f"LoRA. base {NS}/{PFX}-d1sft; PeftModel.from_pretrained+merge_and_unload. True base: runs/qs_researcher_full_sft/d1_r1_0528_qwen3_8b/output/checkpoint-102."),
 (f"{PFX}-m2sft-lora", f"{B}/runs/qs_researcher_lora_train/m2_qwen3_235b_a22b_lora/output_fullep", "Qwen/Qwen3-235B-A22B",
  "Qwen/Qwen3-235B-A22B (public)", "M2-SFT (Qwen3-235B-A22B MoE, SFT LoRA)",
  "LoRA on public Qwen3-235B-A22B; apply+merge -> merged_sft (base for M2-RL)."),
 (f"{PFX}-m2rl-lora", f"{B}/runs/researcher_cot/rl/dpo_m2_qwen3_235b_fullep/output", f"{NS}/{PFX}-m2sft-lora",
  "merged_sft = Qwen3-235B-A22B + M2-SFT adapter", "M2-RL (235B MoE DPO LoRA, full epoch)",
  f"LoRA ~53GB. merged_sft = Qwen3-235B-A22B + {NS}/{PFX}-m2sft-lora; then apply this + merge. 438G blobs not needed."),
 (f"{PFX}-qwen3-8b-rl-lora", f"{B}/runs/researcher_cot/rl/dpo_qwen3_8b", "Qwen/Qwen3-8B",
  "our Qwen3-8B SFT S1 ckpt-102 (uploaded separately); public root Qwen/Qwen3-8B", "Qwen3-8B-RL (DPO LoRA on S1 SFT)",
  "base path runs/qs_researcher_full_sft/s1_qwen3_8b_full/output/checkpoint-102; apply+merge."),
 (f"{PFX}-qwen3-14b-rl-lora", f"{B}/runs/researcher_cot/rl/dpo_qwen3_14b", "Qwen/Qwen3-14B",
  "our Qwen3-14B SFT S2 ckpt-102 (uploaded separately); public root Qwen/Qwen3-14B", "Qwen3-14B-RL (DPO LoRA on S2 SFT)",
  "base path runs/qs_researcher_full_sft/s2_qwen3_14b_full/output/checkpoint-102; apply+merge."),
 (f"{PFX}-qwen3-32b-rl-lora", f"{B}/runs/researcher_cot/rl/dpo_qwen3_32b", "Qwen/Qwen3-32B",
  "our Qwen3-32B SFT S3 ckpt-102 (uploaded separately); public root Qwen/Qwen3-32B", "Qwen3-32B-RL (DPO LoRA on S3 SFT)",
  "base path runs/qs_researcher_full_sft/s3_qwen3_32b_full/output/checkpoint-102; apply+merge."),
]
res=[]
for repo_s,src,bm,bd,title,recipe in ADPT:
    repo=f"{NS}/{repo_s}"; print(f"\n>>> {repo} (adapter)",flush=True)
    try:
        api.create_repo(repo,private=True,repo_type="model",exist_ok=True)
        ci=api.upload_folder(folder_path=src,repo_id=repo,ignore_patterns=IGN,commit_message=f"upload {repo_s}")
        api.upload_file(path_or_fileobj=pcard(title,bm,bd,recipe).encode(),path_in_repo="README.md",repo_id=repo,commit_message="peft card")
        w=any(f.endswith("adapter_model.safetensors") for f in api.list_repo_files(repo))
        print(f"DONE {repo} sha={getattr(ci,'oid','?')} weights={'YES' if w else 'NO'} url=https://huggingface.co/{repo}",flush=True)
        res.append((repo,getattr(ci,'oid','?'),w))
    except Exception as e:
        print(f"FAIL {repo}: {type(e).__name__}: {e}",flush=True); res.append((repo,"FAIL",False))
# qwen25sft (full)
repo=f"{NS}/{PFX}-qwen25sft"; src=f"{B}/qs_researcher_full_sft/v3_qs_full_sft_32b_anchored_1ep_v4/output/checkpoint-101"
print(f"\n>>> {repo} (full)",flush=True)
try:
    api.create_repo(repo,private=True,repo_type="model",exist_ok=True)
    ci=api.upload_folder(folder_path=src,repo_id=repo,allow_patterns=ALLOW_FULL,commit_message="upload qwen25sft weights-only")
    api.upload_file(path_or_fileobj=fcard("Qwen2.5-32B-v3-SFT (qwen25sft)","Qwen/Qwen2.5-32B-Instruct (public)",
        "V3 full-param researcher-CoT SFT of Qwen2.5-32B-Instruct; parent of the rl arm (qwen2.5-32b-v3-dpo). Weights-only (optimizer excluded).").encode(),
        path_in_repo="README.md",repo_id=repo,commit_message="model card")
    w=any(f.endswith(".safetensors") for f in api.list_repo_files(repo))
    print(f"DONE {repo} sha={getattr(ci,'oid','?')} weights={'YES' if w else 'NO'} url=https://huggingface.co/{repo}",flush=True)
    res.append((repo,getattr(ci,'oid','?'),w))
except Exception as e:
    print(f"FAIL {repo}: {type(e).__name__}: {e}",flush=True); res.append((repo,"FAIL",False))
print("\n==== FINISH SUMMARY ====")
for r,s,w in res: print(f"  {r} -> sha={s} weights={w}")
PY
echo "==== finish done $(date -Is) ===="
sleep 15

#!/usr/bin/env bash
# Re-push the 6 LoRA adapters with VALID peft cards (base_model field). Idempotent:
# upload_folder hash-skips already-uploaded weights; only the fixed README is new.
set -uo pipefail
B=/mnt/3fs/lxh/agentic-training
echo "==== LoRA card fix + verify $(date -Is) host=$(hostname) ===="
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
def card(title,bm,base_desc,recipe):
    return (f"---\nlibrary_name: peft\nbase_model: {bm}\ntags:\n- idea-proposal-training\n- research-agent\n---\n\n"
            f"# {title}\n\nPrivate transfer copy (V3 idea-proposal-training).\n\n"
            f"**Base:** {base_desc}\n\n**Reconstruction / apply recipe:**\n{recipe}\n")
ARMS=[
 dict(repo=f"{PFX}-d1rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_d1_r1_0528_qwen3_8b/output/checkpoint-290",
      bm=f"{NS}/{PFX}-d1sft", base_desc=f"{NS}/{PFX}-d1sft (D1-SFT full weights)",
      title="D1-RL (DPO LoRA on D1-SFT)",
      recipe=f"LoRA. Load base {NS}/{PFX}-d1sft, PeftModel.from_pretrained(base,this_repo), merge_and_unload(). True base path: runs/qs_researcher_full_sft/d1_r1_0528_qwen3_8b/output/checkpoint-102."),
 dict(repo=f"{PFX}-m2sft-lora", src=f"{B}/runs/qs_researcher_lora_train/m2_qwen3_235b_a22b_lora/output_fullep",
      bm="Qwen/Qwen3-235B-A22B", base_desc="Qwen/Qwen3-235B-A22B (public HF)",
      title="M2-SFT (Qwen3-235B-A22B MoE, SFT LoRA)",
      recipe="LoRA on public Qwen/Qwen3-235B-A22B. Load base (device_map=auto, tp_plan=None), apply adapter, merge_and_unload() -> merged_sft (base for M2-RL)."),
 dict(repo=f"{PFX}-m2rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_m2_qwen3_235b_fullep/output",
      bm=f"{NS}/{PFX}-m2sft-lora", base_desc="merged_sft = Qwen3-235B-A22B + M2-SFT adapter (merged)",
      title="M2-RL (Qwen3-235B-A22B MoE, DPO LoRA) full epoch",
      recipe=f"LoRA (~53GB all-linear MoE). (1) merged_sft = merge Qwen/Qwen3-235B-A22B + {NS}/{PFX}-m2sft-lora; (2) PeftModel.from_pretrained(merged_sft,this_repo); merge_and_unload(). 438G blobs not needed."),
 dict(repo=f"{PFX}-qwen3-8b-rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_qwen3_8b",
      bm="Qwen/Qwen3-8B", base_desc="our Qwen3-8B SFT S1 ckpt-102 (uploaded separately from /newcpfs); ultimate public base Qwen/Qwen3-8B",
      title="Qwen3-8B-RL (DPO LoRA on S1 SFT)",
      recipe="LoRA on our Qwen3-8B SFT (S1). Base path: runs/qs_researcher_full_sft/s1_qwen3_8b_full/output/checkpoint-102. PeftModel.from_pretrained(S1_sft,this_repo); merge_and_unload()."),
 dict(repo=f"{PFX}-qwen3-14b-rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_qwen3_14b",
      bm="Qwen/Qwen3-14B", base_desc="our Qwen3-14B SFT S2 ckpt-102 (uploaded separately); ultimate public base Qwen/Qwen3-14B",
      title="Qwen3-14B-RL (DPO LoRA on S2 SFT)",
      recipe="LoRA on our Qwen3-14B SFT (S2). Base path: runs/qs_researcher_full_sft/s2_qwen3_14b_full/output/checkpoint-102. PeftModel.from_pretrained(S2_sft,this_repo); merge_and_unload()."),
 dict(repo=f"{PFX}-qwen3-32b-rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_qwen3_32b",
      bm="Qwen/Qwen3-32B", base_desc="our Qwen3-32B SFT S3 ckpt-102 (uploaded separately); ultimate public base Qwen/Qwen3-32B",
      title="Qwen3-32B-RL (DPO LoRA on S3 SFT)",
      recipe="LoRA on our Qwen3-32B SFT (S3). Base path: runs/qs_researcher_full_sft/s3_qwen3_32b_full/output/checkpoint-102. PeftModel.from_pretrained(S3_sft,this_repo); merge_and_unload()."),
]
res=[]
for a in ARMS:
    repo=f"{NS}/{a['repo']}"; print(f"\n>>> {repo}",flush=True)
    try:
        api.create_repo(repo,private=True,repo_type="model",exist_ok=True)
        ci=api.upload_folder(folder_path=a["src"],repo_id=repo,ignore_patterns=IGN,commit_message=f"upload {a['repo']} (weights)")
        api.upload_file(path_or_fileobj=card(a["title"],a["bm"],a["base_desc"],a["recipe"]).encode(),
                        path_in_repo="README.md",repo_id=repo,commit_message="add valid peft model card")
        files=api.list_repo_files(repo)
        has_w=any(f.endswith("adapter_model.safetensors") for f in files)
        sha=getattr(ci,"oid","?")
        print(f"DONE {repo} PRIVATE sha={sha} weights={'YES' if has_w else 'NO'} url=https://huggingface.co/{repo}",flush=True)
        res.append((repo,sha,has_w))
    except Exception as e:
        print(f"FAIL {repo}: {type(e).__name__}: {e}",flush=True); res.append((repo,"FAIL",False))
print("\n==== FIX SUMMARY ====")
for r,s,w in res: print(f"  {r} -> sha={s} weights={w}")
PY
echo "==== done $(date -Is) ===="
sleep 15

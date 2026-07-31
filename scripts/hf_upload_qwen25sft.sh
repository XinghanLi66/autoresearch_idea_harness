#!/usr/bin/env bash
# Follow-up: upload qwen25sft (V3 full-param SFT of Qwen2.5-32B-Instruct, ckpt-101) weights-only to private HF.
set -uo pipefail
B=/mnt/3fs/lxh/agentic-training
echo "==== qwen25sft HF upload $(date -Is) host=$(hostname) ===="
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
src="/mnt/3fs/lxh/agentic-training/qs_researcher_full_sft/v3_qs_full_sft_32b_anchored_1ep_v4/output/checkpoint-101"
repo=f"{NS}/idea-proposal-training-qwen25sft"
ALLOW=["*.safetensors","*.json","tokenizer*","*.txt","*.jinja","*.model","merges.txt","vocab.json","chat_template.*"]
card=("---\nlibrary_name: transformers\ntags: [idea-proposal-training, research-agent]\n---\n\n"
 "# Qwen2.5-32B-v3-SFT (qwen25sft)\n\nPrivate transfer copy (V3 idea-proposal-training).\n\n"
 "**Base:** Qwen/Qwen2.5-32B-Instruct (public HF)\n\n"
 "V3 full-param researcher-CoT SFT of Qwen2.5-32B-Instruct; parent of the rl arm (qwen2.5-32b-v3-dpo). "
 "Full fine-tuned weights (standalone), optimizer state excluded (weights-only). "
 "Original QS path: qs_researcher_full_sft/v3_qs_full_sft_32b_anchored_1ep_v4/output/checkpoint-101.\n")
print(">>>",repo,"src=",src,flush=True)
api.create_repo(repo,private=True,repo_type="model",exist_ok=True)
ci=api.upload_folder(folder_path=src,repo_id=repo,allow_patterns=ALLOW,commit_message="upload qwen25sft weights-only")
api.upload_file(path_or_fileobj=card.encode(),path_in_repo="README.md",repo_id=repo,commit_message="add model card")
print("DONE",repo,"PRIVATE sha=",getattr(ci,"oid","?"),"url=",getattr(ci,"commit_url",repo),flush=True)
PY
echo "==== qwen25sft upload done $(date -Is) ===="
sleep 15

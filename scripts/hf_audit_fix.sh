#!/usr/bin/env bash
set -uo pipefail
B=/mnt/3fs/lxh/agentic-training
echo "==== HF audit + LoRA retry + M1/mlsbench checks $(date -Is) host=$(hostname) ===="
HF_TOKEN=""
for f in /mnt/3fs/lxh/.hf_token "$B/.hf_token"; do [ -f "$f" ] && { HF_TOKEN="$(tr -d '[:space:]' <"$f")"; break; }; done
[ -z "${HF_TOKEN:-}" ] && { echo "ABORT_NO_TOKEN"; exit 7; }
export HF_TOKEN; export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -m pip install -q "huggingface_hub>=0.36" hf_transfer 2>&1 | tail -1 || true
echo "### PART A/B: HF repo audit + LoRA retry (full traceback) ###"
python3 - <<'PY' 2>&1
import os, traceback
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); NS=api.whoami().get("name"); print("NAMESPACE:",NS)
B="/mnt/3fs/lxh/agentic-training"; PFX="idea-proposal-training"
IGN=["checkpoint-*","checkpoint-*/**","optimizer*","*.pt","*.pth","global_step*"]
def audit(repo):
    try:
        fs=api.list_repo_files(repo)
        w=[f for f in fs if f.endswith("adapter_model.safetensors") or (f.endswith(".safetensors") and "adapter" not in f)]
        print(f"  AUDIT {repo}: files={len(fs)} weights={'YES' if w else 'NO'} readme={'README.md' in fs}")
        return bool(w)
    except Exception as e:
        print(f"  AUDIT {repo}: MISSING/ERR {type(e).__name__}: {e}"); return None
for r in ["d1sft","qwen25sft","d1rl-lora","m2sft-lora","m2rl-lora","qwen3-8b-rl-lora","qwen3-14b-rl-lora","qwen3-32b-rl-lora"]:
    audit(f"{NS}/{PFX}-{r}")
ADPT=[
 (f"{PFX}-d1rl-lora", f"{B}/runs/researcher_cot/rl/dpo_d1_r1_0528_qwen3_8b/output/checkpoint-290", f"{NS}/{PFX}-d1sft"),
 (f"{PFX}-m2sft-lora", f"{B}/runs/qs_researcher_lora_train/m2_qwen3_235b_a22b_lora/output_fullep", "Qwen/Qwen3-235B-A22B"),
 (f"{PFX}-m2rl-lora", f"{B}/runs/researcher_cot/rl/dpo_m2_qwen3_235b_fullep/output", f"{NS}/{PFX}-m2sft-lora"),
 (f"{PFX}-qwen3-8b-rl-lora", f"{B}/runs/researcher_cot/rl/dpo_qwen3_8b", "Qwen/Qwen3-8B"),
 (f"{PFX}-qwen3-14b-rl-lora", f"{B}/runs/researcher_cot/rl/dpo_qwen3_14b", "Qwen/Qwen3-14B"),
 (f"{PFX}-qwen3-32b-rl-lora", f"{B}/runs/researcher_cot/rl/dpo_qwen3_32b", "Qwen/Qwen3-32B"),
]
CARD="---\nlibrary_name: peft\nbase_model: {bm}\ntags:\n- idea-proposal-training\n---\n\n# {t}\n\nPrivate transfer copy. Base: {bm}. LoRA adapter; PeftModel.from_pretrained(base,this_repo)+merge_and_unload.\n"
for repo_s,src,bm in ADPT:
    repo=f"{NS}/{repo_s}"; print(f"\n>>> RETRY {repo}",flush=True)
    try:
        api.create_repo(repo,private=True,repo_type="model",exist_ok=True)
        ci=api.upload_folder(folder_path=src,repo_id=repo,ignore_patterns=IGN,commit_message=f"upload {repo_s}")
        api.upload_file(path_or_fileobj=CARD.format(bm=bm,t=repo_s).encode(),path_in_repo="README.md",repo_id=repo,commit_message="peft card")
        w=any(f.endswith("adapter_model.safetensors") for f in api.list_repo_files(repo))
        print(f"DONE {repo} sha={getattr(ci,'oid','?')} weights={'YES' if w else 'NO'} url=https://huggingface.co/{repo}",flush=True)
    except Exception as e:
        print(f"FAIL {repo}: {type(e).__name__}: {e}",flush=True); traceback.print_exc()
PY
echo "### PART C: Qwen3-30B-A3B (M1) checkpoint on /mnt/3fs ###"
for d in "$B/runs/qs_researcher_lora_train/m1_qwen3_30b_a3b_lora/output" "$B/runs/qs_researcher_lora_train/m1_qwen3_30b_a3b_lora/output_fullep"; do
  echo "-- $d --"
  if [ -d "$d" ]; then
    echo "  PRESENT size=$(du -sh "$d" 2>/dev/null|awk '{print $1}')"
    test -f "$d/adapter_config.json" && echo "  TYPE LoRA base=$(grep -o '\"base_model_name_or_path\"[^,]*' "$d/adapter_config.json"|head -1)"
    ls -1d "$d"/checkpoint-* 2>/dev/null | sed 's/^/  ck: /'
    ls -l "$d"/adapter_model.safetensors 2>/dev/null | awk '{print "  final adapter bytes="$5}'
  else echo "  ABSENT"; fi
done
find "$B/runs" -maxdepth 3 -type d -iname "*30b*" 2>/dev/null | sed 's/^/  found30b: /' | head
echo "### PART D: mlsbench-qs-code at /mnt/3fs/lxh/mlsbench ###"
M=/mnt/3fs/lxh/mlsbench
if [ -d "$M" ]; then
  echo "  PRESENT size=$(du -sh "$M" 2>/dev/null|awk '{print $1}') files=$(find "$M" -type f 2>/dev/null|wc -l)"
  if [ -d "$M/.git" ]; then
    git -C "$M" rev-parse --abbrev-ref HEAD 2>&1 | sed 's/^/  branch: /'
    git -C "$M" remote -v 2>&1 | sed 's/^/  remote: /'
    git -C "$M" rev-parse HEAD 2>&1 | sed 's/^/  HEAD: /'
    git -C "$M" status -s 2>&1 | head -8 | sed 's/^/  st: /'
    git -C "$M" log --oneline -3 2>&1 | sed 's/^/  log: /'
  else echo "  (not a git repo)"; fi
else echo "  ABSENT at $M"; fi
echo "==== done $(date -Is) ===="
sleep 15

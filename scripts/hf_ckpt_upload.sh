#!/usr/bin/env bash
# V3 checkpoint upload to PRIVATE HuggingFace repos (lixinghan leaving-company transfer).
# Runs INSIDE a QS 532 pod (reads /mnt/3fs). Token is read from a FILE on /mnt/3fs — its
# value is NEVER placed in the trial command or printed. Uploads weights-only (d1sft) and
# adapters-only (skips both 438G merged blobs + optimizer state + checkpoint-* subdirs).
set -uo pipefail
B=/mnt/3fs/lxh/agentic-training
LOGD=$B/runs/researcher_cot/rl/serve_logs; mkdir -p "$LOGD"
echo "==== HF upload start $(date -Is) host=$(hostname) ===="
# --- locate HF token WITHOUT printing its value (path only) ---
HF_TOKEN=""
for f in "$B/.hf_token" /mnt/3fs/lxh/.hf_token /mnt/3fs/lxh/.cache/huggingface/token /mnt/3fs/lxh/.huggingface/token "$HOME/.cache/huggingface/token"; do
  if [ -f "$f" ]; then HF_TOKEN="$(tr -d '[:space:]' < "$f")"; echo "  token file: $f"; break; fi
done
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ABORT_NO_TOKEN: no HF token found on /mnt/3fs. Place the coder66 (or lixinghan) HF WRITE token at /mnt/3fs/lxh/.hf_token (chmod 600), then rerun."
  exit 7
fi
export HF_TOKEN; export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -m pip install -q "huggingface_hub>=0.36" hf_transfer 2>&1 | tail -1 || true
python3 - <<'PY' 2>&1
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami(); NS = who.get("name")
print("NAMESPACE:", NS)
B = "/mnt/3fs/lxh/agentic-training"
PFX = "idea-proposal-training"
ALLOW_FULL = ["*.safetensors","*.json","tokenizer*","*.txt","*.jinja","*.model","merges.txt","vocab.json","chat_template.*"]
IGN_ADPT = ["checkpoint-*","checkpoint-*/**","optimizer*","*.pt","*.pth","global_step*"]
def card(title, base, recipe):
    lib = "peft" if ("LoRA" in title or "adapter" in recipe.lower()) else "transformers"
    return (f"---\nlibrary_name: {lib}\ntags: [idea-proposal-training, research-agent]\n---\n\n"
            f"# {title}\n\nPrivate transfer copy (V3 idea-proposal-training).\n\n"
            f"**Base:** {base}\n\n**Reconstruction / apply recipe:**\n{recipe}\n")
ARMS = [
 dict(repo=f"{PFX}-d1sft", src=f"{B}/runs/qs_researcher_full_sft/d1_r1_0528_qwen3_8b/output/checkpoint-102",
      kind="full", base="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (public HF)",
      title="D1-SFT (DeepSeek-R1-0528-Qwen3-8B, full-FT) — FLAGSHIP",
      recipe="Full fine-tuned weights (standalone). Load directly with transformers. Trained from public base deepseek-ai/DeepSeek-R1-0528-Qwen3-8B. Optimizer state excluded (weights-only)."),
 dict(repo=f"{PFX}-d1rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_d1_r1_0528_qwen3_8b/output/checkpoint-290",
      kind="adapter", base=f"{PFX}-d1sft (this org) = D1-SFT full weights",
      title="D1-RL (DPO LoRA on D1-SFT)",
      recipe=f"LoRA adapter. Load base {PFX}-d1sft, then PeftModel.from_pretrained(base, this_repo); merge_and_unload(). Base path: runs/qs_researcher_full_sft/d1_r1_0528_qwen3_8b/output/checkpoint-102."),
 dict(repo=f"{PFX}-m2sft-lora", src=f"{B}/runs/qs_researcher_lora_train/m2_qwen3_235b_a22b_lora/output_fullep",
      kind="adapter", base="Qwen/Qwen3-235B-A22B (public HF)",
      title="M2-SFT (Qwen3-235B-A22B MoE, SFT LoRA)",
      recipe="LoRA on public Qwen/Qwen3-235B-A22B. Apply: load base (device_map=auto, tp_plan=None), PeftModel.from_pretrained(base, this_repo), merge_and_unload() -> merged_sft (base for M2-RL)."),
 dict(repo=f"{PFX}-m2rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_m2_qwen3_235b_fullep/output",
      kind="adapter", base="merged_sft = Qwen3-235B-A22B + M2-SFT adapter (merged)",
      title="M2-RL (Qwen3-235B-A22B MoE, DPO LoRA) — full epoch",
      recipe=f"LoRA (~53GB, all-linear MoE). Reconstruct: (1) merged_sft = merge Qwen/Qwen3-235B-A22B + {PFX}-m2sft-lora; (2) PeftModel.from_pretrained(merged_sft, this_repo); merge_and_unload(). 438G pre-merged blobs NOT needed. Only final root adapter uploaded (checkpoint-* excluded)."),
 dict(repo=f"{PFX}-qwen3-8b-rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_qwen3_8b",
      kind="adapter", base="Qwen3-8B SFT (our S1 ckpt-102, uploaded separately from /newcpfs)",
      title="Qwen3-8B-RL (DPO LoRA on S1 SFT)",
      recipe="LoRA on our Qwen3-8B SFT (S1). Base path: runs/qs_researcher_full_sft/s1_qwen3_8b_full/output/checkpoint-102. Apply: PeftModel.from_pretrained(S1_sft, this_repo); merge_and_unload()."),
 dict(repo=f"{PFX}-qwen3-14b-rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_qwen3_14b",
      kind="adapter", base="Qwen3-14B SFT (our S2 ckpt-102, uploaded separately from /newcpfs)",
      title="Qwen3-14B-RL (DPO LoRA on S2 SFT)",
      recipe="LoRA on our Qwen3-14B SFT (S2). Base path: runs/qs_researcher_full_sft/s2_qwen3_14b_full/output/checkpoint-102. Apply: PeftModel.from_pretrained(S2_sft, this_repo); merge_and_unload()."),
 dict(repo=f"{PFX}-qwen3-32b-rl-lora", src=f"{B}/runs/researcher_cot/rl/dpo_qwen3_32b",
      kind="adapter", base="Qwen3-32B SFT (our S3 ckpt-102, uploaded separately from /newcpfs)",
      title="Qwen3-32B-RL (DPO LoRA on S3 SFT)",
      recipe="LoRA on our Qwen3-32B SFT (S3). Base path: runs/qs_researcher_full_sft/s3_qwen3_32b_full/output/checkpoint-102. Apply: PeftModel.from_pretrained(S3_sft, this_repo); merge_and_unload()."),
]
results = []
for a in ARMS:
    repo = f"{NS}/{a['repo']}"
    print(f"\n>>> {repo}  ({a['kind']})  src={a['src']}", flush=True)
    try:
        api.create_repo(repo, private=True, repo_type="model", exist_ok=True)
        kw = dict(folder_path=a["src"], repo_id=repo, commit_message=f"upload {a['repo']}")
        if a["kind"] == "full": kw["allow_patterns"] = ALLOW_FULL
        else: kw["ignore_patterns"] = IGN_ADPT
        ci = api.upload_folder(**kw)
        sha = getattr(ci, "oid", "?"); url = getattr(ci, "commit_url", repo)
        api.upload_file(path_or_fileobj=card(a["title"], a["base"], a["recipe"]).encode(),
                        path_in_repo="README.md", repo_id=repo, commit_message="add model card")
        print(f"DONE {repo} PRIVATE sha={sha} url={url}", flush=True)
        results.append((repo, sha))
    except Exception as e:
        print(f"FAIL {repo}: {type(e).__name__}: {e}", flush=True)
        results.append((repo, f"FAIL:{type(e).__name__}"))
print("\n==== UPLOAD SUMMARY ====")
for r, s in results: print(f"  {r}  ->  {s}")
PY
echo "==== HF upload done $(date -Is) ===="
sleep 20

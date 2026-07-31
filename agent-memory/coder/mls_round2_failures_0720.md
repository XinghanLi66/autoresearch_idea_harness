# Round-2 eval failures → cc002 (env/data fixes needed before re-run)

**From:** cc001 · **Date:** 2026-07-20 · Source: failure-investigation subagent (evidence-based, pod logs + leaderboards)

Plain re-dispatch will deterministically fail for these — they need env/data fixes first. **No torch-leak
found** (checked). 3 "Untracked" actually succeeded (see bottom, no action).

## Still fully broken — ZERO valid metrics on 07-20 across EVERY arm (fix didn't take)
1. **mlsys-sparse-attention-inference** — data-missing: `FileNotFoundError: /newcpfs/lxh/MLS-Bench/vendor/data/longbench-qasper/qasper.jsonl` (+ `multifieldqa_en.jsonl`). Stage those.
2. **ai4sci-pla-binding-affinity** — env-broken: `DirectoryNotACondaEnvironmentError` for all 4 baselines (schnet/gign/ehign/egnn) — the target dirs exist but aren't conda envs. Rebuild them. Whole task empty (all arms).
3. **llm-pretrain-optimizer** — data-missing: `FAIL: piqa: Couldn't find a dataset script` + `ERROR: Checkpoint not found: .../lion/seed_1/ckpt_gpt-345m.pt`. Stage piqa (+ arc_easy/winogrande if same) and the baseline ckpt.
4. **robomimic-bc-loss** — broken-eval: 159 rows on 07-20 across all arms, **0** with non-empty `success_rate` (80 pre-0720 rows had values). Eval silently yields no metric — needs investigation.

## Baseline-only breakage — AGENT arms produce metrics, but BASELINE fails → no rescaling anchor (so agent scores can't normalize; they show 0.0)
5. **cv-3dgs-densification** — baseline gsplat CUDA build: `TypeError: _jit_compile() missing 2 required positional arguments 'verbose' and 'with_sycl'` (torch/gsplat API mismatch) + `RuntimeError: Error building extension 'gsplat_cuda'`; one scene `exit=137` (OOM). Agent arms OK (85/120 rows have metrics).
6. **cv-vae-loss** — baseline harness bug: `KeyError: 'LOCAL_RANK'` — baseline `train_medium.sh`/`train_large.sh` run as plain python, not torchrun. Agent arms OK (34/91 rows).
7. **quant-concept-drift** — baseline data-missing: `ValueError: instrument ... does not contain data for day` → qlib `~/.qlib/qlib_data/cn_data` absent. Agent arms OK.

## No action — Untracked but VERIFIED succeeded (fresh 07-20 is_final rows w/ real values)
- llm-ptq-algorithm base32bq3 s3 (ppl ~5.01) · rl-value-discrete sft14b s3 (cartpole 500.0) · llm-qat-algorithm sft32b s2 (wikitext2_ppl 11.62)

## After you fix each, ping me the slug and I re-dispatch (baseline + arms as needed).

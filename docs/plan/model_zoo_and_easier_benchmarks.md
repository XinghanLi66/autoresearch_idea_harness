# Plan (setting): Model Zoo × Easier Benchmarks × RL-v2

Branch: `V3`. Status: design locked with user (2026-07-17); execution starts after API keys are refreshed
(Phase 0 is key-independent and can start immediately).

## Motivation (grounded in the failure-mode investigation)
The V3 researcher-CoT SFT/RL did **not** beat base on the MLS closed loop, but the investigation showed the
worker is faithful and SFT already fixed proposal *form*. The two binding constraints were:
1. **Benchmark ceiling / variance** — thresholds ≈ baseline+0.5pp within single-seed 200-epoch noise; even a
   frontier(opus) proposer scored 0/4 on the subset. We were also evaluating only **10 of 179** MLS tasks,
   **all** ResNet-20/32/56 on CIFAR (the most-saturated, monoculture corner).
2. **Reward** — the RL reward was non-verifiable and mis-built (saturated format term = 30% no-gradient; KL k1
   sign flip; 16 rollouts). Absolute opus creativity scores saturate at 4–5; creativity RM Spearman ~0.33.

Two user directions address these: (A) a **model zoo** across size×series×structure for a real scientific
comparison of the base model, and (B) **easier / higher-power benchmarks** (MLAgentBench + MLS-lite).

## Infra ceiling (GB200 worker = 4 GPU × ~189 GB ≈ 756 GB; "use whatever is free")
| Model class | Full-FT (~16 B/param fp32 AdamW) | LoRA (bf16 weights only) |
|---|---|---|
| 8B–32B dense | ~1 worker (32B tight — proven) | « 1 worker |
| 70B dense | ~2 workers | 1 worker |
| Qwen3-235B-A22B MoE | ~5–6 workers | 1–2 workers |
| DeepSeek-V3 671B / Kimi-K2 1T | impractical | 2–4 workers (LoRA only) |
Each arm is sized to fit idle capacity at submit time (no fixed reservation).

## A. Model zoo (arms chosen by user; spans size × series × structure)
Controls held fixed: same anchored dataset (`v3_researcher_cot_anchored`, 1626/85), same eval stack, same worker.
Qwen3 dense (8/14/32B are the original hybrid-thinking release) trained/served **non-thinking**
(`enable_thinking=False`) to avoid `<think>` colliding with our Mocking/Core-idea/Non-trivial-crux anchors.

| ID | Model | Method | Fit | Recipe notes |
|---|---|---|---|---|
| S1 | Qwen3-8B | full-FT | 1 worker (L20Z LoRA pre-check) | lr 5e-6, seq 1664, 1–2 epoch, eff bsz 16 |
| S2 | Qwen3-14B | full-FT | 1 worker | lr 3e-6 |
| S3 | Qwen3-32B | full-FT | 1 worker (tight; empty_cache callback) | lr 2e-6, 1 epoch — **clean A/B vs Qwen2.5-32B** |
| M1 | Qwen3-30B-A3B (MoE) | LoRA | 1 worker | r64/α128, lr 1e-4, cheap active compute |
| M2 | Qwen3-235B-A22B (MoE) | LoRA | 1–2 workers | r32/α64, lr 5e-5 — scale-via-sparsity |
| D1 | latest trainable DeepSeek (R1-0528-Distill-Qwen, or V3 LoRA) | full-FT if ≤32B distill else LoRA | fit-dependent | reasoning-native `<think>` → reconcile format; confirm newest tag |

Axes isolated: dense ladder = **size**; +MoE arm = **structure**; +DeepSeek = **series**.
(Kimi-K2 dropped by user.) Orthogonal high-leverage option once keys return: regenerate CoT data with an
**opus-4.8/fable teacher** and retrain the best arm — better distillation data likely moves *novelty* more than
swapping the student.

## B. Benchmarks (locked: MLAgentBench primary + MLS-lite)
### B1. MLS-lite = official MLS-Bench-Lite 30 tasks, run via the native mlsbench CLI
- Use the **official 30-task MLS-Bench-Lite** subset (`docs/eval/mls_bench_lite_tasks.json`; 12 domains,
  ≥2 each) — vs our old 10 same-corner ResNet-CIFAR tasks. Source = GitHub `Imbernoulli/MLS-Bench` (local
  `/newcpfs/lxh/MLS-Bench`); HuggingFace is deprecated.
- **Eval engine = native `mlsbench` CLI** (not our bespoke worker, which was CV-only and is now deleted):
  `mlsbench baseline <slug> --seed S` + `mlsbench agent <slug> --model <worker> --mode eng
  --extra-context <our-proposal>`, scored on each task's `leaderboard.csv`. Driver:
  `scripts/run_mls_lite_eval.py` (+ helpers `scripts/_mls_lite_common.py`); orchestrator
  `scripts/ablation_failure_mode.sh` (stages gen-qs/frontier/judge/eval). Eval-env (26 docker images +
  data + provider routing) provisioned by cc002 (`agent-memory/coder/mls_lite_eval_env_request.md`).
- **Multi-seed n≥3** per proposal×task; report **normalized Δ-over-baseline + CI** (effect size), not knife-edge pass.
- `--free-hparams` worker by default. Extend `run_mls_eval_3arm.py` → multi-seed + Δ aggregation.
- Two compounding noise sources fixed: between-task (n: 10→~50) **and** within-task (single-seed→multi-seed + continuous Δ).

### B2. MLAgentBench (novelty-favorable primary)
- Port its 13 tasks; metric = **≥10% improvement over provided baseline** (open delta, real headroom, single-file
  edits map 1:1 to our worker). Task adapter → worker pipeline; same multi-seed + Δ reporting.
- (Optional stretch tier later: RE-Bench — purest novelty stress, but expensive; not SWE/MLE.)

### Cost
~50 tasks × arms × 3 seeds ≈ many ~20-min worker runs → parallelize worker eval on QS GPUs and/or shorten to
proxy runs (verify ranking preserved).

## C. RL-v2 (reward is a rubric, NOT verifiable → pairwise is the primitive)
Decision (2026-07-17): because the reward is non-verifiable and the reliable rubric primitive is a **pairwise
preference** (absolute scores saturate), prefer **DPO-family**, keep **GRPO only if fixed**, **drop PPO**.
1. **DPO → IPO (start here):** train on opus **pairwise** chosen≻rejected pairs; high-margin pairs only; hard
   validity gates (format + fingerprint pass on both sides). No rollouts / critic / KL-estimator — avoids the exact
   bugs that flattened our GRPO. Fallbacks: **KTO** (binary good/bad, robust to imbalance), **ORPO** (no ref model,
   memory win on 32B).
2. **GRPO-BT (on-policy comparison):** swap the saturated Ridge scalar for a **Bradley-Terry pairwise RM**; k3 KL;
   drop the saturated format term; ≥32 rollouts. Head-to-head vs DPO on the new benchmarks.
3. **PPO: skip** — adds a second ~32B critic on tight GB200 for sample-efficiency that doesn't address reward quality.

**Preference-data recipe (built via `scripts/build_v3_preference_pairs.py`):** chosen = gold anchored CoT;
negatives, all format-matched (differ only on novelty): **(1) trivialize** = opus-4.8 rewrite of the gold that
keeps persona/anchors/length but neuters the crux (margin-5 backbone); **(3) route** = within-(researcher,case)
route pairs judged pairwise by opus-4.8 (weak/saturated, min-conf 3, auxiliary); **(2) on-policy** = gold vs each
SFT checkpoint's own samples (per-arm, real margin). Same pairs feed DPO/IPO and the Bradley-Terry RM.
**Open direction (deferred, user 2026-07-17):** an arxiv/MLE-sourced "easy negative" slice — only if reformatted
into our template (raw text = genre shortcut); arxiv corpus available (7,441 papers), MLE reports not sourceable
locally; pairing has a setup/topic confound. Not injected for now.

## Sequencing
- **Phase 0 (no keys):** benchmark infra — audit 179, build MLS-lite selection + multi-seed Δ harness, port
  MLAgentBench. This is the binding-constraint fix; start now.
- **Phase 1:** model-zoo SFT, cheap→expensive as capacity frees: S1→S2→S3, then M1, M2, then D1.
- **Phase 2:** eval every base+SFT on MLAgentBench + MLS-lite (multi-seed Δ). Compare across size/series/structure.
- **Phase 3:** RL-v2 (DPO/IPO first, then GRPO-BT) on the best base.
- **Phase 4 (keys):** teacher upgrade (opus-4.8 data) + retrain best; frontier proposer ceiling with 4.8/fable.
- **Reporting:** fold into the two Chinese RedDocs (recipe/results doc + dataset/eval doc).

## Deliverables
Replicable QS scripts (stored locally + GitHub `V3`): `audit_mls_tasks.py`, MLS-lite multi-seed Δ harness,
MLAgentBench adapter, per-arm QS SFT prep, DPO/IPO + GRPO-BT trainers, updated `ablation_failure_mode.sh`.

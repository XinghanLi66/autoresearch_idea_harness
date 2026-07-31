# Proposal-Training Findings & Practitioner Takeaways
_Compiled by cc001 (benchmarker), 2026-07-30, for the arXiv report (cc003). Sources: `runs/researcher_cot/mls_lite_eval/report.json` (MLS-Bench-Lite, 21 arms × 30 tasks, rescaled 50=strong baseline) + `runs/researcher_cot/mab_eval/report.json` (MLAgentBench, 21 arms × 5 tasks, Δ-over-baseline %)._

## TL;DR
Proposal-training **works** — the evidence is not purely negative. A researcher-CoT-SFT'd **8B** reasoning model (DeepSeek-R1-0528-Qwen3-8B → `d1sft`) generates research proposals that, when implemented by a **fixed worker**, are the **#1 arm on MLS (40.4)** and **beat frontier-API proposers (GPT-5.5 31.4 / Claude-fable-5 32.0)** — and on MLAgentBench, training **rescues** a harmful base (`d1base` −21% → `d1rl` +19% mean Δ; a +40-point swing). The honest counterpart: proposal-SFT **helps reasoning-native/MoE bases but hurts instruct-tuned dense ones**, and on MLS most arms don't beat the strong anchor — so the method must be applied selectively.

---
## POSITIVE RESULTS (proposal-training is effective)

### P1 — Flagship: the DeepSeek-R1-8B SFT ladder (idea quality is *trainable*)
- MLS 30-task means: `d1base` **29.7** → `d1sft` **40.4** (**+10.7**, the single largest training gain of any family) → `d1rl` **36.2** (retains +6.5 over base).
- `d1sft` is the **top arm overall on MLS** (40.4), ahead of a 30× larger checkpoint (`m2base` 235B = 38.6) and every base Qwen3.
- **Beats frontier proposers head-to-head** under the identical fixed worker: `d1sft` vs best-of-{fable5,gpt55} = **14 wins / 10 losses / 6 ties**; mean **40.4 vs 32.0/31.4 (+8)**.
- Big concrete wins for the trained proposer's idea (rescaled pts over frontier): Unconditional Graph Generator **+75**, Diffusion Policy for Robot Control **+52**, Convolutional Activation Nonlinearity **+50**, Discrete Causal Graph Discovery **+41**, Value-Based Discrete Control **+20**, Nonlinear 2D Embedding **+20**.

### P2 — MAB: training *rescues* a harmful base into a top performer
- `d1` mean Δ ladder: `d1base` **−21.1%** → `d1sft` **+13.6%** → `d1rl` **+18.9%** (monotonic; **+40-pt** swing from training).
- Per-task rescue is dramatic: ogbn-arxiv `d1base` **−38.7%** → `d1rl` **+61.9%** (**~+100 pts**); cifar10 −3.4% → +22.7%.
- → RL(DPO) on preference data converts a base model whose raw proposals *actively hurt* into one of the strongest arms.

### P3 — Trained-proposer ≥ frontier-proposer (the fixed-worker ceiling test)
- MLS top-8 arms are **all trained/base checkpoints**, every one above both frontier proposers: `d1sft` 40.4, `base32bq3` 39.6, `m2base` 38.6, `base14b` 38.4, `base32b` 37.8, `m2sft` 36.4, `sft32b` 36.2, `d1rl` 36.2 — vs `fable5` 32.0 / `gpt55` 31.4.
- → With the worker held fixed, the bottleneck is **idea generation tuned to the task distribution**, not the proposer's raw scale. Frontier scale ≠ better research ideas here.

### P4 — MAB: broad, consistent positive Δ; trained arms lead
- 17/21 arms have positive mean Δ. Leaders are trained checkpoints: `sft14b` +21.6% (3/5), `qwen3-14b-rl` +19.5% (3/5), `d1rl` +18.9%, `sft32b` +17.8%, `m2rl` +16.3% (3/5). Frontier `fable5` +14.3% / `gpt55` +8.2% sit mid-pack again (cross-benchmark-consistent with P3).
- On MAB, **SFT helps**: `base14b` +16.3% → `sft14b` +21.6% (+5.3 from SFT).

### P5 — Native no-proposal agent is benchmark-dependent (scaffolding pays off by task type)
- `purefable5` (native fable-5, NO master proposal) is **mid-pack on MLS (35.2)** but **dominant on MAB (+49.8%, 4/5)**.
- → A master proposal helps most on **from-scratch/novel-design** tasks (MLS); on **incremental-improvement-of-existing-code** tasks (MAB) a strong native agent already excels.

---
## HONEST NEGATIVES (keep the story balanced, not purely negative)
- **Proposal-SFT hurts instruct-tuned dense bases on MLS from-scratch tasks:** Qwen3-8B 35.0→29.0 (−6.0), Qwen3-14B 38.4→32.5 (−5.9), Qwen3-32B 39.6→36.2 (−3.4), Qwen2.5-32B 37.8→33.5 (−4.3). It **helps** reasoning-native (DeepSeek +10.7) and is ~neutral on MoE (235B).
- **Most MLS arms score <50** (rarely beat the strong rescaled baseline) — implementing a research idea from scratch against a *tuned* anchor is genuinely hard; this is the conservative benchmark.
- `d1sft` loses to frontier where its idea misfires (Latent World-Model Planner, Frequency-Aware AE = faithful-0) — training raises the mean, not every cell.

---
## HIGH-LEVEL TAKEAWAYS FOR PRACTITIONERS
1. **Idea quality is trainable, not just scale-bought.** Researcher-CoT SFT on a reasoning-native 8B yields proposals beating GPT-5.5/fable-5 under identical implementation. You do not need a frontier model to generate strong ML-research ideas.
2. **Match the training to the base model.** Proposal-SFT helps reasoning-native (DeepSeek-R1) and MoE bases; it hurts instruct-tuned dense models. Choose the base accordingly, or the same recipe backfires.
3. **Use RL (DPO) as a stabilizer/rescuer.** It gives monotonic gains on high-headroom tasks (MAB d1 base→sft→rl) and recovers when SFT overshoots (Qwen3-8B MLS 29.0→33.9).
4. **The payoff of a master proposal depends on task type.** Biggest gains on novel-design/from-scratch tasks (MLS); on incremental-improvement tasks (MAB) a strong native agent can already dominate. Deploy proposal-scaffolding where design novelty matters.
5. **Evaluate against BOTH strong anchors and weak baselines.** MLS (strong anchor, 50=baseline) exposes that most methods don't beat a tuned baseline; MAB (weak baseline) reveals real headroom and large training gains. Using only one over- or under-claims the method.

## TL;DR

<redoc-highlight emoji="dengpao" fillColor="orange">
把 V3 的 32B **base → SFT → RL** 三个 checkpoint 放进 **V2 worker-MLS 闭环**（worker = Claude Code 实现 proposal，真跑 200-epoch 训练，回签 metric vs 阈值）评测，并做 failure-mode 消融。**核心问题：SFT/RL 在 MLS pass-rate 上没有超过 base。** 本文给出 recipe、结果、逐环节 buggy 判定与解决方案。
</redoc-highlight>

<redoc-highlight emoji="dui" fillColor="green">
**好消息（不是 bug）：** (1) worker 100% 忠实执行——30/30 run 都改了 edit region 并完整跑完 200 epoch；(2) SFT 显著提升了 proposal 的 **形式质量**（specificity/implementability/meaningful 相对 base 大涨）。
</redoc-highlight>

<redoc-highlight emoji="cuo" fillColor="red">
**坏消息（负面/限制）：** (1) 训练没有提升 proposal 的 **novelty / 有效性**；(2) **RL reward 是坏的**（format 项已饱和 → 30% 无梯度；KL 用 k1 估计器符号跑负）；(3) 更关键——**benchmark ceiling 主导**：连 frontier(opus-4-6) 的高分 proposal 在 MLS 子集上也 0/4，阈值≈baseline+0.5pp 落在单 seed 噪声内。
</redoc-highlight>

## 1. 训练 recipe（SFT + RL）

### 1a. Full-param SFT（Qwen2.5-32B-Instruct）

| 项 | 值 |
| --- | --- |
| base | Qwen2.5-32B-Instruct |
| 方式 | 全参 FSDP（full_shard + activation checkpointing） |
| 数据 | anchored judged 集 1,626 train / 85 val（sha256 校验） |
| epoch / steps | 1 epoch = 101 steps |
| lr / scheduler | 2e-6 / cosine，warmup 0.03，wd 0.01 |
| batch | per-device 1 × grad-accum 4 × 4 GPU = 16 |
| seq_len | 1664 |
| 精度 | bf16 + gradient checkpointing |
| 用时 / 显存 | ~17 min（4×GB200）/ loss 2.59→1.89，eval 1.92 |

<redoc-highlight emoji="cuo" fillColor="red">
**OOM 限制：** 2-epoch 的 3 次尝试都在 **epoch-2 边界 step 101** OOM（~175/184GB，sharded fp32 AdamW 主导），故最终只训 **1 epoch**。2-epoch 需换 Adafactor/8-bit optimizer 或 2 worker。
</redoc-highlight>

### 1b. RL（GRPO，composite reward）

| 项 | 值 |
| --- | --- |
| policy | 冻结的 full-SFT 32B + 可训 LoRA（r64/α128） |
| reference（KL） | 同 model 关掉 adapter（零额外显存） |
| reward | **R = 0.4·fingerprint + 0.3·format + 0.3·creativity** |
| fingerprint RM | 125-way guess-who（MiniLM emb + LogReg），held-out top1 68.9% / top5 89.9% |
| creativity RM | Ridge on opus 分数，Spearman ~0.33（弱） |
| steps / rollouts | 60 步 × 16 rollout（4 prompt × group 4） |
| kl / lr | 0.05（k1 估计器）/ 1e-5 |
| 用时 | ~4.5 h（单 GB200） |

<redoc-highlight emoji="cuo" fillColor="red">
**RL reward flat（负面）：** first-5 mean 0.5814 → last-5 0.5789（**没涨**）。原因：format 项 SFT 后已饱和到 1.0（占 30% 权重却无梯度）；KL 的 k1 估计器跑到 −0.13（相当于奖励发散）；rollout 只有 16（advantage 噪声大）；步数少。
</redoc-highlight>

## 2. MLS 闭环评测结果（10 tasks × 3 arms）

worker 忠实实现每个 proposal 并真跑 200-epoch 训练，回签 metric。higher = better，pass = metric ≥ 阈值。

| arm | pass | 说明 |
| --- | --- | --- |
| **base**（训练前） | **2 / 10** | cv_pooling_aggregation、dl_residual_connection |
| SFT（checkpoint-101） | 1 / 10 | dl_regularization（且 cv_multitask_loss 退化到 25.89） |
| RL（+GRPO adapter） | 1 / 10 | cv_pooling_aggregation |

mean metric：base 80.28 · SFT 75.76 · RL 80.05。对照旧 V2.5：base 1/10，old-sft 0/10。

<redoc-highlight emoji="cuo" fillColor="red">
**负面结论：** researcher-CoT 的 SFT/RL **没有** 在 MLS 上超过 base；SFT 因一次退化 run（learned log-variance 权重把 model 训崩到 25.89）反而略降。
</redoc-highlight>

## 3. Failure-mode 调查（逐环节 buggy 判定）

沿因果链 proposal → worker 实现 → metric 逐段消融。

### 3a. 因果链 & 消融矩阵（4-task 子集）

| task | 阈值 | base | sft | rl | frontier(opus-4-6) | rl + free-hparams |
| --- | --- | --- | --- | --- | --- | --- |
| dl_lr_schedule | 93.19 | 92.65 | 92.8 | 93.08 | 92.15 | 92.89 |
| cv_pooling_aggregation | 72.19 | 72.64✅ | 71.75 | 72.27✅ | 71.9 | 71.1 |
| cv_multitask_loss | 68.79 | 68.67 | 25.89 | 68.51 | 68.6 | 68.88✅ |
| dl_residual_connection | 92.96 | 93.12✅ | 92.28 | 92.83 | 91.68 | 92.53 |
| **子集 pass** | <br/> | **2/4** | 0/4 | 1/4 | **0/4** | 1/4 |

### 3b. Proposal 质量 rubric（opus-4-6 judge，1–5）

| arm | novelty | specificity | implementability | effectiveness | correctness | meaningful |
| --- | --- | --- | --- | --- | --- | --- |
| base | 2.1 | 2.6 | 2.7 | 1.8 | 2.4 | 0.30 |
| **SFT** | 2.4 | **3.8** | **4.2** | 2.5 | 3.3 | **0.80** |
| RL | 2.4 | 3.5 | 3.8 | 2.5 | 2.8 | 0.70 |
| frontier | 2.7 | 4.9 | 4.9 | 3.2 | 4.0 | 1.00 |

### 3c. 逐环节判定

| 环节 | 是否是 bug | 证据 |
| --- | --- | --- |
| **Worker 执行** | 🟢 **不是** | 30/30 run 都改 edit-region 且完整跑完 200 epoch；抽查 RL/dl_lr_schedule 的 get_lr 与 proposal 逐条对应 |
| **Worker prompt 太严** | 🟠 次要 | --free-hparams 子集 1/4（比 rl-constrained 略好，翻过 cv_multitask_loss），但非决定性 |
| **Worker model 太弱** | 🟠 未直接换（key 受限） | 但 worker 已忠实实现、且 frontier proposal 也不 work → worker-model 不太可能是主因 |
| **Proposal 形式质量** | 🟢 **已被 SFT 修好** | specificity 2.6→3.8、implementability 2.7→4.2、meaningful 0.3→0.8 |
| **Proposal novelty/有效性** | 🔴 训练没提升 | base→sft/rl 的 novelty 平（~2.4）、effectiveness 平（~2.5） |
| **RL reward** | 🔴 **是 bug** | reward flat；format 饱和无梯度；KL k1 符号错 |
| **数据 creativity 信号** | 🔴 弱/饱和 | opus judge 几乎全给 4–5（creativity RM Spearman 仅 0.33） |
| **Eval / benchmark** | 🔴 **主导限制** | frontier 高分 proposal 仍 0/4；阈值≈baseline+0.5pp，落在单 seed 200-epoch 噪声内；各 arm 差异 &lt; 1pp |

<redoc-highlight emoji="dengpao" fillColor="orange">
**最关键的一条：** 把 proposal 换成 frontier(opus-4-6)——rubric 分数高得多（novelty 2.7、specificity 4.9、effectiveness 3.2）——经同一 constrained worker 后 **MLS 子集仍 0/4，甚至 3 个 task 低于 base**。说明"proposal 不够好"**不是** MLS 不过的充分原因；**benchmark 太紧 + 单 seed 方差**才是主导。
</redoc-highlight>

## 4. 结论 & 解决方案

<redoc-highlight emoji="dui" fillColor="green">
**判定：worker 与 proposal-形式 环节健康；bug 集中在 RL reward、creativity 数据信号，而 MLS 评测本身的 ceiling/方差是当前最大限制。**
</redoc-highlight>

| 环节 | 解决方案 |
| --- | --- |
| RL reward | KL 改 **k3** 估计器；**去掉/降权** 饱和的 format 项；creativity 改 **pairwise / Bradley-Terry**（绝对分饱和）；加 novelty-via-dissimilarity；rollout 16→≥32；步数加大 |
| 数据 creativity 信号 | 用 **pairwise** creativity 标注（不饱和）；挖更难/更分化的 case |
| Eval / benchmark（最高优先） | 每 proposal **多 seed（n≥3）** 压方差；每 task **多 proposal（k≥3）**；把 --free-hparams 设为默认；报告 **相对 baseline 的 Δ + 置信区间**，而非贴近 baseline 的硬阈值 pass/fail；或换 headroom 更大的 task |
| Recipe / model | 先修 reward+eval（当前是 binding constraint）；再考虑 2-epoch（换 optimizer 解 OOM）/更大 model |

## 5. 可复现

- 消融编排脚本：scripts/ablation_failure_mode.sh（stage：gen-qs / frontier / judge / worker / worker-freehp），QS 出 proposal + 本地跑 worker。

- 关键脚本：generate_mls_proposals_3arm.py、judge_mls_proposals.py、run_mls_eval_3arm.py（--free-hparams）、verify_v3_qs_checkpoint.py。

- 数据 receipts：runs/researcher_cot/mls_eval/（proposals、proposal_judge、worker_runs、worker_runs_freehp、ablation_subset.json）。均在分支 **V3**（GitHub 同步）。
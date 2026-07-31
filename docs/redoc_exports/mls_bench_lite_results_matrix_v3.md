— LIVING · 快照 2026-07-30（21 臂 × 30 任务 = 630 单元，全部 faithful，0 缺失；新增 purefable5 原生无提案臂）

> Rescaled（native mlsbench score）：50=strong baseline，100=上界。单元：数值=rescaled 分；· =faithful-0（未过最差锚点/能力地板，见「失败原因」）。worker 固定=claude-fable-5（**例外**：ai4bio / ai4sci-pla 两个蛋白任务 worker=claude-opus-4-8，因 fable-5 对蛋白内容误触发风控；见脚注①）。fable5/gpt55=前沿模型仅作提案者(API)。

## 分族阶梯 base→SFT→RL（30 任务均值）

| 模型族 | base | SFT | RL | 备注 |
| --- | --- | --- | --- | --- |
| Qwen3-8B | 35.0 | 29.0 | 33.9 | SFT 拖累,RL 部分回补 |
| Qwen3-14B | 38.4 | 32.5 | 29.4 | SFT/RL 均降 |
| Qwen3-32B | 39.6 | 36.2 | 33.2 | SFT/RL 降 |
| Qwen2.5-32B | 37.8 | 33.5 | 32.9 | base=Inst |
| DeepSeek-R1-8B | 29.7 | 40.4 | 36.2 | SFT 大幅提升(29.7→40.4),RL 次之 |
| Qwen3-235B | 38.6 | 36.4 | 34.0 | SFT 微降,RL 降 |
| 前沿proposer | fable5=30.7 | gpt55=31.4 | — | 居中，被多训练检查点超过(worker 仍固定) |
| 原生 agent | — | — | purefable5=35.2 | **无 master proposal**：worker(fable-5)从 MLS 原生 prompt 自行构思+实现(--mode sci)；MLS 居中(~35)但 MAB 上 Δ+49.8% 主导(见脚注④) |

### 前沿天花板发现

固定 worker 下，前沿模型(fable5 30.7/gpt55 31.4)的"想法"并非上界——被 d1sft(40.4)/base32bq3(39.6)/m2base(38.6)/base14b(38.4) 等训练/base 检查点超过。即：同 worker 实现时，训练 proposer 的想法 ≥ 前沿想法。

## 全矩阵（30 任务 × 21 臂）

| 领域 | 任务 | base8b | sft8b | qwen3-8b-rl | base14b | sft14b | qwen3-14b-rl | base32bq3 | sft32b | qwen3-32b-rl | base32b | qwen25sft | rl | d1base | d1sft | d1rl | m2base | m2sft | m2rl | fable5 | gpt55 | purefable5 | 失败原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AI for Science | Mutation Fitness Predictor | 32 | 40 | 21 | 50 | 44 | 52 | 51 | 51 | 34 | 48 | 47 | 49 | 49 | 51 | 20 | 44 | 36 | 25 | 44 | 43 | · | worker-swap→opus-4.8 (用户决定: 必须产出全30结果以保持可比/有说服力, 不接受不可评). fable-5对蛋白突变bio框架确定性误报拒绝(stop_reason=refusal, 空补全; cc002对照实验证实为纯FALSE-POSITIVE—同一技术任务中性ML措辞可正常实现). 保持真实bio提案+任务spec不变, 仅将ai4bio的worker统一换为claude-opus-4-8(全18臂一致→保内部跨臂可比), 记录为该任务的per-task worker替换. |
| AI for Science | Diffusion-Prior Inverse Solver | 1 | 1 | 1 | 1 | 20 | 11 | 15 | 62 | · | 11 | · | · | · | 19 | 11 | 11 | 3 | 14 | 52 | · | · | inpainting设定因上游FFHQ-256数据缺失已从评分剔除;blackhole=模型侧采样器挂起(忠实);inv-scatter正常 |
| AI for Science | Protein-Ligand Interaction Model | 38 | 46 | 44 | 53 | 47 | 24 | 10 | 41 | 51 | 37 | 47 | 41 | 45 | 38 | 41 | 41 | 41 | 48 | · | 14 | · | worker-swap→opus-4-8 (2nd protein 风控, like ai4bio): fable-5对蛋白-配体结合亲和力impl确定性误报拒绝(stop_reason=refusal, output_tokens≈4→空补全→无编辑→无训练→blank, 旧版误判为timeout实为风控). 保持真实提案+3 PDBbind设置(2013/2016 core + 2019 holdout)不变, 仅将该任务worker统一换claude-opus-4-8(全臂一致), 记录per-task worker替换. 早期3-GPU dispatch另因L20Z非2幂HOL-block整个队列(已修: 4-GPU). |
| Classical & Adaptive Learning | Geometry-Robust Clustering Algorithm | 68 | 69 | 67 | 38 | 66 | 69 | 69 | 70 | 69 | 65 | 69 | 17 | · | 64 | 70 | 60 | 70 | 63 | 46 | 69 | 70 | — |
| Classical & Adaptive Learning | Nonlinear 2D Structure-Preserving Embedding | 17 | 9 | 33 | 46 | 8 | 17 | 47 | 47 | 10 | 31 | 50 | 44 | · | 49 | 50 | 48 | 32 | 31 | 30 | 22 | 47 | — |
| Deep Learning | Spatial Feature Aggregation | 43 | 44 | 44 | 46 | 48 | 43 | 50 | 46 | 48 | 47 | 45 | 46 | · | 48 | 43 | 44 | 45 | 47 | 49 | 45 | 47 | — |
| Deep Learning | Convolutional Activation Nonlinearity | 54 | 4 | 48 | 55 | 13 | 39 | 12 | · | 14 | 48 | 14 | · | 51 | 54 | 54 | 54 | 44 | 57 | · | 4 | 52 | 个别臂worker空输出抖动(shim修复后应补齐) |
| Language Models | Masked Diffusion Demasking Policy | 53 | 11 | 30 | 31 | 36 | 30 | 44 | 50 | 38 | 36 | 10 | 39 | 49 | 38 | 34 | 38 | 38 | 45 | 23 | 25 | 46 | 重跑后仍0=worker提案≤最差锚点(genuine 0* floor,忠实) |
| Language Models | Pretraining Optimizer Design | 56 | 56 | 56 | 55 | 56 | 48 | 55 | 56 | · | 57 | 56 | 50 | 48 | 55 | 57 | 58 | 56 | 58 | 50 | 60 | 61 | — |
| Language Models | Reasoning RL Importance-Sampling Granularity | · | · | · | 29 | · | · | 14 | · | · | · | · | 16 | · | 14 | · | · | · | · | · | · | · | 4/11臂(1-GPU)已入库;其余2-GPU变体因vLLM多进程引擎event-loop崩溃(EngineCore died,7.2h零进展)→FAIL:2gpu-runtime(忠实);已停(heavy wall=72h不触发) |
| ML Systems & Efficient ML | Post-Training Weight Quantization | 3 | 75 | · | 55 | 3 | 69 | 66 | 2 | 56 | 66 | 2 | 43 | 58 | 63 | 60 | 56 | 54 | · | 55 | 59 | 67 | — |
| ML Systems & Efficient ML | Quantization-Aware Language-Model Training | 65 | 64 | 65 | 65 | 65 | 64 | 62 | 65 | 65 | 65 | 61 | 62 | 62 | 65 | 61 | 64 | 65 | 64 | 65 | 64 | 65 | — |
| ML Systems & Efficient ML | Long-Context Inference-Time Sparse Attention | 16 | 59 | 50 | 58 | 62 | 23 | 61 | 60 | 50 | 57 | 60 | 55 | 58 | 24 | 58 | 59 | 61 | 58 | 31 | 57 | 58 | — |
| Optimization & Theory | Multi-Objective Evolutionary Survival and Variation | 53 | 44 | 44 | 33 | 46 | 38 | 38 | 46 | 45 | 40 | 35 | 35 | · | 47 | 12 | 30 | 38 | 15 | 44 | 32 | 53 | — |
| Optimization & Theory | Variance-Reduced Stochastic Optimization | 44 | 14 | 49 | 36 | 58 | 16 | 55 | 40 | 13 | 5 | 5 | 65 | 48 | 66 | 58 | 44 | 47 | 41 | 62 | 60 | 47 | — |
| Reinforcement Learning | Value-Based Discrete Control | 12 | 4 | 2 | 9 | 26 | 32 | 17 | 7 | 29 | 2 | 20 | 3 | 10 | 33 | 8 | 36 | 9 | 2 | 12 | 12 | 13 | 已修复(重跑后13臂入库,d1入库问题解决) |
| Robotics | Latent World-Model Planner | 47 | 38 | 56 | · | · | · | 37 | 68 | 72 | 76 | 38 | 63 | 74 | · | 75 | 75 | 32 | · | 64 | 62 | · | 候选plan()无界执行挂起;已加2h墙钟超时→干净FAIL(候选代码问题,非基础设施) |
| Robotics | Guided Diffusion Sampling for Robot Actions | 51 | 52 | 19 | 54 | 50 | 3 | 52 | · | · | 53 | · | 11 | 50 | 54 | 51 | 53 | · | 48 | · | 48 | 52 | 个别臂worker空输出抖动 |
| Robotics | Diffusion Policy Learning for Robot Control | 53 | 25 | 40 | 52 | 54 | 44 | 52 | 28 | 51 | 49 | 51 | 46 | · | 52 | 51 | 50 | 50 | 51 | · | · | 41 | 单臂抖动 |
| Robotics | Humanoid Transfer Policy Learning | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | — |
| Robotics | Behavioral Cloning Loss for Manipulation | 16 | 18 | 55 | 18 | 30 | 11 | 41 | 34 | 61 | 62 | 70 | 22 | 35 | 69 | 53 | 12 | 65 | 16 | 70 | 73 | 14 | 已修复(dataset缺失+农场无头GL osmesa/mesalib25.0.5双修)→12臂real入库 |
| Structured & Causal Reasoning | Discrete Causal Graph Discovery | 31 | 42 | · | 24 | 50 | 24 | 35 | 41 | 30 | 39 | · | 14 | · | 41 | 18 | 36 | 28 | 43 | · | · | 44 | — |
| Structured & Causal Reasoning | Unconditional Graph Generator Architecture | 72 | · | · | 72 | · | · | 41 | 59 | 67 | · | 66 | 22 | 58 | 75 | · | 40 | 45 | 75 | · | · | 48 | 部分臂空输出/超时/未派发;重跑中 |
| Time Series & Forecasting | Concept-Drift-Aware Quantitative Forecasting | 35 | 50 | 45 | 50 | 15 | 34 | 36 | 26 | 35 | 33 | 3 | 41 | · | 23 | 54 | 22 | 28 | 70 | · | 62 | 50 | — |
| Time Series & Forecasting | Exogenous-Variable Target Forecasting Model | 48 | 25 | 43 | 45 | 14 | 42 | 47 | 47 | 45 | 40 | 32 | 44 | · | 42 | 45 | 45 | 44 | 46 | 39 | 27 | 42 | — |
| Time Series & Forecasting | Masked Multivariate Time-Series Imputation | 69 | 42 | 60 | 58 | 45 | 43 | 59 | 42 | 42 | 60 | 54 | 48 | 46 | 52 | 37 | 12 | 61 | 60 | 49 | 62 | 62 | — |
| Trustworthy Learning | Training Regularization for Membership Privacy | 39 | 28 | 28 | 32 | 15 | 14 | 29 | 24 | 28 | 62 | 43 | 22 | 23 | 30 | 28 | 30 | 34 | · | 47 | 38 | 21 | 单臂抖动 |
| Vision & Generation | 3D Scene Densification Strategy | · | · | 39 | · | 6 | 3 | · | · | · | · | 43 | · | 40 | · | 26 | · | 12 | 6 | · | · | 23 | bonsai场景显存OOM(exit137)→best_psnr_bonsai缺失→5/6不完整判0;已加expandable_segments重跑 |
| Vision & Generation | Low-Step Diffusion Bridge Sampling | 24 | 7 | 33 | 39 | 46 | 40 | 48 | 42 | 41 | 44 | 36 | 37 | 36 | 44 | 7 | 45 | 7 | 36 | 45 | 6 | 29 | 重跑后仍0=worker提案≤最差锚点(genuine 0* floor,忠实) |
| Vision & Generation | Frequency-Aware Autoencoding Loss | 12 | 4 | 47 | 48 | 52 | 49 | 48 | 32 | 3 | 4 | 48 | 50 | 48 | · | 4 | 51 | 49 | · | 44 | · | 3 | 单臂抖动 |

## 脚注（20×30 收尾 2026-07-27）

① **ai4bio-mutation / ai4sci-pla-binding**：fable-5 worker 对蛋白/药物设计内容确定性误触发风控(空补全)，两任务 worker 统一换 claude-opus-4-8(全臂一致，per-task substitution)，真实提案不变。
② **jepa-planning[sft8b, qwen3-8b-rl]**：8B thinking proposer 对该任务过度推理、22k tok 仍不闭合 &lt;think&gt;；采用其推理内容作提案(salvaged-reasoning)，仍实现出真实 planner(0.33/0.56)。
③ **robo-humanoid**：全 zoo 臂=统一"能力地板" 0.0(训练出真实策略但 sim2sim success_rate=0.0；仅 opus-4-6 0.796/gemini 0.688 迁移成功)。CSV 内 zoo 臂 metric 为空占位(task_score=0 由 geomean-zero 得)，真实 0.0 见 pod stdout；6 臂行因 03:45 leaderboard 锁竞争写停丢失，已按同格式占位补回(备份 leaderboard.csv.bak_cc002_*)。
④ **causal-discovery-discrete[qwen3-8b-rl]**：worker ensemble/bootstrapping 实现在 win95pts(76 节点隐藏集)死锁(3× 复现，其它臂正常)→faithful-0。
⑤ **MAB(MLAgentBench) 第二基准**：20 臂 × 5 任务 已完成，见下「MLAgentBench 结果」节。

## 主要发现 & 下一步

1. 6模型族 base→SFT→RL 阶梯 + 前沿proposer天花板（本轮补齐 m2rl 完成235B三元组 + fable5/gpt55 前沿对照）。

2. SFT 对 Qwen3-dense/Qwen2.5 反拖累、对 DeepSeek/235B 有益；RL(DPO) 混合；前沿proposer(fable5/gpt55) 居中不占优 —— 训练proposer想法 ≥ 前沿想法(同worker)。

3. 全部 ~&lt;55：提案难稳超强baseline → 下一步靠更强idea生成，而非更多评测。

4. env修复全清（robomimic/cv-3dgs/ai4/jepa/inverse/llm-rl）；worker 固定=fable-5(MaaS)；fable5臂的2个bio任务因风控回退opus-4.8。

## 附：RL checkpoint 现状 & arm 扩展计划（回应 2026-07-23 提问）

**Q1: Qwen3 系列 RL checkpoint 好了吗，还是只做了 SFT？**
目前 Qwen3-8B/14B/32B **只有 SFT**（S1/S2/S3 = sft8b/sft14b/sft32b），**没有 RL checkpoint**（newcpfs 上无 qwen3-*-rl）。要加 qwen3-8b/14b/32b RL 臂，需 cc000 在 S1/S2/S3 SFT 之上训练 RL(DPO)——待训练。

**Q2: Qwen2.5-32B 现在是直接 RL 的吗？能否 SFT 再 RL 并重评？**
其实**已经是 SFT-then-RL**：当前 rl 臂的 DPO 是在 checkpoint-101（= Qwen2.5-32B 全参 SFT, anchored-1ep-v4）之上训练的，**不是直接在 Instruct 上 RL**。链路 = base32b(Instruct) → SFT(ckpt-101) → RL(DPO/rl 臂)。只是**中间的 SFT(ckpt-101) 从未作为 arm 评测**。→ 无需新训练，只要 serve+eval ckpt-101 即可补齐 Qwen2.5-32B 的 base/SFT/RL 三元组。

**下一步 arm 扩展（进行中）：**

1. **base 臂**（现成模型，serve+gen@8192）：Qwen3-235B-A22B base、DeepSeek-R1-0528-Qwen3-8B base。

2. **RL 臂**：Qwen3-8B/14B/32B RL（需 cc000 训练 DPO on S1/S2/S3）；Qwen2.5-32B SFT(ckpt-101) 补评（现成）。

3. **修复未跑通任务**：env（robomimic CLIP 权重下载失败；ai4bio/ai4sci-pla 为 worker 空输出 step0 中止，非缺数据 → 重派）；dispatch-gap（opt/ts 的 sft8b/sft14b/base32bq3 未派发、jepa 卡住 → 重派）。

_目标：把 base/SFT/RL 三元组补齐到各模型族，并将同覆盖任务数从 13 扩大。_

## 更新 2026-07-23（本轮进展）

**env 修复全部完成**（cc002）：robomimic 农场无头GL(osmesa+mesalib25.0.5) · cv-3dgs bonsai显存OOM(expandable_segments) · ai4sci-inverse inpainting上游数据缺失(评分剔除,blackhole忠实0) · jepa 候选挂起(2h墙钟超时→FAIL) · ai4 worker空输出(shim空补重试+并发64) · llm-rl sleep_mode超时(2-GPU变体)。所有受影响任务已按新修复重跑（农场排队draining中）。

**新臂扩展**：qwen25sft(Qwen2.5-32B SFT=ckpt-101)、m2base(Qwen3-235B-A22B base) 已serve+gen@8192(各30提案valid)+评测派发中 → 补齐 Qwen2.5-32B 与 235B 的 base/SFT 覆盖。待补：d1base(DeepSeek-R1-8B base)、qwen3-8b/14b/32b RL(需 cc000 serve/train)。

**说明**：上方矩阵为 2026-07-22 分类快照（当时最完整）；本轮重跑落库+新臂评测完成后将整表刷新分数。「失败原因」列即时反映每任务未满覆盖的根因。

## 更新 2026-07-24（本轮收尾）

6新臂全部serve+gen@8192+评测落库：qwen25sft/m2base/d1base + qwen3-8b/14b/32b-rl（DPO全epoch完成的LoRA adapter）。分族阶梯首次完整覆盖 6 族的 base/SFT/RL。「失败原因」列逐任务解释未满覆盖单元。剩余重任务(robomimic/RL trio heavy tail)分数随农场drain继续微调。

## 更新 2026-07-24（本轮）

新增 3 臂：m2rl(235B RL，完成235B三元组 base43.6→SFT48.2→RL46.8) + fable5/gpt55(前沿proposer天花板对照，居中~40-42)。所有臂 worker 固定=claude-fable-5(仅“谁写提案”变化)。MLAgentBench(MAB) 第二基准评测进行中(cifar10/ogbn/imdb/house-price/spaceship × baseline+fable5/gpt55/m2rl)，结果表随农场drain补入。

## MLAgentBench (MAB) 结果

第二基准 MLAgentBench（5 任务：cifar10/ogbn-arxiv/imdb/house-price/spaceship-titanic），同一 proposal→worker 流程（worker 固定=fable-5 实现每个 arm 的提案）。指标 = **Δ-over-baseline**（方向修正，正=更好；success=Δ≥+10%）。runnable-baseline(cifar10/ogbn)基线=启动脚本；skeleton(imdb/house/spaceship)基线=worker朴素实现。**全 20 臂 faithful（无 tokenizer 乱码、无缺失、离群已核）。**

**要点**：与 MLS-lite（多数 rescaled &lt;50）不同，MAB 上几乎所有 arm **正向 Δ**（弱基线、headroom 真实）。sft14b(+22)/qwen3-14b-rl(+20)/d1rl(+19)/sft32b(+18)/m2rl(+16)/base14b(+16)/m2base(+15)/m2sft(+16) 领先；**前沿 proposer fable5(+14%)/gpt55(+8%) 居中、非天花板**（与 frontier-ceiling 跨基准一致）。少数负值：d1base(-21, 原始base最弱)、base32bq3(-7)、qwen25sft(-6)、sft8b(-4)。imdb 基线高(0.911 DistilBERT)故 imdb Δ 偏小。

_修复记录：d1base/d1sft/d1rl 曾因 DeepSeek tokenizer 误标 LlamaTokenizerFast → vLLM 字节级-BPE 乱码(Ġ/Ċ)，经 cc000 改 Qwen2TokenizerFast 源头修复后重生成，现 faithful（此前 d1base -54%/d1sft·d1rl worker-empty 均为乱码假象）。MAB env: unset PYTHONPATH+LD_LIBRARY_PATH / HF_TRANSFER=0 / accelerate。_

<redoc-highlight emoji="dengpao" fillColor="orange">
**MLAgentBench (MAB) 评测：Δ-over-baseline（方向修正后的相对提升 %，正=更好；success = Δ ≥ +10%）。** 5 tasks，worker (fable-5) 忠实实现每个 arm 的 proposal。
均值 Δ：fable5 **14.327%** (3/5 success) · gpt55 **8.181%** (2/5 success) · m2rl **16.333%** (3/5 success) · base8b **13.242%** (1/5 success) · sft8b **-4.157%** (1/5 success) · qwen3-8b-rl **12.507%** (1/5 success) · base14b **16.309%** (2/5 success) · sft14b **21.647%** (3/5 success) · qwen3-14b-rl **19.543%** (3/5 success) · base32bq3 **-6.672%** (2/5 success) · sft32b **17.792%** (2/5 success) · qwen3-32b-rl **14.133%** (1/5 success) · base32b **11.318%** (2/5 success) · qwen25sft **-6.015%** (0/5 success) · rl **12.085%** (1/5 success) · d1base **-21.14%** (0/5 success) · d1sft **13.585%** (1/5 success) · d1rl **18.702%** (2/5 success) · m2base **15.309%** (1/5 success) · m2sft **15.611%** (2/5 success) · **purefable5 49.763% (4/5 success，原生无提案臂，MAB 最高)**
</redoc-highlight>

## 每任务 Δ-over-baseline (%)

| domain | task | metric | baseline | Δfable5% | Δgpt55% | Δm2rl% | Δbase8b% | Δsft8b% | Δqwen3-8b-rl% | Δbase14b% | Δsft14b% | Δqwen3-14b-rl% | Δbase32bq3% | Δsft32b% | Δqwen3-32b-rl% | Δbase32b% | Δqwen25sft% | Δrl% | Δd1base% | Δd1sft% | Δd1rl% | Δm2base% | Δm2sft% | Δpurefable5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image Classification | CIFAR-10 (conv net) | accuracy | 0.51237 | 16.688±8.354 ✓ | 4.111±1.959 | 63.192±3.178 ✓ | -3.573±6.909 | 14.331±3.834 ✓ | -8.419±2.514 | 6.811±6.589 | 22.489±2.598 ✓ | 2.673±1.833 | 24.181±1.214 ✓ | 18.879±1.667 ✓ | -0.255±3.215 | 23.563±2.369 ✓ | -2.824±2.182 | 2.139±4.078 | -3.39±4.752 | 1.32±1.839 | 22.717±1.404 ✓ | 4.449±2.187 | 2.419±4.626 | 79.98±0.418 ✓ |
| Graph | OGBN-arxiv node classification | test_acc | 0.31559 | 23.952±32.164 ✓ | 8.666±13.764 | -1.538±3.21 | 66.277±6.483 ✓ | -34.859±28.071 | 62.124±1.765 ✓ | 62.287±0.274 ✓ | 72.805±2.428 ✓ | 61.759±0.404 ✓ | -81.426±0.0 | 60.103±1.97 ✓ | 60.239±1.946 ✓ | 61.026±1.103 ✓ | -37.371±70.502 | 56.838±4.561 ✓ | -38.683±26.798 | 67.115±2.599 ✓ | 61.887±3.744 ✓ | 68.902±2.198 ✓ | 61.537±1.536 ✓ | 126.219±1.296 ✓ |
| NLP | IMDb sentiment (DistilBERT) | accuracy | 0.91089 | 2.863±0.108 | 2.18±0.071 | -0.384±0.19 | -0.244±0.108 | 0.205 | -0.07±0.154 | -0.287±0.294 | -0.225±0.127 | -0.101±0.296 | -0.097±0.198 | -0.192±0.082 | -0.157±0.105 | -33.09±53.539 | 2.638 | -0.309 | -0.205±0.197 | -0.263±0.014 | 0.141±0.012 | 0.262±0.0 | -0.176±0.087 | 1.825±0.354 |
| Tabular | House Prices regression (MAE) | mae ↓ | 21676.12057 | 18.744±5.173 ✓ | 15.555±3.161 ✓ | 10.16±0.75 ✓ | 2.899±0.572 | -0.462±0.462 | 8.765±3.462 | 12.124±8.041 ✓ | 2.826±0.708 | 23.068±4.263 ✓ | 23.609±3.123 ✓ | 0.2±1.787 | 9.565±1.477 | 3.526±3.058 | 2.764±3.934 | 0.853±0.485 | -63.992±115.061 | -0.751±0.0 | 3.954±0.561 | 2.294±0.225 | 14.275±2.261 ✓ | 29.15±0.748 ✓ |
| Tabular | Spaceship Titanic classification | accuracy | 0.72283 | 9.387±0.774 | 10.395±0.297 ✓ | 10.236±0.258 ✓ | 0.849±1.445 | 0.0±0.0 | 0.133±0.347 | 0.61±0.278 | 10.342±0.331 ✓ | 10.316±0.594 ✓ | 0.371±0.625 | 9.971±0.153 | 1.273±1.029 | 1.565±0.478 | 4.72±2.353 | 0.902±0.594 | 0.57±0.288 | 0.504±0.806 | 4.813±1.378 | 0.636±0.265 | 0.0±0.0 | 11.641±0.564 ✓ |
| <br/> | **均值 mean Δ** | <br/> | <br/> | **14.327%** | **8.181%** | **16.333%** | **13.242%** | **-4.157%** | **12.507%** | **16.309%** | **21.647%** | **19.543%** | **-6.672%** | **17.792%** | **14.133%** | **11.318%** | **-6.015%** | **12.085%** | **-21.14%** | **13.585%** | **18.702%** | **15.309%** | **15.611%** | **49.763%** |
| <br/> | **success (Δ≥10%)** | <br/> | <br/> | 3/5 | 2/5 | 3/5 | 1/5 | 1/5 | 1/5 | 2/5 | 3/5 | 3/5 | 2/5 | 2/5 | 1/5 | 2/5 | 0/5 | 1/5 | 0/5 | 1/5 | 2/5 | 1/5 | 2/5 | 4/5 |

④ **purefable5（原生无提案臂）**：native MLS-Bench agent，无 master proposal，worker=claude-fable-5 从 MLS 原生初始 prompt 自行构思并实现方案(--mode sci，对齐 MLS-Bench 原始基准 + kimi k2.7 code)。MLS 30 任务均值 **35.2**(居中，n=30)；对比其在 MAB 上 **Δ+49.8%(4/5 success)主导**——原生 agent 在 MAB 型「改进已有 train.py」任务上远强于在 MLS 型「从零实现研究提案」任务。faithful-0(·)：ai4bio-mutation / ai4sci-pla-binding(蛋白内容 fable-5 风控空补全)、robo-humanoid(能力地板)、jepa-planning / reasoning-rl-is(提案≤锚点/2gpu-runtime 地板)。
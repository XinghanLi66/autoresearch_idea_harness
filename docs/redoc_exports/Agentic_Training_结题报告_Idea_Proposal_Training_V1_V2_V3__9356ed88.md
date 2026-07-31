<redoc-highlight emoji="tuding" fillColor="yellow">
**结题一句话**：Agentic Training 项目历时三期（2026-04 → 07），回答了一个问题——**后训练能否教会模型提出值得实现的研究 idea？答案是：能，但有条件。** V1 建成"related-work 条件化 → SFT+GRPO → 未来对齐评测"全栈并把评测方法论做对；V2 交付一个可信的 32B CoT-SFT 负结果；V3 用 researcher-CoT distillation + 两阶段训练（SFT → DPO）在 21 臂 × 两个 benchmark 的完整矩阵上给出正结果：**SFT 后的 8B reasoning 模型是 MLS 全场第一（40.4），在固定 worker 下击败 GPT-5.5 与 Claude fable-5**；MAB 上训练把有害的 base（−21%）拉升为最强臂之一（+19%）。全部代码 / 数据 / 模型已开源并在独立集群复现验证。
</redoc-highlight>

本文为 Agentic Training 的**结题文档**：合并 V1+V2 总结报告（RedDoc bea2efdab361c667d496a3fe819b9280）与 V3 报告（现存于 arXiv 论文仓库 tex 版），自成一体，附完整产出物索引。

## 1. 项目定位与三期结构

出发点是用户提出的核心问题：**给定一篇（晚于模型 knowledge cutoff 的）论文的完整 related work，能否通过后训练让模型"提出"这篇论文的 idea？** 动机综述确认的空白：把"完整 related-work 条件化 + 严格时间切分防泄漏 + 参数级后训练 + 未来对齐评测"整合进一个系统——最接近的公开工作只覆盖部分维度。

| 期 | 时间(2026) | 内容 | 一句话结论 |
| --- | --- | --- | --- |
| **V1** | 04–06 | proposal_rl：7B SFT+GRPO 全栈，PRS/FAS/PPL 三类 reward，17 组消融 | 微调 7B 的 FAS 超过 Opus 4.6；评测方法论是最有生命力的产出 |
| **V2** | 06 | strict929 高质量数据 + 32B CoT-SFT | 可信负结果（SFT 0/10 vs base 1/10）→ 触发转向 |
| **V3** | 06–07 | researcher-CoT distillation + SFT→DPO 两阶段，21 臂 × (MLS-Bench-Lite 30 任务 + MLAgentBench 5 任务) 固定 worker 评测 | **idea 质量可训练**：d1sft 全场第一并击败 frontier proposer；训练可把有害 base 拉回正收益 |

## 2. 全程时间线

| 时间(2026) | 里程碑 |
| --- | --- |
| 04 下旬 | 动机综述定位；Ver 1.1 首个 pipeline（LoRA SFT+GRPO），发现 GRPO 使 FAS 回退（0.6232→0.6062） |
| 04-27 | Ver 1.2：exp01–08（RL-only）；exp02 full_refs FAS 0.6767，全部微调 7B 超过 Opus 4.6 API |
| 05-10/11 | Ver 1.3：TRL+DeepSpeed → veRL；exp09–17（SFT+RL）上线 4 机；记录 24 个 bug |
| 05-18 | 端到端 benchmark v2：proposal → loyal worker → MLS-Bench，HMAC 签名 |
| 05-20 | benchmark 缺陷分析：greedy → 20 条相同 proposal；pass@k 实际测 worker 能力 |
| 05-22~29 | 第一版正式评测（约 900 次 worker 评测）；dl_lr_schedule 退役；Jigsaw 定性分析 |
| 06-02~05 | Jigsaw MLE n=20（8% 百分位壁垒 + worker 自降级）；worker-only 阈值校准 |
| 06-09~17 | V2：strict929 cache、928 条 CoT 目标、32B LoRA @16k；**负结果** base 1/10 vs SFT 0/10 |
| 06-29 | V3 转向：researcher-CoT distillation 设计定稿（125 位研究者） |
| 07 上旬 | V3 数据合成（两阶段 synth + fact-check + judge，1,626 train / 85 val）；7B/14B/32B/235B 系列 SFT+DPO |
| 07-14 | V1+V2 开源 wave：GitHub 两仓 public、HF 4 工件 public（coder66）、QS 独立集群复现验证完成 |
| 07 中下旬 | V3 评测收口：21 臂 × 30 MLS-Bench-Lite（原生 harness）+ 21 × 5 MLAgentBench，0 缺格 |
| 07-30 | arXiv 论文核心章节（Method / Experiments / Takeaways）完稿并经数值复核 |

## 3. V1：proposal_rl（SFT + GRPO 全栈）

### 3.1 任务与数据

任务：仅以目标论文的 reading list（参考文献，含摘要）为条件，生成结构化 research proposal（&lt;thinking&gt; + &lt;proposal&gt; XML：problem / gap / key_insight / approach / expected_contributions），与论文真实摘要对齐。

| 项 | 值 |
| --- | --- |
| 语料 | arXiv cs.LG/AI/CL/CV/IR/NE/stat.ML，每篇 ≥8 条可解析引用，每例 ≤40 refs |
| Train | 2025-04→10，**7,357** 例（全部晚于 base 模型发布日，防泄漏） |
| Val | 2025-11→12，**190** 例（= GRPO FAS reward 的检索索引） |
| Test | 2026-01→03，**170** 例（仅终评） |

### 3.2 Reward 设计与训练配方

| Reward | 定义 | 使用 |
| --- | --- | --- |
| **PRS** | proposal 与源论文摘要的 embedding 余弦相似度；reward = 0.8·PRS + 0.2·format | exp01–06/08/09–14/16 |
| **FAS** | 对 held-out 未来语料索引的检索相似度；reward = 0.6·FAS + 0.2·format + 0.2·anti-leak | exp07/15 |
| **PPL** | 以 proposal 为条件时真实摘要的困惑度奖励 exp(−mean_CE/3) | exp17 |

配方演进：Ver 1.1 全程 LoRA（GRPO 对 val 索引过优化 → FAS 净回退）→ Ver 1.2 共享 SFT 起点 + RL-only 消融（收敛到 lr=5e-6 / kl=0.02）→ Ver 1.3 换 veRL（FSDP+Ray+vLLM），SFT 3×3 + RL 3×3 超参搜索；期间记录 24 个 bug，其中 CRITICAL 的 8 行 parquet 残留导致 exp12/13/14 首轮 RL 作废重跑——这是后来"复现验证先于发布"原则的直接教训。

### 3.3 核心结果

**离线指标（test split，节选）**：exp02（full_refs / PRS）FAS **0.6767** / recall@50 0.7529，**所有微调 7B 的 FAS 超过 Opus 4.6 zero-shot（0.6400）**；直接以 FAS 为 reward 的 exp07 在 test 上灾难性崩溃（0.3648）——优化检索指标本身会过拟合索引。

**Worker benchmark（exp09–17，节选）**：exp09（top_k / PRS）max p@1 0.30；exp15（FAS 辅助）分布质量最佳（mean Δ +0.082）；exp14（LoRA）失败于模板泄漏（−0.178）→ **LoRA 不足以承载该任务**。推理策略均值 related_work 最佳、裸 top_k 最差——叙事化上下文优于裸引用列表。

### 3.4 评测方法论（V1 最有生命力的产出）

1. **pass@k 失效**：greedy decoding + 共享 prompt cache → 20 条 proposal 逐字节相同；pass@k 实际测"worker 能力 × proposal 可实现性"。修复：采样 T=0.8，旧指标改名 worker_pass_rate。

2. **Worker 忠诚度失败**：约 85% 的情况 worker 无视 proposal、用自己的先验实现；base 模型 38.9% 通过率 ≈ worker 先验成功率 → dl_lr_schedule 结构性缺陷、退役。pass/fail reward 无法区分"有意义 proposal + 好实现"与"空 proposal + worker 先验"。

3. **Worker-only 阈值校准**：以无 proposal 的 worker 上限重设阈值后，proposal 增益依然存在但稀疏（5–12%）。

4. **微调真正买到了什么**：base 在无显式任务说明的策略下 p@1 = 0%，微调后稳定通过——SFT+RL 教会的是"从参考文献生成 proposal"这一任务本身。

5. **Jigsaw 双报告**：8% 百分位壁垒（数据/标签/模型量级的结构性差距）；**worker 系统性自降级**（20/20 无视 proposal 指定的 RoBERTa-large）——评测测到的是"worker 愿意实现什么"。

<redoc-highlight emoji="dengpao" fillColor="blue">
这五条方法论直接催生了 V3 的评测设计：**固定 worker、跨 benchmark、强弱双 anchor**——V3 的结论之所以可信，根子在 V1 交的学费。
</redoc-highlight>

## 4. V2：32B CoT-SFT（strict929）——一个可信的负结果

### 4.1 数据底座

- **strict929 quality cache**：929 篇 / 25,383 refs / 审计 0 问题；compact abstracts、长 research question（均值 171 词）、TeX 细节片段（method / implementation / training-recipe / evaluation / results）。

- **V1-semantics prompt 重建**：在新 cache 底座上复刻 V1 条件化语义，929/929 硬校验通过，保证与 V1 可比。

- **CoT 目标**：928/929 条 schema 完整（1 篇 safety refusal 排除）；发布数据集 **928 train / 93 val**。

### 4.2 结果

32B LoRA @ 16k context，约 49 min 训完。10 个 MLS 任务 worker 评测（两臂均 10/10 完成、0 error）：

| 臂 | Pass | 备注 |
| --- | --- | --- |
| Qwen2.5-32B base | **1/10** | cv_pooling_aggregation 72.63 vs 线 72.19 |
| CoT-SFT | **0/10** | 5/10 任务 raw metric 提升但均未过线 |

负结果的可信度来自同期 harness 修复（双执行 guard、worker 过早终止修复、报告合并优先取带指标重试）。案例研究显示 worker 忠诚度已高（近逐字实现）→ 瓶颈转移到 proposal 本身"概念正确但机制浅"。

<redoc-highlight emoji="cuo" fillColor="red">
**V2 结论**：轻量 V1-like CoT SFT 不 promising（SFT 0/10 vs base 1/10），不建议继续同类 SFT。**格式与合理性层面的能力已具备，缺的是机制深度与创造力**——这直接触发 V3 的 researcher-CoT distillation 转向。
</redoc-highlight>

## 5. V3：Researcher-CoT Distillation + 两阶段训练（正结果）

### 5.1 方法：proposer 与 worker 解耦

系统把 **idea 生成（可训练的 proposer）** 与 **idea 执行（固定的 worker agent）** 分离：proposer 读取任务包生成结构化 research proposal，worker（前沿 coding agent，**所有条件下完全一致**）负责实现。固定 worker 把 benchmark 结果变成对 proposal 质量的受控读数——这是对 V1 方法论教训的直接回应。

**Stage 1 — Researcher-CoT SFT**：把 **125 位顶尖 ML 研究者**的推理风格蒸馏为监督目标。对每位研究者收集其关键创新的文档化案例，合成第一人称 chain-of-thought（入口观察 → 显然做法为何失败 → 跃迁 → 为何非显然），固定收尾两个锚点：**Core idea** 与 **Non-trivial crux**。合成为两阶段（生成 + 独立 fact-check 去除虚构结果 / 时代错置 / 后见之明），再过 rubric judge；发布训练集 **1,626 条** judge-accepted traces（+85 val）。每条样本以 researcher 命名的 system prompt 条件化，推理时可 steer。

**Stage 2 — Preference Optimization（DPO/RL）**：在 SFT checkpoint 上做 proposal 偏好对的 DPO。实验显示它是**稳定器与救援器**：高 headroom 任务上单调增益，SFT 过冲时可回收。

### 5.2 评测：21 臂 × 双 benchmark，0 缺格

21 个 proposer 臂：DeepSeek-R1-0528-Qwen3-8B（d1base/d1sft/d1rl）、instruct 稠密 base（Qwen3-8B/14B/32B、Qwen2.5-32B）及其训练变体、235B MoE（Qwen3-235B-A22B，m2 系列）、两个 frontier API proposer（GPT-5.5、Claude fable-5）、以及无 proposal 对照（purefable5 = worker 单干）。

- **MLS-Bench-Lite**（30 任务、12 个 ML 领域）：from-scratch 方法设计，对强人工 baseline；分数 rescale 使 **50 = 强 baseline anchor**。保守 benchmark。

- **MLAgentBench**（5 任务）：对已有代码做增量改进，报告相对 baseline 的平均 Δ%。高 headroom benchmark。

<redoc-highlight emoji="gantanhao" fillColor="orange">
**强弱双 anchor 是有意设计**：只用强 anchor 会低估方法（几乎没人能赢 tuned baseline），只用弱 baseline 会高估方法。两者并用才能同时看到"离人类水平还有多远"和"训练真实带来多大增益"。
</redoc-highlight>

### 5.3 旗舰正结果：idea 质量可训练

**DeepSeek-R1-8B 训练阶梯**（同一固定 worker）：

| <br/> | d1base | d1sft | d1rl |
| --- | --- | --- | --- |
| MLS-Bench-Lite（30 任务均值） | 29.7 | **40.4** | 36.2 |
| MLAgentBench（5 任务均值 Δ%） | −21.1 | +13.6 | **+18.9** |

- MLS 上 SFT 带来 **+10.7**（全部模型家族中最大的训练增益），d1sft 成为**全场第一臂**——超过 30 倍大的 235B MoE（38.6）和所有 Qwen3 base。

- **正面击败 frontier proposer**：同一固定 worker 下，d1sft 对 fable-5 / GPT-5.5 的逐任务最优 **14 胜 10 负 6 平**，均值 **40.4 vs 32.0 / 31.4**。大幅胜局包括 Unconditional Graph Generator **+75**、Diffusion Policy **+52**、Convolutional Activation Nonlinearity **+50**、Discrete Causal Graph Discovery **+41**。

- **MAB 上训练完成"救援"**：d1base 的 proposal 让 worker 净受损（−21.1%），SFT → DPO 单调拉升到 **+18.9%**（40 分摆幅）；ogbn-arxiv 单任务从 −38.7% 到 **+61.9%**（约 100 分摆幅）。

### 5.4 训练后的 proposer ≥ frontier proposer

MLS 排行榜 **top-8 全部是训练/开源 checkpoint**，每一个都高于两个 frontier API proposer（fable-5 32.0 / GPT-5.5 31.4）：d1sft 40.4、Qwen3-32B base 39.6、235B MoE base 38.6、Qwen3-14B base 38.4、Qwen2.5-32B base 37.8、m2sft 36.4、Qwen3-32B SFT 36.2、d1rl 36.2。MAB 上跨 benchmark 一致：**17/21 臂为正**，领先者是训练 checkpoint（sft14b +21.6%、qwen3-14b-rl +19.5%、d1rl +18.9%、sft32b +17.8%、m2rl +16.3%），frontier 居中游（+14.3% / +8.2%）。

<redoc-highlight emoji="dui" fillColor="green">
**固定 worker 时，瓶颈是"贴合任务分布的 idea 生成"而非 proposer 的规模**——frontier 规模买不到更好的研究 idea。
</redoc-highlight>

无 proposal 对照 purefable5：MLS 仅中游（35.2），MAB 却统治级（**+49.8%**，4/5 任务最佳）→ master proposal 的收益依赖任务类型：from-scratch 新设计任务收益最大，增量改代码任务上强 native agent 本已足够。

### 5.5 诚实的负结果

<redoc-highlight emoji="cuo" fillColor="red">
1. **Proposal-SFT 伤害 instruct 稠密 base（MLS）**：Qwen3-8B 35.0→29.0（−6.0）、Qwen3-14B 38.4→32.5（−5.9）、Qwen3-32B 39.6→36.2（−3.4）、Qwen2.5-32B 37.8→33.5（−4.3）。同一配方给 reasoning-native base +10.7、给 MoE 约中性、给所有 instruct 稠密 base 减分。

2. **多数 MLS 臂低于 50**：from-scratch 实现研究 idea 去赢一个 tuned 人工 baseline 真的难。

3. **训练抬升的是均值而非每一格**：d1sft 在 idea 失手的任务上仍输给 frontier（两个任务 faithful-0）。
</redoc-highlight>

## 6. 五条 Practitioner Takeaways

1. **Idea 质量可训练，不是只能靠规模买。** 对 reasoning-native 8B 做 researcher-CoT SFT，其 proposal 在完全相同的实现条件下击败 GPT-5.5 与 Claude fable-5。生成强 ML 研究 idea 不需要 frontier 模型。

2. **训练配方要匹配 base 模型。** Proposal-SFT 帮助 reasoning-native（DeepSeek-R1）与 MoE base，伤害 instruct 稠密模型。选错 base，同一配方就反噬。

3. **把 RL（DPO）当稳定器与救援器用。** 高 headroom 任务上单调增益（MAB d1 阶梯 base → SFT → RL），SFT 过冲时可回收（Qwen3-8B MLS 29.0→33.9）。

4. **Master proposal 的收益取决于任务类型。** from-scratch 新设计任务（MLS）收益最大；增量改进任务（MAB）上强 native agent 本已可统治。把 proposal scaffolding 用在设计新颖性重要的地方。

5. **同时对强 anchor 与弱 baseline 评测。** 强 anchor（MLS，50 = tuned baseline）暴露"多数方法赢不了 tuned 参照"；弱 baseline（MAB）揭示真实 headroom 与训练增益。只用其一必然高估或低估。

## 7. 开源与复现（最终状态）

**GitHub（均 public）**：XinghanLi66/proposal_rl（V1，tag v1-oss-rc1，Apache-2.0）；XinghanLi66/autoresearch_idea_harness（V2+V3 harness，tag v2-oss-rc1）。内部基础设施标记 internal-only，配置全部环境变量化；双仓 git grep /newcpfs = 0。

**HuggingFace（namespace coder66，均 public）**

| 工件 | 内容 | 规模 |
| --- | --- | --- |
| arxiv-research-proposals-v1（dataset） | 7,357 / 190 / 170 时间切分 | ≈790 MB |
| proposal-rl-qwen2.5-7b-ppl-grpo（model） | V1 最优 RL 7B | 15 GB |
| arxiv-proposal-cot-sft-32b-v2（dataset） | 928 / 93 CoT-SFT 行 | 211 MB |
| proposal-cot-sft-qwen2.5-32b-lora（model） | 32B LoRA adapter | 1.1 GB |

许可：代码 Apache-2.0，数据 CC-BY-4.0，模型 Apache-2.0。全部工件逐行 JSON 校验 + SHA256SUMS。

**独立集群复现（QS2 GB200，aarch64——与原始运行不同硬件/镜像/文件系统）**：V2 32B LoRA 全量复训 ✅（trial 1699595，loss 2.033→1.837）；V1 7B SFT ✅（trial 1699771）；V1 PPL-RL 复现至 rollout 启动、被镜像级 Ray/CUDA 库问题阻断（documented-blocked，见 REPRODUCE.md）。复现过程发现并上游修复 2 个真实缺陷（过期依赖 pin、多 rank parquet 竞态）——**先复现、后发布**得到验证。

## 8. 产出物总索引

| 类别 | 产出 | 位置 |
| --- | --- | --- |
| 代码仓 | proposal_rl（V1）、autoresearch_idea_harness（V2/V3） | github.com/XinghanLi66/ |
| 数据与模型 | 4 个 HF 工件 | huggingface.co/coder66 |
| arXiv 论文 | ICLR 2027 格式，Method/Experiments/Takeaways 已完稿并经数值复核（8 页） | github.com/XinghanLi66/idea-proposal-training-paper（本地 idea-proposal-training-paper/） |
| Notion 博客 | Idea Proposal Training（§1–9，§8 V3 待补） | Notion 工作区根页面 |
| V1+V2 总结报告 | 方法、结果、开源、贡献者说明 | RedDoc bea2efdab361c667d496a3fe819b9280 |
| V3 结果矩阵 | 21 臂 × 30 MLS 任务（LIVING） | RedDoc 33de9d2d061b638554b21775b8c51b55 |
| V3 数据集与创造力评测 | B3 dataset funnel + Part C | RedDoc 793237f34e6c34d138de444b831735c9 |
| V3 SFT+RL 配方与 failure modes | 训练配方、失败模式调查 | RedDoc b6b0310b597cc5c26a49a5f1a6fb0b60 |
| V3 program tracker | 任务/计划/agenda（LIVING） | RedDoc 6653ed0a74f069511a75ff1b88d6b9bf |
| 结果 dashboard | 30 任务 × 21 臂可点击矩阵 + MAB | autoresearch_idea_harness/runs/researcher_cot/mls_lite_dashboard/ 与 …/mab/ |
| Findings 底稿 | 引用级 findings + takeaways 包 | autoresearch_idea_harness/docs/eval/proposal_training_findings.md |

## 9. 结语

三期走完，这个项目最终交付了两样东西：一个**方法学答案**（idea 质量可以后训练获得，条件是 reasoning-native base + researcher-CoT SFT + DPO，并用固定 worker、双 anchor 的方式诚实测量），和一套**被负结果与复现打磨过的流程**（V1 的评测方法论、V2 的可信负结果、发布前独立复现）。正结果告诉别人这条路能走；负结果与方法论告诉别人这条路上的坑在哪。两者合在一起，才是这次 Agentic Training 的完整答案。

> 本文为结题文档。后续若有更新（arXiv 投稿、V3 追加实验），以 program tracker 与论文仓库为准。
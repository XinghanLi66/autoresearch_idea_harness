## 0. TL;DR

<redoc-highlight emoji="tuding" fillColor="yellow">
**一句话**：V1 建立了"related-work 条件化 → 后训练 → 未来对齐评测"的完整栈，微调后的 7B 在 FAS 上超过 Claude Opus 4.6 zero-shot（0.6767 vs 0.6400），但更大的收获是把评测本身做对（pass@k 失效、worker 忠诚度、worker-only 阈值校准）；V2 用 strict929 高质量数据对 32B 做 CoT-SFT，得到一个**可信的负结果**（base 1/10 vs SFT 0/10 pass），并据此转向 V3（researcher distillation，进行中、另行报告）。本轮已完成 V1+V2 开源准备：GitHub 分支+tag 已推送、HuggingFace staging 就绪、QS 独立集群复现验证进行中。
</redoc-highlight>

- **V1（proposal_rl，2026-04~06）**：7,357/190/170 时间切分数据集、PRS/FAS/PPL 三类 reward、17 组消融（SFT+GRPO）、端到端 worker benchmark。结论：full_refs 条件化最强；微调 7B 的 FAS 超过 Opus 4.6；直接优化 FAS 反而崩溃（exp07=0.3648）；LoRA 不足以承载该任务（exp14 模板泄漏）。

- **V1 评测方法论（同等重要的产出）**：发现 pass@k 在 greedy decoding 下退化为"worker 能力测量"；worker 约 85% 概率用先验覆盖 proposal（dl_lr_schedule 因此退役）；以 worker-only 上限重设阈值后，proposal 增益依然存在但稀疏（5–12%）。

- **V2（32B CoT-SFT，2026-06）**：strict929 质量底座（929 篇/25,383 refs/审计 0 问题）+ V1-semantics prompt 重建 + 928 条 CoT 目标 + 32B LoRA @16k（≈49 min）。结果 **SFT 0/10 vs base 1/10**（5/10 任务 raw metric 提升但未过线）；harness 可靠性修复使该负结果可信；明确结论"轻量 CoT SFT 不 promising"→ 转向 V3。

- **开源（2026-07-14）**：两个去内部化分支 + tags 已推送；HF manifest（≈17 GB）+ 6 张卡 + SHA256SUMS 就绪；QS 复现验证已委托执行。

## 1. 项目背景与定位

出发点是用户提出的核心问题：**给定一篇（晚于模型 knowledge cutoff 的）论文的完整 related work，能否通过后训练让模型"提出"这篇论文的 idea？** 动机综述（survey doc）把该设想与四条研究线对齐后确认的空白是：把"完整 related-work 条件化 + 严格时间切分防泄漏 + 参数级后训练 + 未来对齐评测"整合进一个系统——最接近的公开工作只覆盖其中部分维度。项目由此分为三期：V1（7B，SFT+GRPO，全栈与评测方法论）→ V2（32B，CoT-SFT，高质量数据）→ V3（researcher-CoT distillation，进行中，不在本报告范围）。

## 2. 时间线

| 时间(2026) | 里程碑 |
| --- | --- |
| 04 下旬 | 动机综述定位；Ver 1.1 首个 pipeline（LoRA SFT+GRPO），发现 GRPO 使 FAS 回退（0.6232→0.6062） |
| 04-27 | Ver 1.2：exp01–08（RL-only，TRL）；exp02 full_refs FAS 0.6767，全部微调 7B 超过 Opus 4.6 API |
| 05-10/11 | Ver 1.3：TRL+DeepSpeed → veRL（BF16 overflow）；exp09–17（SFT+RL）上线 4 机；记录 24 个 bug（含 CRITICAL 8 行 parquet） |
| 05-18 | 端到端 benchmark v2：proposal → loyal worker（Claude Code）→ MLS-Bench，SlotPool 24 GPU，HMAC 签名 |
| 05-20 | benchmark 缺陷分析：greedy → 20 条相同 proposal；pass@k 实际测 worker 能力 |
| 05-22~26 | 第一版正式评测（≈900 次 worker 评测，200–300 GPU·h）；base 模型 with_rq 38.9% 揭示任务缺陷 |
| 05-27~29 | dl_lr_schedule 退役；Jigsaw 定性分析（with_rq 73% 对齐；exp14 LoRA 模板泄漏排除） |
| 06-02~05 | Jigsaw MLE n=20（"8% 百分位壁垒"+worker 自降级）；worker-only 基线 → 阈值重设；发现旧 Jigsaw grader bug |
| 06-09~15 | V2：strict929 cache、V1-semantics 重建、928 条 CoT 目标、32B LoRA @16k（≈49 min） |
| 06-17 | **负结果**：base 1/10 vs CoT-SFT 0/10（harness 修复后 0 error，可信） |
| 06-29 | held-out 创造力 demo（5/5 schema 完整）；集群缩编 → V3 转向 |
| 07-14 | 开源 wave 1：两分支+tags 推送；HF staging 就绪；QS 复现委托 |

## 3. V1：proposal_rl（SFT + GRPO 全栈）

### 3.1 任务与数据

任务：仅以目标论文的 reading list（参考文献，含摘要）为条件，生成结构化 research proposal（&lt;thinking&gt; + &lt;proposal&gt; XML：problem/gap/key_insight/approach/expected_contributions），与论文真实摘要/未来方向对齐。

| 项 | 值 |
| --- | --- |
| 语料 | arXiv cs.LG/AI/CL/CV/IR/NE/stat.ML，每篇 ≥8 条可解析引用，每例 ≤40 refs |
| Train | 2025-04→10，**7,357** 例（全部晚于 base 模型 2025-03-07 发布，防泄漏） |
| Val | 2025-11→12，**190** 例（= GRPO FAS reward 的检索索引） |
| Test | 2026-01→03，**170** 例（仅终评） |
| SFT 目标 | Claude 合成 CoT proposal（合成 prompt 刻意禁用真实论文的方法名/数字防泄漏——后被确认为"抽象化"根因） |

### 3.2 Reward 设计与训练配方

| Reward | 定义 | 使用 |
| --- | --- | --- |
| **PRS** | proposal 与源论文摘要的 embedding 余弦相似度；RL reward = 0.8·PRS + 0.2·format | exp01–06/08/09–14/16 |
| **FAS** | proposal 对 held-out 未来语料索引的检索相似度（mean_sim + recall@50）；reward = 0.6·FAS + 0.2·format + 0.2·anti-leak | exp07/15 |
| **PPL** | 以 proposal 为条件时真实摘要的困惑度奖励 exp(−mean_CE/3) | exp17 |

配方演进：Ver 1.1 全程 LoRA（GRPO 提升 mean_sim/format 但 recall@50 回退 → FAS 净回退，对 val 索引过优化）→ Ver 1.2 共享 SFT 起点 + RL-only 消融（收敛到 lr=5e-6/kl=0.02）→ Ver 1.3 换 veRL（FSDP+Ray+vLLM），逐策略 SFT 3×3 + RL 3×3 超参搜索，每实验 4×H800；期间记录 24 个 bug，其中 CRITICAL bug 22（train parquet 被 smoke 测试残留成 8 行）导致 exp12/13/14 首轮 RL 作废重跑——这也是后续"复现验证"必要性的直接教训。

### 3.3 结果：离线指标 + worker benchmark

**exp01–08（test split，按 FAS 排序，节选）**

| 模型 | 条件/reward | FAS | recall@50 | PRS |
| --- | --- | --- | --- | --- |
| exp02 | full_refs / PRS | **0.6767** | **0.7529** | **0.6222** |
| exp03 | related_work / PRS | 0.6615 | 0.7353 | 0.6093 |
| Opus 4.6 API | — | 0.6400 | 0.7235 | 0.5703 |
| Sonnet 4.6 API | — | 0.6228 | 0.6882 | 0.5700 |
| exp07 | top_k / FAS reward | **0.3648（崩溃）** | 0.0882 | 0.4140 |

要点：full_refs 优于 top-k 选择；**所有微调 7B 的 FAS 超过 Opus 4.6 zero-shot**；直接以 FAS 为 reward 在 test-time FAS 上灾难性失败（val→test 漂移 + KL 漂移 0.089 vs 0.018）。

**exp09–17（worker benchmark，dl_lr_schedule 旧阈值 93.01，节选）**

| 实验 | 条件/reward | max p@1 | mean p@1 | mean Δ |
| --- | --- | --- | --- | --- |
| exp09 | top_k_refs / PRS | **0.30** | 0.171 | +0.016 |
| exp16 | full_refs 20×800 / PRS | 0.25 | **0.180** | −0.008（reward-hacking 征兆） |
| exp15 | FAS（自 exp09 SFT） | 0.20 | 0.164 | **+0.082**（分布质量最佳） |
| exp17 | top_k / PPL | 0.25 | 0.150 | −0.010（多样性低） |
| exp14 | full_refs / **LoRA** / PRS | 0.15 | 0.040 | −0.178（失败：模板泄漏） |

推理策略均值：related_work 最佳（p@1 0.168, Δ+0.052），top_k_refs 最差（0.096, −0.107）——叙事化上下文优于裸引用列表。Reward 结论：PRS 峰值强但助长抽象化；FAS 适合做辅助项；PPL 方向合理无净胜；**LoRA 不足以承载本任务**。

### 3.4 评测方法论（V1 最有生命力的产出）

1. **pass@k 失效**：greedy decoding + 共享 prompt cache → 20 条 proposal 逐字节相同（diff 验证），pass@k 实际测"worker 能力 × proposal 可实现性"。修复：采样 T=0.8/top_p=0.95，旧指标改名 worker_pass_rate。

2. **Worker 忠诚度失败**：约 85% 的 worker 无视 proposal 用自己的先验（warmup+cosine）实现；base 模型 with_rq 38.9% 通过率 ≈ worker 先验成功率 → **dl_lr_schedule 结构性缺陷，退役**。核心诊断：pass/fail reward 无法区分"有意义 proposal+好实现"与"空 proposal+worker 先验"，RL 会强化后者。

3. **Worker-only 阈值校准**：无 proposal 的 worker 上限 + 0.01pp 重设三任务阈值（93.19/93.35/93.76），重评后 proposal 增益仍在但稀疏（LR：exp13 12%；AUG：exp16 10%，首次非零）。同时发现旧 Jigsaw grader bug（0.82–0.91 全为 norm_score 而非 leaderboard 百分位，作废）。

4. **微调真正买到了什么**：base 在 full_refs/related_work/top_k_refs 三个"无显式任务说明"策略下 p@1=0%，而微调后稳定通过——SFT+RL 教会的是"从参考文献生成 proposal"这一任务本身，而非指令跟随。

5. **Jigsaw 双报告**：with_rq 是唯一可靠对齐策略（11 模型中 73% 对齐 vs full_refs 18%）；exp14 LoRA 模板泄漏（脚手架问题文本进入 XML）被系统定位并排除；MLE n=20 全员卡在 leaderboard 6–9% 百分位（"8% 壁垒"= 数据/标签/模型量级的结构性差距），并发现 **worker 系统性自降级**（20/20 无视 proposal 指定的 RoBERTa-large，以想象中的显存约束改用 base）——评测测到的是"worker 愿意实现什么"。

## 4. V2：32B CoT-SFT（strict929）

### 4.1 数据底座（回应 V1 的"抽象化"根因）

- **strict929 quality cache**：逐篇构建 compact abstracts（target+refs）、长 research question（均值 171 词）、TeX 细节片段（method/implementation/training-recipe/evaluation/results）与 6 种 prompt cache。规模：**929 篇 / 25,383 refs / 审计 problem_count 0**。上游：1,024 条严格合成目标经 LLM 审计接收 929（均分 91.05，主要 hard-failure 为 implementation_recipe）。

- **V1-semantics prompt 重建**：在 V3 cache 底座上复刻 V1 条件化语义（seed-42 shuffle、top-k 索引、related-work 叙事、with_rq），929/929 硬校验通过——保证与 V1 可比。

- **CoT 目标**：928/929 条 schema 完整（1 篇因 Opus 对 malware 内容 safety refusal 而排除）；审计接收 852/928（均分 91.85）。发布数据集 = **928 train / 93 val**。

### 4.2 训练与评测：一个可信的负结果

32B LoRA @ max_seq_length 16,384，全部 928 行，PAI DLC ≈49 min 完成。10 个 MLS 任务 worker 评测（两臂均 10/10 完成、0 error）：

| 臂 | Pass | 备注 |
| --- | --- | --- |
| Qwen2.5-32B base | **1/10** | cv_pooling_aggregation 72.63 vs 线 72.19 |
| CoT-SFT | **0/10** | 5/10 任务 raw metric 提升（最高 +1.33pp）但均未过线；1 例 NaN 崩溃 |

负结果的可信度来自同期 harness 修复：DLC 双执行 guard、worker 过早终止修复（允许自修复重试）、报告合并优先取带指标的重试。案例研究显示 worker 忠诚度已高（activation/weight-init 近逐字实现）→ 瓶颈转移到 proposal 本身"概念正确但机制浅"（无初始化/约束/schedule/消融/失败处理）。held-out demo（5 篇 2505 论文）5/5 schema 完整、可跨域桥接——**格式与合理性层面的能力存在，缺的是机制深度与创造力**。

### 4.3 结论与转向

V2.5 报告原文结论：轻量 V1-like CoT SFT 不 promising（SFT 0/10 vs base 1/10），不建议继续同类 SFT；下一步应结合 reward 与 outcome（PRS/RL、质量过滤、更强 master、master-worker 对话进入训练目标）；strict929/V1-semantics cache 保留为数据底座，本次 SFT 作为可靠 baseline。这一结论直接触发 V3（researcher-CoT distillation）——V3 进行中，另行报告。

## 5. 开源发布（2026-07-14 状态）

本轮开源 V1+V2（V3 仍在进行、暂不发布）。原则：**外部可复现 = 从发布数据集训练 + 标准 Anthropic API 评测**；内部基础设施（Runway proxy、PAI-DLC、QS 提交、hi CLI）标记 internal-only，以发布数据集替代其产物。

**GitHub（已推送）**

- proposal_rl@opensource-prep（tag v1-oss-rc1，7 commits / 49 文件）：Apache-2.0 LICENSE、pinned requirements（按真实运行环境；并纠正 README 错误的 trl 依赖——实际 trainer 是 veRL）、36 个文件的 /newcpfs 硬编码与 13 处内部 proxy 引用全部移除（超出前期 audit 估计，已补齐）、ANTHROPIC_API_KEY/.env 标准化、README Setup + REPRODUCE.md（expected results 取自 exp01–07 真实 eval summary，如 exp02 FAS 0.677/PRS 0.622、exp07 崩溃 0.365 亦如实写入）。

- autoresearch_idea_harness@opensource-v2（tag v2-oss-rc1，7 commits / 29 文件）：同等处理；REPRODUCE.md 开篇即声明诚实预期（指标可复现、pass-rate 无增益是已记录结果）；Runway 合成与 PAI/QS 提交标记 internal-only。

- 验证门（两分支均零命中）：git grep /newcpfs = 0、内部 proxy IP = 0、他人目录 = 0；全量 py_compile / bash -n 通过；配置改为环境变量/HF hub id 解析并实测。

**HuggingFace（staging 就绪，等 owner 上传后填入 namespace）**

| 工件 | 内容 | 规模 |
| --- | --- | --- |
| arxiv-research-proposals-v1（dataset） | train/train_cot/val/test + refs_val/test，**7,357/190/170** | ≈790 MB |
| proposal-rl-qwen2.5-7b-ppl-grpo（model） | V1 最优 RL 7B（exp17 lr5e-6/kl0.01，mean_reward 0.3287） | 15 GB |
| arxiv-proposal-cot-sft-32b-v2（dataset） | **928/93** CoT-SFT 行 | 211 MB |
| proposal-cot-sft-qwen2.5-32b-lora（model） | 32B LoRA adapter（16k checkpoint） | 1.1 GB |

可选关闭项：refs_train 1.5 GB、RL full-run 15 GB、strict929 非 CoT 变体、63 GB merged 32B。许可：代码 Apache-2.0；数据 CC-BY-4.0；模型 Apache-2.0（Qwen2.5-7B/32B-Instruct 衍生，两者均 Apache-2.0）。质检：全部数据集逐行 JSON 校验、行数与卡片一致；模型 config/safetensors header 校验、无内部路径泄漏；SHA256SUMS 已生成。

**复现验证（进行中）**：已按用户指示委托 cc000/exp-manager 在 QS2 独立集群执行——V2 = pod 内 clone opensource-v2 分支 + 发布数据集训 32B LoRA 对齐已记录 run；V1 = opensource-prep 小样本 SFT+PPL-RL（纯环境变量配置）。receipts 落地后回填本节与两个 REPRODUCE.md。

**待 owner 操作（已按指示延后）**：huggingface-cli login + 上传（hf_release/README.md 步骤）；GitHub 仓库设 public；上传后替换 REPRODUCE 中的 namespace 占位符。

## 6. 贡献者说明

本项目由用户主导研究方向与设计决策，经 Claude/Codex agent fleet 执行；以下贡献在文档/检查点中可直接溯源：

- **问题提出**：核心设想（"以晚于 cutoff 论文的完整 related work 为条件，后训练模型提出该论文的 idea"）为用户原始框架，动机综述围绕其构建。

- **实验矩阵设计**：exp01–17 的消融纲要（top-k→full refs→related-work 叙事→RQ；full-FT vs LoRA；PRS→FAS→PPL）逐字出现在用户撰写的 Ver 1.1 "Next Steps" 中。

- **评测哲学**："要超过能拿到的最强 API"（Ver 1.1）；从代理指标推进到端到端 worker benchmark；V2.5 目标与约束（时间序训练、无目标泄漏、"pass@1 长期接近 80%"、"提出明显 nontrivial 的改进"）均为用户设定。

- **规模押注**："7B 已超过 Claude，那训大模型呢？"（Ver 1.2）→ V2 的 32B 决策。

- **过程纠偏**：DLC 排队作业的配额政策修正（06-10）、agent 自治规则（06-11）、复现任务改道 cc000（07-14）、发布顺序"RedDoc 先行、arXiv 延后"（07-14）等，见 checkins。

## 7. 关键证据文档索引

| 主题 | 文档 |
| --- | --- |
| 动机综述 | a1f81937d42ea5dae85d233042359403 |
| Ver 1.1/1.2/1.3 配方与消融 | f15c2ac8…, cb9de9af…, ba3f780c… |
| benchmark 设计 / 缺陷分析 | 3c29af5e…, 8e2ced71… |
| 第一版评测 + 追加 baseline | 42c91b51…, a8e799d8…（子文档 9cce7c29…） |
| worker-only 阈值校准 | 6be068ee… |
| Jigsaw 定性 / MLE n=20 | d9e61e9c…, 42b50ab0… |
| V2.5 主报告 / tracker | cf1d2385…, d50f68bd… |
| V2 demo（paper / MLS） | 9658a39b…, 8f8624bc… |
| 负结果数据 | runs/reports/v3_base_vs_v1sem_cot_sft_eval_partial.md |
| 开源 staging | /newcpfs/lxh/agentic-training/hf_release/（manifest+cards+SHA256SUMS） |

> 本报告作为 V1+V2 的收口文档；V3（researcher-CoT distillation）进行中，将另行报告。arXiv preprint 按用户指示延后，届时以本报告与上表文档为底稿。

## 8. 复现验证结果（2026-07-15 追记）

<redoc-highlight emoji="dui" fillColor="green">
**V1+V2 的训练路径已在独立集群（QS2，4×GB200，aarch64——与原训练完全不同的硬件/镜像/文件系统）上复现成功**：pod 内直接 HTTPS clone 已推送的 opensource 分支执行。
</redoc-highlight>

| 项 | 结果 | 证据 (trial) |
| --- | --- | --- |
| 分支 clone + 全量 py_compile（新镜像） | ✅ PASS | 1697055 |
| V2 smoke（256 行，32B LoRA，4 卡） | ✅ PASS（loss 1.87–2.00，checkpoint 保存） | 1699582 |
| **V2 完整 run**（928 行 @16k，LoRA r64/α128，1 epoch） | ✅ **PASS**（loss 2.033→1.837(min)→1.880，checkpoint global_step_14） | 1699595 |
| **V1 SFT**（7B LoRA，1,024 行切片，1 epoch） | ✅ **PASS**（loss 1.701→1.25 / 16 步，checkpoint 保存） | 1699771 |
| V1 PPL-RL（GRPO + vLLM rollout） | ⏸ 部分验证：vLLM 0.18+Ray 2.55 在 aarch64 可装可导入、RL parquet+PPL reward 就绪、rollout 启动；阻塞于镜像级问题（Ray worker 无法解析 libcudart.so.12） | 1699836→1700268 |

**复现过程发现并已修复的真实 reproducibility 缺陷**（这正是做独立复现的价值）：

1. 发布 pins 过期：peft==0.13.2 / accelerate==1.7.0 无法运行 transformers-5 + verl-0.7 训练路径（本地环境后期升级导致的 drift）→ 两分支已改为 peft&gt;=0.15 / accelerate&gt;=1.10（1.14.0 verified）+ torchao 冲突说明（commits dea30b6 / da9fd43）。

2. train/sft.py 多 rank parquet 竞态：dist barrier 在初始化前静默失效 → rank0 原子写 + 其余 rank 轮询（commit 7bfc31e）。

3. 运行时补充说明（已写入 REPRODUCE）：verl FSDP→HF merge 的枚举小 bug（sharded ckpt 完好、可离线 merge）；RL 需 Ray worker 可见 CUDA runtime libs。

**发布状态更新**：两个 GitHub 仓库已由 owner 设为 **public**（proposal_rl / autoresearch_idea_harness，分支 opensource-prep / opensource-v2 + tags v1-oss-rc1 / v2-oss-rc1）。剩余：HF 上传（owner，staging 就绪）→ 填 namespace 占位 → （建议）将 opensource 分支 merge 回 main 作为公开门面。

### §8 追记（2026-07-15 下午）— 发布收口

- **HuggingFace 上传完成**（namespace coder66，当前为 private，owner 翻 public 即对外可见）：
coder66/arxiv-research-proposals-v1（7,357/190/170 + CoT + refs_val/test）、
coder66/proposal-rl-qwen2.5-7b-ppl-grpo（V1 最优 RL 7B）、
coder66/arxiv-proposal-cot-sft-32b-v2（928/93）、
coder66/proposal-cot-sft-qwen2.5-32b-lora（32B LoRA）。上传文件集与卡片核验一致。

- **GitHub main 已合并**：两仓库 public 且 main 已 fast-forward 到清理后的 opensource 分支
（proposal_rl main=cc5dc27，autoresearch_idea_harness main=b01b5a1）；REPRODUCE 中的
namespace 占位已替换为 coder66。

- 至此 V1+V2 开源、独立集群复现、RedDoc 报告三项全部完成；arXiv 版本按计划延后。
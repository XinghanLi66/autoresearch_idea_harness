## 概述

本报告对 **Jigsaw Unintended Bias in Toxicity Classification**（Civil Comments 数据集）任务下，11 个模型 × 5 个策略共 55 条提案（Proposal）进行深度质量分析。所有提案均为 generate-only 模式（n=1），目的是考察各模型/策略组合能否理解并针对毒性分类任务产出有意义的研究提案。

<redoc-highlight emoji="tuding" fillColor="yellow">
**任务背景**：Jigsaw Civil Comments 要求分类器在保证整体 AUC 的同时，避免对包含身份群体词汇（如「gay」「Muslim」「Black」）的非毒性评论打出虚高的毒性分数。官方指标为加权偏差感知 AUC（含 subgroup AUC、BPSN AUC、BNSP AUC 四项的幂均值）。
</redoc-highlight>

---

## 核心结论

<redoc-highlight emoji="dengpao" fillColor="blue">
1. **研究问题提示（with_research_question）效果显著**：几乎所有模型在该策略下都能产出与 Jigsaw 毒性任务直接相关的提案，而在其他策略下普遍退化为通用 NLP 公平性研究。

2. **两个模型系统性跑偏至视觉/多模态**：exp11（topk_rw SFT+RL）和 exp15（FAS from exp09）在多个策略下产出的是视觉语言模型（VLM）或视频理解相关提案，与 NLP 毒性任务完全无关。

3. **exp14（LoRA）存在模板泄漏**：所有 5 条提案的 XML 标签内均残留问题模板文本（如「What core research problem should be addressed?」），说明该模型未能正确替换 scaffold。

4. **claude-opus-4-6 质量最高**：提案结构完整、领域针对性强，with_research_question 策略下提案精准覆盖 BPSN/BNSP AUC、身份词偏差解耦等核心问题；另有 1 条截断失败（fc0feb72，仅有 thinking 块，无 proposal 输出）。

5. **qwen25_7b_instruct_base 仅在 with_research_question 下对齐**：其余 4 条提案为通用 fairness 研究，结构浅薄，无任务针对性。
</redoc-highlight>

---

## 各模型逐策略分析

### 评分维度说明

| 维度 | 说明 |
| --- | --- |
| **任务对齐** | 提案是否针对 Jigsaw 毒性/Civil Comments 任务（YES / PARTIAL / NO） |
| **结构完整** | 是否包含 &lt;problem&gt;、&lt;gap&gt;、&lt;key_insight&gt;、&lt;approach&gt; 四个完整 XML 块 |
| **创新性** | 是否提出超出简单微调的方法（多任务学习/去偏目标/层级分析等） |
| **异常标记** | 截断/模板泄漏/跑偏至其他领域 |

---

### claude-opus-4-6

**总体**：最高质量，5 条有效提案，1 条截断。

| 策略 | 任务对齐 | 结构完整 | 创新性 | 异常 |
| --- | --- | --- | --- | --- |
| full_refs (5d77c103) | PARTIAL | 完整 | 高（DIR 公平感知预训练目标） | — |
| full_refs (fc0feb72) | NO | 仅 thinking | — | **截断**：无 &lt;proposal&gt; |
| related_work | PARTIAL | 完整 | 高（反事实不变性预训练目标） | — |
| top_k_refs | YES | 完整 | 高（逐层注意力头偏差归因） | — |
| top_k_related_work | PARTIAL | 完整 | 中（公平感知预训练时间轨迹） | — |
| with_research_question | **YES** | 完整 | **最高** | — |

**with_research_question 亮点**（run_id ）：

提案直接命中 Civil Comments 评估框架：

- 明确提出「解耦身份词显著性与毒性信号」的研究问题

- 覆盖 BPSN AUC、BNSP AUC、subgroup AUC、加权幂均值四项指标

- 提出整合 RoBERTa 骨干 + 身份感知采样/损失加权 + 多任务子类型辅助头 + 子群校准后处理的完整训练方案

- Gap 分析具体引用了文献 [1]（nuanced bias metrics）和 [27]（counterfactual augmentation + logit pairing）

**top_k_refs 亮点**（run_id ）：

- 提出架构感知去偏：识别「偏差关键注意力头」（bias-critical attention heads），对身份标记词（「gay」「Muslim」）的注意力激活做反事实对比分析

- 明确引用 FPED/FNED 指标

- 两阶段方案：逐层偏差归因 → 定向正则化

---

### qwen25_7b_instruct_base

**总体**：5 条提案，仅 with_research_question 对齐任务，其余均为通用 NLP 公平性研究，提案短浅（4-5 KB）。

| 策略 | 任务对齐 | 结构完整 | 创新性 | 异常 |
| --- | --- | --- | --- | --- |
| full_refs | NO | 完整（浅） | 低 | — |
| related_work | NO | 完整（浅） | 低 | — |
| top_k_refs | NO | 完整（浅） | 低 | — |
| top_k_related_work | NO | 完整（浅） | 低 | — |
| with_research_question | YES | 完整 | 中 | — |

**with_research_question 亮点**（run_id ）：正确描述毒性分类问题、身份群体提及、子群 AUC 校准；提出多任务学习 + 身份感知采样 + 子群校准方案。但创新深度不及 Claude。

---

### exp09（top_k_refs SFT+RL）

**总体**：5 条提案，无一直接命中 Jigsaw 任务，with_research_question 下仍偏离（提案变成预训练范式对比研究）。

| 策略 | 任务对齐 | 结构完整 | 创新性 | 异常 |
| --- | --- | --- | --- | --- |
| full_refs | NO | 完整 | 中（合成数据偏差检测） | — |
| related_work | NO | 完整 | 低（预训练泛化研究） | — |
| top_k_refs | NO | 完整 | 低（通用多目标 ML） | — |
| top_k_related_work | PARTIAL | 完整 | 中（跨模型偏差分析） | — |
| with_research_question | NO | 完整 | 低 | **研究问题提示无效**：提案变成预训练范式比较研究 |

**诊断**：exp09 在面对 Jigsaw 公平性参考文献时，倾向于生成通用的「偏差检测框架」或「预训练泛化」提案，无法识别 Civil Comments 的具体评估协议。with_research_question 策略也未能将其拉回任务，说明该模型的 top_k_refs SFT 训练数据可能过于偏向通用 NLP 场景。

---

### exp10（related_work SFT+RL）

**总体**：5 条提案，with_research_question 下部分对齐（明确提及 BPSN/BNSP），其余策略通用。

| 策略 | 任务对齐 | 创新性 | 说明 |
| --- | --- | --- | --- |
| full_refs | NO | 低（LLM 训练可重复性） | — |
| related_work | NO | 低（高效迁移学习） | — |
| top_k_refs | PARTIAL | 中（分布偏移 + 群体公平） | 提及仇恨言论检测 |
| top_k_related_work | PARTIAL | 中（LLM 人口属性编码机制） | — |
| with_research_question | **YES** | 中 | 提及 BPSN/BNSP、身份感知 |

---

### exp11（topk_rw SFT+RL）⚠️

**总体**：**系统性跑偏**，多个策略下产出视频/多模态 VLM 提案。

| 策略 | 任务对齐 | 异常 |
| --- | --- | --- |
| full_refs | NO | **视频理解/时序视觉内容**，与 NLP 毒性任务完全无关 |
| related_work | NO | 通用 NLP 鲁棒性 |
| top_k_refs | NO | **多模态 LLM 视频-语言基准** |
| top_k_related_work | PARTIAL | 少数族裔歧视；无毒性针对性 |
| with_research_question | NO | 开放式生成评估框架；研究问题提示完全无效 |

**诊断**：exp11 在 topk_rw 策略下 SFT 训练时可能接触了大量视觉/多模态参考文献，或 RL 阶段的奖励信号无法有效区分 NLP vs. VLM 领域提案。这是目前最严重的领域跑偏问题，需要检查训练数据来源。

---

### exp12（research_q SFT+RL）

**总体**：5 条提案，质量参差，with_research_question 下意外退化（输出微调策略对比，不涉及毒性）。

| 策略 | 任务对齐 | 说明 |
| --- | --- | --- |
| full_refs | NO | 领域自适应 |
| related_work | NO | 预训练推理目标 |
| top_k_refs | PARTIAL | 公平性指标综述 |
| top_k_related_work | PARTIAL | PLM 预训练中的公平性 |
| with_research_question | NO | **退化**：变成微调策略对比研究 |

**诊断**：exp12 的 research_q SFT 训练应当最擅长处理 with_research_question 策略，但在迁移至 Jigsaw 任务时反而产生任务偏移。说明其 SFT 数据主要覆盖 dl_lr_schedule 类任务的研究问题格式，与 Jigsaw 毒性分类的问题模式存在 gap。

---

### exp13（full_refs SFT+RL）

**总体**：5 条提案，with_research_question 下准确命中任务，其余策略部分相关。

| 策略 | 任务对齐 | 说明 |
| --- | --- | --- |
| full_refs | NO | 常识推理 |
| related_work | PARTIAL | OOD + 人口偏差 |
| top_k_refs | PARTIAL | 偏差量化指标 |
| top_k_related_work | PARTIAL | 层级偏差分析 |
| with_research_question | **YES** | 准确描述d Civil Comments 身份词偏差问题 |

**with_research_question 亮点**（run_id ）：明确提出「毒性分类器对包含频繁被攻击群体词汇的非毒性评论打出虚高分数」这一问题，引用 Civil Comments + 加权偏差感知 AUC。

---

### exp14（full_refs LoRA）⚠️

**总体**：**模板泄漏**问题，所有 5 条提案的 XML 标签内残留问题模板文本。

<font color="#CC3030">**此后将不会评测exp14 checkpoint.**</font>

| 策略 | 任务对齐 | 异常 |
| --- | --- | --- |
| full_refs | NO | **模板泄漏**：&lt;problem&gt; 标签内含「What core research problem should be addressed?」 |
| related_work | NO | 模板泄漏 |
| top_k_refs | PARTIAL | 模板泄漏（反事实公平性，部分相关） |
| top_k_related_work | NO | 模板泄漏 |
| with_research_question | **YES** | 模板泄漏（但内容正确）：直接引用研究问题原文，提及 Civil Comments + BPSN/BNSP |

**诊断**：LoRA 精调可能降低了模型遵循指令格式的能力。提案格式训练时的 scaffold 模板文本「渗漏」进输出，表明 LoRA 对模板填写指令的泛化弱于全量 SFT。这个问题值得重点关注，即使提案内容有效，格式问题会导致 Worker 解析失败。

#### exp14 LoRA 模板泄漏：根本原因深度分析

**现象回顾**

exp14 在所有 5 条提案中，XML 标签内均残留提示词模板文本，例如：

```plaintext
<problem>
What core research problem should be addressed?
The core research problem is to develop...
</problem>
```

即模型先原封不动地复述了问题文本，再给出自己的答案。

**根本原因：训练数据无误，LoRA 容量不足**

定位过程验证了以下三个链路节点：

1. **Scaffold 文本在 User Prompt 中，而非训练目标**

2. train/prompt_builder.py 中的 _PROPOSAL_FORMAT 字符串将提示格式嵌入 **user 消息末尾**，其中包含如 &lt;problem&gt;What core research problem should be addressed?&lt;/problem&gt; 这样的带问号占位文本。这段文本 **不在 assistant 目标（cot_proposal）中**。

3. **训练目标是干净的**

4. 抽样检验 50 条 full_refs 策略的 SFT 样本，0/50 条的 cot_proposal（assistant 目标）含有 scaffold 问题文本。训练数据本身没有问题。

5. **verl 损失掩码实现正确**

6. verl MultiTurnSFTDataset 对每个 token 的 loss_mask 规则：assistant 消息 loss_mask=1，user/system 消息 loss_mask=0。User prompt 中的 scaffold 文本不会产生梯度更新。训练代码无 bug。

**为什么全量 SFT 能抑制，LoRA 不能？**

问题的核心在于 LoRA 可训练参数量远不足以克服基础模型的上下文复制倾向：

- **全量 SFT** 更新全部 ~7B 参数，模型可以学会「user prompt 中出现问题文本 → 在对应 XML 标签内直接填入答案，不复述问题」

- **LoRA**（r=64，all-linear）仅更新约 **0.5-1% 的参数**（~35-70M），适配器容量不足以充分覆盖所有层对格式条件的响应

- Qwen2.5-7B-Instruct 基础模型在预训练/指令微调中形成了「输入上下文出现在 &lt;tag&gt; 前时倾向于复述它」的偏置，少量参数更新后仍然残留

**格式设计的「对抗性」**

_PROPOSAL_FORMAT 的设计无意中放大了 LoRA 的容量问题：scaffold 问题文本（如「What core research problem?」）被嵌套在与答案相同的 XML 标签 &lt;problem&gt;...&lt;/problem&gt; 内部。「问题在标签内、答案也在标签内」的结构使得适配器难以学习选择性抑制——模型倾向于把整个标签内容原样复制再补充答案。

**修复方案**

| 方案 | 操作 | 代价 |
| --- | --- | --- |
| **① 修改格式模板（推荐）** | 将 _PROPOSAL_FORMAT 中的问题提示移出 XML 标签，或改为空标签 &lt;problem&gt;&lt;/problem&gt; | 改动最小；对所有模型都有效 |
| **② 增大 LoRA rank** | 将 lora_r 从 64 提高到 128 或更高 | 计算开销增加；不保证彻底消除 |
| **③ 全量 SFT 替代** | 用全量参数重跑 exp14 | 训练成本高；exp13（同配置全量 SFT）可作直接对照 |

**MLS 任务同样存在模板泄漏**

该问题并非 Jigsaw/MLE 任务特有。对 dl_lr_schedule 全量 20 样本统计，exp14 LoRA 的泄漏率如下：

| 策略 | 泄漏样本数 / 总数 |
| --- | --- |
| with_research_question | **20/20（100%）** |
| full_refs | 12/20（60%） |
| related_work | 10/20（50%） |
| top_k_refs | 9/20（45%） |
| top_k_related_work | 8/20（40%） |

MLS 任务下泄漏是**概率性**的（40–60%），并非每次必现，说明 LoRA 在大多数提示下仍无法完全抑制复述倾向，但偶尔能输出正确格式。with_research_question 策略 100% 泄漏，原因是该策略在 prompt 中已明确写出研究问题，给模型提供了更多可复制的上下文。

MLE 预览（n=1）呈现 100% 泄漏率，是小样本噪声所致，与 MLS 结果一致。

---

### exp15（FAS from exp09）⚠️

**总体**：**系统性跑偏至 VLM/多模态**，与 exp11 类似。

| 策略 | 任务对齐 | 异常 |
| --- | --- | --- |
| full_refs | NO | **视觉语言模型 CoT 推理基准** |
| related_work | NO | VLM CoT 推理基准（同上） |
| top_k_refs | NO | 多模态 VLM 公平性（视觉场景） |
| top_k_related_work | NO | 多模态模型安全/幻觉风险 |
| with_research_question | NO | **多语言视觉推理基准**；研究问题提示完全无效 |

**诊断**：exp15 是 exp09 的 FAS（Full-Answer Supervision）变体。exp09 本身领域对齐已较弱，FAS 训练进一步将其推向视觉/多模态方向。建议检查 FAS 监督数据中是否混入了大量视觉类论文的答案。

---

### exp16（full_refs 20×800）

**总体**：5 条提案，with_research_question 准确对齐，其余策略部分相关（偏 PLM 偏差分析）。

| 策略 | 任务对齐 | 说明 |
| --- | --- | --- |
| full_refs | PARTIAL | PLM 编码人口关联机制 |
| related_work | PARTIAL | PLM 跨多样性评估 |
| top_k_refs | PARTIAL | 隐式偏差在 PLM 嵌入中的位置 |
| top_k_related_work | PARTIAL | PLM 预训练中的人口偏差来源 |
| with_research_question | **YES** | 毒性分数虚高 + 身份群体 + 保护社群 |

**特点**：exp16 在非 rq 策略下虽然没有精准命中 Jigsaw 任务，但都停留在 PLM 偏差分析领域，属于「主题相关但具体任务偏离」的状态，不像 exp11/exp15 那样完全跑偏。

---

### exp17（top_k_refs PPL RL）

**总体**：5 条提案，with_research_question 对齐（但提出多语言泛化角度），部分策略有异常（full_refs 产出 NLP 伦理政策论文）。

| 策略 | 任务对齐 | 说明 |
| --- | --- | --- |
| full_refs | NO | **NLP 伦理/政策论文**（不是实验性研究） |
| related_work | PARTIAL | PLM 偏差评估框架 |
| top_k_refs | PARTIAL | 政治偏见 + 毒性检测（交叉领域） |
| top_k_related_work | PARTIAL | Transformer 层级偏差分析 |
| with_research_question | **YES** | 毒性检测多语言泛化 + 子群公平性 |

---

## 跨模型对比总结

### 任务对齐得分（YES=2, PARTIAL=1, NO=0，满分 10）

| 模型 | full_refs | related_work | top_k_refs | top_k_rw | with_rq | 总分 |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 1 | 1 | 2 | 1 | 2 | **7** |
| exp13_full_refs | 0 | 1 | 1 | 1 | 2 | **5** |
| exp16_full_refs_20x800 | 1 | 1 | 1 | 1 | 2 | **6** |
| exp10_related_work | 0 | 0 | 1 | 1 | 2 | **4** |
| exp14_full_refs_lora | 0 | 0 | 1 | 0 | 2 | **3** |
| exp17_top_k_refs_ppl | 0 | 1 | 1 | 1 | 2 | **5** |
| exp12_research_q | 0 | 0 | 1 | 1 | 0 | **2** |
| exp09_top_k_refs | 0 | 0 | 0 | 1 | 0 | **1** |
| qwen25_7b_base | 0 | 0 | 0 | 0 | 2 | **2** |
| exp11_topk_rw | 0 | 0 | 0 | 1 | 0 | **1** |
| exp15_fas_exp09 | 0 | 0 | 0 | 0 | 0 | **0** |

### 策略效果对比

| 策略 | 对齐比例（n=11） | 说明 |
| --- | --- | --- |
| with_research_question | **8/11（73%）** | 显著最优 |
| top_k_related_work | 5/11（45%） | 次之 |
| top_k_refs | 4/11（36%） | 中等 |
| related_work | 3/11（27%） | 较低 |
| full_refs | 2/11（18%） | 最低（含截断/跑偏） |

---

## 异常问题汇总

| 问题类型 | 影响模型 | 影响条数 | 建议 |
| --- | --- | --- | --- |
| 系统性跑偏至视觉/VLM 领域 | exp11、exp15 | 5+5=10 | 检查训练数据中视觉类论文占比；考虑数据清洗 |
| 模板泄漏（scaffold 问题文本残留） | exp14 LoRA | 5 | LoRA 格式泛化能力弱于全量 SFT；建议全量复现 |
| 生成截断（无 &lt;proposal&gt; 输出） | claude-opus-4-6 | 1 | 单次失败；重试即可 |
| with_research_question 提示无效 | exp09、exp11、exp15 | 3 | 模型未能跟随研究问题上下文 |
| 内容退化为政策/综述论文（非实验方案） | exp17 full_refs | 1 | 轻微；full_refs 策略本身信噪比低 |

---

## 结论与后续建议

<redoc-highlight emoji="dui" fillColor="green">
**确认有效的做法**

- 策略对所有模型都有显著的任务对齐改善效果，应作为 Jigsaw 等 MLE 任务的首选策略。

- claude-opus-4-6 在所有策略下均表现最稳定，可作为提案质量的上界参考。

- exp13/exp16 表现较好，with_rq 下提案质量接近 Claude。
</redoc-highlight>

<redoc-highlight emoji="gantanhao" fillColor="red">
**需要处理的问题**

- **exp11 和 exp15** 存在系统性领域跑偏（→视觉/VLM）：在运行正式 MLE 评估前，需要确认这两个模型是否具备生成 NLP 类提案的能力，或者其跑偏是 MLE 任务通用问题还是 Jigsaw 特有问题。

- **exp14 LoRA 模板泄漏**：该模型格式稳定性差，建议在 MLE 任务 Pipeline 中加入 XML 格式校验步骤，检测 &lt;problem&gt; 等标签内是否残留模板问题文本。

- **Jigsaw 任务的 non-rq 策略普遍对齐率低**：若要在 full_refs/related_work 策略下获得有意义的评估，需要重新审视 frontline papers 的选择是否有效将模型引导至 Civil Comments 任务上下文。
</redoc-highlight>

<redoc-highlight emoji="dengpao" fillColor="blue">
**下一步建议**

1. 优先在 with_research_question 策略下为 Jigsaw 启动正式多样本评估（n=20），选择 claude-opus-4-6、exp13、exp16 作为首批对象。

2. 对 exp11/exp15 在其他 MLE 任务（smartphone-decimeter、petfinder）的提案进行检查，判断 VLM 跑偏是否为 Jigsaw 特有现象。

3. 为 MLE 评估 Pipeline 增加提案格式合规检查：验证 &lt;problem&gt;/&lt;approach&gt; 标签存在且不含模板文本。

4. 考虑为 Jigsaw 任务补充更多任务专属 frontline papers（直接针对 Civil Comments 数据集的方法论论文），降低模型在非 rq 策略下对齐的难度。
</redoc-highlight>
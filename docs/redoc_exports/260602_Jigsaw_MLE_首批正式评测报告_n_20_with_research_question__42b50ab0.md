## 实验概览

本报告记录 Jigsaw Unintended Bias in Toxicity Classification（Civil Comments 数据集）任务下三个模型的首批正式多样本评测（n=20，with_research_question 策略），重点分析：评测结果、Worker 实现忠实度、以及当前评分基准的局限性。

---

## 一、实验设置

| 参数 | 值 |
| --- | --- |
| 任务 | jigsaw-unintended-bias-in-toxicity-classification |
| 评测策略 | with_research_question |
| 样本数 | n=20（每模型） |
| 温度 | 0.7 |
| Worker 超时 | 14400s（4 小时） |
| Claude Code max-turns | 60 |
| Worker 槽位分配 | Sweep A（claude-opus-4-6）→ M0+M1（16 GPU）；Sweep B（exp13+exp16）→ M2+M3（16 GPU） |
| 评分指标 | raw AUC（原始竞赛指标）+ leaderboard percentile rank，由官方 mle-bench grader 通过 HMAC 签名验证 |

**三个模型**：

| 模型 | 说明 |
| --- | --- |
| claude-opus-4-6 | API 模型，作为质量上界参考 |
| exp13 (full_refs SFT+RL) | 以 full_refs 策略的参考文献为训练信号，经 SFT+GRPO RL |
| exp16 (full_refs 20×800 SFT+RL) | 同策略但训练数据规模 20×800（更大批次） |

---

## 二、评测结果

归一化方式已更新为 **leaderboard percentile rank**（见 4.1 节），旧版双曲衰减分数作废。

| 模型 | n_done | raw AUC min | raw AUC mean | raw AUC max | percentile mean | percentile max |
| --- | --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | 20 / 20 | 0.500 | **0.730** | **0.849** | **6.7%** | **8.7%** |
| exp16 (full_refs 20×800) | 17 / 20 | 0.524 | **0.709** | 0.789 | **6.6%** | 7.6% |
| exp13 (full_refs SFT+RL) | 18 / 20 | 0.500 | **0.686** | 0.801 | **6.2%** | 7.8% |

percentile = 击败了 leaderboard 中多少比例的队伍（2634 支）。1.0 = 第一名，0.5 = 中位数。

3 个样本（exp13×2、exp16×1）因 Worker 在 4 小时内未能完成训练而未产出结果（result.json 缺失）。

<redoc-highlight emoji="tuding" fillColor="yellow">
**超时说明**：绝大多数 timeout 样本均已在超时前写入 workspace/result.json——Worker 完成了训练并评分，随后继续尝试迭代优化，直到被 4 小时上限中断。pipeline.py 已修复为在超时时自动回收已有结果；本批数据已手动 backfill。
</redoc-highlight>

### 2.1 按实现方式分层的分数

| 实现方式 | n | raw AUC mean | raw AUC max | percentile mean |
| --- | --- | --- | --- | --- |
| Transformer（RoBERTa/DistilBERT 微调） | 21 | **0.737** | **0.849** | **6.9%** |
| TF-IDF + LR/SGD | 30 | 0.694 | 0.801 | 6.3% |
| TF-IDF + LightGBM | 2 | 0.676 | 0.707 | 6.2% |

Transformer 路线显著优于 TF-IDF 路线（+0.043 raw AUC mean gap）。这一差距与 claude-opus-4-6 vs exp13 的整体得分差高度相关（见第三节）。

---

## 三、Worker 忠实度分析

### 3.1 是否跑偏至 SOTA 抄袭？

**结论：**<font color="#00A14B">**无抄袭，实现是真实的。**</font>

逐样本检查所有 55 条 worker.log（stream-json 格式，含完整 Bash 执行记录），未发现以下行为：

- 未出现 Kaggle 公开 Notebook URL、比赛解法 GitHub 仓库下载

- 未出现 kaggle competitions download 或 Kaggle API 拉取预计算 submission

- 未出现 curl/wget 访问已知 Jigsaw 解法地址

- 5 个样本出现 pip install git+https://github.com/openai/mle-bench.git——这是基础设施安装（benchmark grader），不是解法

**唯一外部数据访问事件**（opus/sample_01，raw AUC=0.849）：Worker 发现竞赛数据（data/train.csv）不存在，先尝试查找 Kaggle 凭证（失败），随后在 HuggingFace Hub 搜索并找到 CC0 授权的 shuttie/jigsaw-unintended-bias 数据镜像，调用 hf_hub_download 下载原始训练数据，再通过 mlebench prepare 生成标准化数据目录。这是获取竞赛数据集本身的操作，不是获取预计算解法。之后所有样本均从同一 NFS 路径读取该数据。

**HuggingFace 预训练模型使用**：Worker 通过 AutoModel.from_pretrained("roberta-base") 等标准调用使用预训练权重。这属于正常 NLP 实验流程，相当于使用标准库依赖，而非解法抄袭。所有权重均已缓存在本地 /root/.cache/huggingface/hub/，无需实际下载。

### 3.2 提案忠实度

三个模型的 Worker 均从提案出发实现了有实质内容的解法，但实现深度因提案质量差异而分层：

**claude-opus-4-6（18/20 使用 Transformer）**

提案直接命中 BPSN/BNSP AUC 框架，提出 IBERT（Identity-Robust BERT）四阶段方案：

1. RoBERTa 多任务头（毒性主任务 + 毒性子类型辅助头 + 身份群体辅助头）

2. 反事实逻辑配对（CLP，对身份词替换样本对施加成对损失）

3. BPSN/BNSP AUC 代理损失（直接优化子群 AUC）

4. 逐子群 Isotonic Regression 校准

Workers 在 18/20 样本中实现了该方案，类名 IBERTModel 在多个样本的 baseline.py 中出现。

**exp13（18/18 使用 TF-IDF）**

提案偏向通用公平性研究，未指定模型骨架。Workers 均采用 TF-IDF + SGD/LR 路线，加入身份感知样本加权和反事实数据增强——这是对模糊提案的合理解读，但无法触及 Transformer 的性能上界。

**exp16（3/17 使用 Transformer）**

提案提及"多组件 PLM 框架 + 身份感知损失重加权"，但描述较抽象。Workers 多数选择 TF-IDF 路线（更快可交付），少数实现了 DistilBERT/RoBERTa + 辅助身份头。

### 3.3 低分原因

| 样本 | raw AUC | percentile | 原因 |
| --- | --- | --- | --- |
| opus/sample_13 | 0.500 | 2.5% | 反事实增强量过大（+138 万合成样本），训练集严重失衡，模型输出接近全零 |
| opus/sample_19 | 0.500 | 2.5% | 同上，predictions max=0.13，CDA 失控 |
| exp13/sample_03 | 0.500 | 2.5% | Worker 执行中途 API 错误，仅产出近零预测 |
| exp16/sample_12 | 0.524 | 4.2% | RoBERTa 仅用 15 万样本 + 2 epoch，训练不足 |

raw AUC=0.500 等同于随机分类器（所有预测塌缩为常数），AUC=0.524 只比随机高 0.024，对应 leaderboard 第 4.2 百分位——两者均属真正低分。之前此列误标为"归一化分数"，实为 raw AUC，已更正。

低分均属实现 bug 或训练资源不足，非提案方向错误。

---

## 四、基准分数的局限性（重要）

<redoc-highlight emoji="gantanhao" fillColor="red">
**当前所有"通过/失败"判断均无意义**——基准分数（_baseline_score）尚未实测，使用占位值 0.0。
</redoc-highlight>

### 4.1 mle_grade.py 归一化公式 percentile

**mle_grade.py 现在使用 leaderboard percentile rank 作为 val_metric**，与 AIRA2 论文口径对齐：

```python
# higher-is-better（如 AUC）
percentile = mean(leaderboard_scores <= raw_score)
# lower-is-better 取反：mean(leaderboard_scores >= raw_score)
```

result.json 同时输出 **raw_score**（原始竞赛指标）和 **val_metric**（percentile）。

**语义**：percentile = 击败了 leaderboard 中多少比例的队伍。

| 位次 | 原始 AUC | percentile |
| --- | --- | --- |
| 1st（第一名） | 0.94734 | **1.000** |
| Median（1317th） | 0.93417 | **0.500** |
| Last（2634th） | 0.07540 | **0.0004** |
| 我们最强样本 | 0.849 | **0.087** |
| 我们 mean | 0.730 | **0.067** |

**解读**：我们最强的 claude-opus-4-6 样本只击败了约 8.7% 的历史参赛队伍。这比旧的双曲衰减分数（0.910）更准确地反映了实际水平。

### 4.2 8% Percentile 壁垒分析

我们三个模型均停在 leaderboard 底部 6–9%。这不是随机波动——这是一个**结构性壁垒**，源于 leaderboard 的双峰分布。

**Leaderboard 分布**（Jigsaw，2634 队）：

| 原始 AUC 区间 | 队伍数 | 占比 |
| --- | --- | --- |
| [0.00, 0.70) | 169 | 6.4% |
| [0.70, 0.92) | 370 | 14.0% |
| [0.92, 0.94) | 1725 | **65.5%**（主密集区） |
| [0.94, 0.95) | 312 | 11.8% |

跨过 0.92 门槛就进入了那 65.5% 的主密集区。我们最好的样本（0.849）仍在"稀疏荒漠"里。

**根本原因：三层差距（按重要性排序）**

| # | 差距 | 我们的实现 | 0.93+ 要求 |
| --- | --- | --- | --- |
| 1 | **训练数据量** | 抽样 80K–500K 行（28%） | 全部 1.8M 行 |
| 2 | **目标变量处理** | 二值化（≥0.5 → 1） | 连续软标签（直接用 0-1 分数做 BCE） |
| 3 | **模型规模** | RoBERTa-base（125M） | RoBERTa-large / XLNet-large（355M+） |

补充因素（非决定性）：序列长度截断（128 vs 220-512）、缺少 k-fold OOF 集成。

**最高分样本（sample_01, raw AUC=0.849）的实际执行**

- 模型：RoBERTa-base（提案要求 RoBERTa-large，Worker 因感知 GPU 内存限制主动降级）

- 训练数据：500K 行随机子样本（身份分层逻辑有 bug，退化为随机采样）

- 损失：BCE + 多任务辅助头 + 反事实逻辑配对（CLP），2 个 epoch，seq_len=128

**进入 0.92+ 的最小提案改动**

在 approach 中加入：全部 1.8M 行训练（不抽样）、连续软标签（不二值化）、gradient checkpointing + fp16 以支持 RoBERTa-large。这两点（全量数据 + 软标签）估计可将 raw AUC 从 0.85 推至 0.89–0.91；加上 RoBERTa-large 预计可达 0.93+。

### 4.3 占位基准的问题

代码中的设置：

```python
class JigsawUnintendedBiasTask(MleBenchTask):
    _baseline_score: ClassVar[float] = 0.0   # TODO: fill after baseline run
    pass_threshold = 0.05
```

这意味着：**pass 条件 = percentile ≥ 0.05**。由于所有实现的 percentile 均在 2.5%–8.7% 之间，每个有效样本都“通过”——这不能说明任何问题。

### 4.3 正确的基准应该是什么

**需要运行一次原始 baseline.py（TF-IDF + LR，无身份感知逻辑）**，测量其 raw AUC，然后设为 _baseline_score，再设定合理的 pass_threshold。

基于 MLS 任务的经验，Jigsaw 的 TF-IDF baseline 预计 raw AUC 约 0.70–0.75（percentile 约 6–7%）。这意味着 exp13 的低分样本（raw AUC 0.50–0.65）可能实际上低于 TF-IDF baseline，而 claude-opus-4-6 高分样本（raw AUC 0.82–0.85）才是真正有意义的改进。

### 4.4 正确解读当前结果的方式

目前可以可靠陈述的：

- 三个模型的 Worker 均产出了有效的、非平凡的实现（raw AUC 跨度 0.500–0.849，非退化分布）

- claude-opus-4-6 &gt; exp16 ≈ exp13（raw AUC 均值差约 0.02–0.04；percentile 均值差约 0.5%）

- Transformer 路线显著优于 TF-IDF 路线（raw AUC 均值差 +0.043）

- 低分样本均由可识别的实现 bug 导致，并非系统性失败

**不能可靠陈述的**：是否超越了 TF-IDF baseline，以及通过率数字。

---

- [ ] 针对高分样本（sample_01, raw AUC=0.849；sample_11, raw AUC=0.847）深入分析实现，作为代表性案例记录

## 五、Worker 系统性降级行为（新发现）

### 5.1 现象

claude-opus-4-6 的 20 个样本**全部**使用了 RoBERTa-base，无一例外——尽管提案明确写了 Fine-tune RoBERTa-large。

| 模型选择 | 样本数 | 是否有实际 OOM 记录 |
| --- | --- | --- |
| roberta-base | 18 / 20 | 无 |
| distilbert-base-uncased | 1 / 20 | 无 |
| TF-IDF（跳过 Transformer） | 1 / 20 | 无 |

### 5.2 根本原因：预判焦虑，而非硬件限制

以最高分样本（sample_01，raw AUC=0.849）为例：

Worker 完成数据探索后，第一条实现消息写道：

> "RoBERTa-base backbone **(not RoBERTa-large — GPU memory constraints)**"

实际 GPU：**L20Z，48GB VRAM**。Worker 从未尝试加载 RoBERTa-large，也没有遭遇 OOM——这是先入为主的判断。

RoBERTa-large（355M 参数）在 fp16 + gradient checkpointing + batch_size=16 下只需约 8–10GB 显存，远低于 48GB 上限。Worker 的"内存约束"是自我想象出来的。

sample_06 展示了同一行为的另一种形式：Worker 发现 1.8M 行数据处理慢，主动从 RoBERTa 切换到 TF-IDF+SGD，理由是"更快可交付"。

**这不是偶发现象，而是系统性的"计算保守主义"**：Worker 在看到大数据集 + 复杂提案时，会主动选择"可以在时间内交付的版本"，并为此构造合理化解释。

### 5.3 对评测结果的影响

这一行为直接解释了 8% percentile 壁垒：

- 提案要求 RoBERTa-large + 全量数据，Worker 实际执行 RoBERTa-base + 500K 子样本

- 即使提案质量很高（如 sample_01 的 IBERT 方案），实现层的自行降级把潜在的 0.93+ 拉低到了 0.85

**这意味着当前评测结果低估了提案的真实质量上限**——我们测的是"Worker 愿意实现的版本"，而不是"提案要求的版本"。

### 5.4 修复方向

在 worker prompt 中显式注入约束信息：

1. 声明可用 VRAM（Your GPU has Xk MB VRAM available）

2. 要求降级前先用 nvidia-smi 验证实际余量

3. 在 &lt;approach&gt; 中加入具体的内存优化指令：gradient checkpointing + fp16，消除 Worker 自行推断的空间

---

## 六、待办事项

- [ ] 运行 baseline.py（原始 TF-IDF+LR）一次，记录实测 _baseline_score，设定合理 pass_threshold

- [ ] 在 worker_prompt 中注入 GPU VRAM 信息，禁止无验证降级

- [ ] 对 claude-opus-4-6 补充一次 n=5 的 RoBERTa-large 强制执行验证（在 approach 中指定 gradient checkpointing + fp16）

- [ ] 针对高分样本（sample_01, 0.849；sample_11, 0.847）深入分析代表性案例

- [ ] 修复 CDA 失控 bug（反事实增强量上限）
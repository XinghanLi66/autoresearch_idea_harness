## 一、概览

本文档对 with_research_question 策略下的6组基准评测进行深入分析，涵盖 **Proposal 质量**（模型是否提出了有意义的想法）与 **Worker 忠实度**（Worker 是否忠实执行了 Proposal）两个维度。

| 任务 | 模型 | n | Pass | Pass@1 | Mean | Max | Pass 阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dl_lr_schedule | claude-opus-4.6 | 20 | 1 | 5.0% | 92.57% | 93.23% | ≥93.01% |
| dl_lr_schedule | qwen25_7b base | 18 | 7 | 38.9% | 92.86% | 93.26% | ≥93.01% |
| dl_activation_function | claude-opus-4.6 | 20 | 1 | 5.0% | 92.77% | 93.31% | ≥93.27% |
| dl_activation_function | qwen25_7b base | 20 | 0 | 0.0% | 92.87% | 93.20% | ≥93.27% |
| cv_data_augmentation | claude-opus-4.6 | 20 | 0 | 0.0% | 93.06% | 93.55% | ≥93.87% |
| cv_data_augmentation | qwen25_7b base | 20 | 0 | 0.0% | 93.22% | 93.78% | ≥93.87% |

**核心发现**：Pass 率极低（多数为 0%）。但这并不意味着模型没有提出有价值的想法，而是 Pass 阈值本身设计偏高，且存在 **dl_lr_schedule 任务结构性缺陷**（详见第五节）。

---

## 二、Proposal 质量分析

### 2.1 dl_lr_schedule：提案收敛于同一模式

两个模型均提出了"warmup + cosine decay + 非零 floor"的学习率调度方案。

**Opus-4.6 典型提案（sample_15，唯一通过样本，93.23%）**：

- 提出"Curvature-Gated Warmup"：根据 arch 类型（resnet56 / mobilenetv2 等）动态调整 warmup 长度

- 主体为 warmup → 单段 cosine 衰减 → 最后 10% 触发短 cosine restart

- min_lr = base_lr * 0.001，明确设置非零 floor

**Base 模型典型提案（sample_01，93.26%）**：

- 方案相对简单：5 epoch warmup，线性升温后 cosine 衰减

- min_lr = 1e-4，同样设置非零 floor

- Proposal 主体抽象（"How would the proposed method work..."），Worker 自主实现细节

**结论**：dl_lr_schedule 存在结构性问题——任务描述明确提及 "warmup+cosine"，Worker 无论提案内容如何均会实现标准方案，通过率由 min_lr 是否设置决定，而非提案创新性。此任务建议退役。

### 2.2 dl_activation_function：提案有差异化，实现趋于同质

Opus-4.6 的 20 份提案呈现明显分层：

- **11/20 提出 Learnable Swish**（参数化 x * sigmoid(β·x)，β 可学习）：多数附有"spectral coupling"等理论包装

- **5/20 提出 Mish 变体**（x * tanh(softplus(x))）

- **2/20 提出标准 Swish**（无可学习参数）

- **2/20 提出 ELU 变体**（sample_04，91.69%，最低分）

Base 模型方案：

- **9/20 Learnable Swish**，**9/20 Mish 变体**，**2/20 标准 Swish**

- 无 ELU 等偏离分布的方案

**通过样本分析（Opus sample_03，93.31%）**：

```python
self.beta = nn.Parameter(torch.ones(1))
self.alpha = nn.Parameter(torch.ones(1))

def forward(self, x):
    return self.alpha * x * torch.sigmoid(self.beta * x)
```

Proposal 中理论包装为"Spectral Coupling Parameters"，实质是双参数 Learnable Swish，Worker 忠实实现了该设计。

**关键洞察**：Learnable Swish（β·Swish）在激活函数任务上具有优势。Base 模型 best=93.20%（差 0.07pp），Opus best=93.31%（通过）。两者实现质量几乎相同，差异来自随机性而非提案创新性。

### 2.3 cv_data_augmentation：提案华丽，实现退化

这是最有趣的一组结果：模型提出了极具创意的复杂方案，但 Worker 最终实现均退化为简单的现成 API。

**Opus-4.6 提案创意统计**：

- **9/20 AutoAugment 系列**（含 CIFAR10 policy）

- **7/20 TrivialAugmentWide**

- **4/20 AugMix**（混合多个增广链）

- 部分提案有自定义 FrequencyBandDrop、FrequencyDomainMask 等频域增广类，但最终未通过

**Base 模型实现统计**：

- **13/20 AutoAugment**

- **6/20 TrivialAugmentWide**

- **1/20 AugMix**

**Pass 阈值 93.87% 的挑战**：Cutout（93.67%）+ 0.20pp buffer 设计合理，但已知最强现成方案 TrivialAugment 本地测值为 93.57%，AugMix 未测，Worker 的组合策略（如 AutoAugment + RandomErasing）也未能突破阈值。

**Base 最近距离（sample_17，93.78%）**：

```python
transforms.RandAugment(num_ops=2, magnitude=9)
...
transforms.RandomErasing(p=0.5, ...)
```

使用 RandAugment + RandomErasing 组合，差 0.09pp 未能通过。

---

## 三、Worker 忠实度分析

### 3.1 忠实度评估框架

分析 Proposal &lt;approach&gt; 标签 vs editable_region.py 实现的对应关系。

### 3.2 dl_lr_schedule

| 模型 | 忠实度 |
| --- | --- |
| Opus-4.6 | 高——Proposal 通常给出具体参数（warmup fraction, min_lr），Worker 严格实现 |
| Base | 中——Proposal 抽象（"Define the LR schedule..."），Worker 根据 docstring 自主实现 |

**Base 的特殊现象**：Base 模型 Proposal 质量较低（多以"How would the proposed method work?"开头，未给出具体方案），但 Worker 根据 editable_region.py 的 docstring（含 Design considerations）自行实现了合理方案，Pass 率反而更高（38.9% vs 5.0%）。

这说明对于 dl_lr_schedule，**Worker 受 docstring 约束更强，受 Proposal 约束较弱**。

### 3.3 dl_activation_function

忠实度高：Proposal 给出激活函数设计思路（learnable parameter、spectral coupling 等），Worker 实现中均可看到对应的 nn.Parameter 和 forward 逻辑。

偏差来源：Opus sample_04（91.69%，最低分）提案描述复杂的 Spectral Co-Design，Worker 却实现了简单 F.elu(x, alpha=1.0)——是最不忠实的样本，也是最差结果。

### 3.4 cv_data_augmentation

**低忠实度**是这组结果的核心问题：

- Opus sample_02 提案设计了复杂 FrequencyBandDrop 频域遮蔽类（约 60 行代码），Worker 确实实现了该类，并组合 AutoAugment + FrequencyBandDrop + RandomErasing，最终 92.07%——实现复杂度最高，但分数最低

- Opus sample_04 提案提出 "Spectral Co-Adaptive Search"，Worker 只用了 TrivialAugmentWide + RandomErasing（93.49%）——实现简单，但分数接近最高

- Base 模型：Proposal 普遍抽象（Bayesian 优化、NAS co-search），Worker 均实现简单 API 组合

**核心矛盾**：复杂提案 → 复杂实现 → 分数更低；简单提案 → 简单 API → 分数更高。Worker 在实现复杂自定义类时引入了 bug 或参数不优。

---

## 四、通过样本深析

### 4.1 dl_lr_schedule Opus sample_15（93.23%，通过）

Proposal 核心：根据 arch 参数动态调整 warmup 长度，主体 cosine 衰减带 min_lr = base_lr * 0.001，最后 10% 加 short cosine restart。

实现关键：

- warmup_epochs 对 resnet56/mobilenetv2 更长（避免早期震荡）

- restart_start = int(0.90 * total_epochs) 触发最终短 restart

- 非零 floor（0.001 * base_lr）是通过的关键——参考基线 warmup_cosine 衰减至 0

### 4.2 dl_lr_schedule Base sample_01（93.26%，通过，最高分）

实现最简洁：线性 warmup 5 epoch，cosine 衰减，min_lr = 1e-4。Proposal 几乎无指导价值，Worker 凭 docstring 自主完成。

### 4.3 dl_activation_function Opus sample_03（93.31%，通过）

Proposal 包装为"Spectral Coupling"，实质双参数 Learnable Swish：α * x * sigmoid(β * x)。两个可学习参数（α 控制缩放、β 控制门锐度）与 BatchNorm + ResNet 残差结构配合良好。

**为何通过**：learnable_swish 在该 ResNet-20/CIFAR-10 设置下有微小优势，β 和 α 参数提供了额外自由度，训练收敛到更优点。

---

## 五、结论与启示

### 5.1 dl_lr_schedule 任务缺陷（建议退役）

任务参考基线 warmup_cosine 衰减至精确的 0 LR（92.71%）。由于 docstring 明确列出 "Warmup phase"/"Decay shape"/"Final learning rate"，Worker 在 100% 的样本中实现了 warmup+cosine，96% 设置了非零 floor，使任何非零 floor 实现都能通过（通过门槛 93.01%）。通过率与 Proposal 内容无关，仅取决于 Worker 是否设置 min_lr。

### 5.2 dl_activation_function 任务有效

两个模型均集中于 Learnable Swish 和 Mish 变体，实现质量决定最终分数。Pass 阈值（93.27%）接近但不超过已知方法，给模型留出空间。建议保留此任务。

### 5.3 cv_data_augmentation 阈值偏高

Base 最高 93.78%（差 0.09pp），Opus 最高 93.55%（差 0.32pp）。现成 API 组合（RandAugment + RandomErasing）已非常接近阈值。Cutout+0.20pp 的 buffer 对 7B 模型过于严苛。建议将阈值调整为 ≥93.70%（Cutout+0.03pp），使得 base 最优样本能通过，鼓励探索。

### 5.4 训练信号质量

- with_research_question 策略使模型生成了有主题的 Proposal（光谱分析、联合搜索、mean-field 理论等），但 Worker 实现往往退化为简单 API

- Base 模型 Proposal 质量低，但 Worker 表现不差（依赖 docstring 导引）

- **核心问题**：当前奖励信号（Pass/Fail）无法区分"有意义提案+好实现"与"无意义提案+好实现"，RL 训练将强化后者

### 5.5 下一步建议

1. **退役 dl_lr_schedule**，引入阈值更难的任务（如非零 floor 无法通过的参考基线）

2. **调低 cv_data_augmentation 阈值**至 ≥93.70%

3. **增加 Proposal 忠实度奖励**：奖励 Worker 实现的方案与 Proposal 描述的关键特征（如可学习参数、特定增广操作）匹配

4. **设计更好的评测**：区分 Worker 是否真正实现了 Proposal，还是仅凭 docstring 自行决定

---

_分析时间：2026-05-27。原始数据：proposal_rl/runs/benchmark/ 下 6 个 run 目录，共 118 个样本。_
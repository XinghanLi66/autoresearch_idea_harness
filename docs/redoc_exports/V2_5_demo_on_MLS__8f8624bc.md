## 读者导览

这篇 demo 记录三个 V3 SFT checkpoint 在 MLS task 上的 case study：dl_activation_function、dl_regularization、dl_weight_initialization。关注点不是最终分数，而是一个更基础的问题：

<redoc-highlight emoji="fangdajing" fillColor="yellow">
**我们要判断 master proposal 是否已经足够具体，以及 worker 是否忠实地把 proposal 落到了代码里。**
</redoc-highlight>

核心结论先说清楚：

- <font color="#1B7F3A">进展</font>：三个 case 中，worker 大多不是乱做。activation 和 weight initialization 的代码实现与 proposal 高度一致，regularization 也实现了主体机制。

- <font color="#D97706">方法重点</font>：这三个 demo 把 evaluation 拆成三层：master 提了什么、worker 做了什么、benchmark 结果如何。

- <font color="#C62828">主要缺陷</font>：proposal 仍然偏“概念正确但机制浅”。它们能生成像样的 XML 和常见 idea，但缺少稳定初始化、约束、schedule、ablation 和 failure handling。

## 总览表

| Case | Task / Subtask | Master idea | Worker 忠实度 | SFT metric | Pass line | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | dl_activation_function / resnet20-cifar10 | learnable smooth non-monotonic activation | 高 | 92.67 | 93.35 | 忠实但 idea 增量小 |
| 2 | dl_regularization / resnet56-cifar100 | orthogonal regularization + confidence penalty | 中 | 71.53 | 73.00 | 主体忠实，但 adaptive 部分缺失 |
| 3 | dl_weight_initialization / resnet56-cifar100 | He + orthogonal + Fixup-like initialization | 高 | 72.62 | 73.26 | 忠实但机制组合不稳妥 |

### 和 32B base proposal 的对比

| Task | Benchmark baseline | 32B base proposal result | V1-semantics CoT SFT proposal result | SFT vs base proposal |
| --- | --- | --- | --- | --- |
| dl_activation_function | 92.97 | 92.79 | 92.67 | -0.12 |
| dl_regularization | 72.99 | 72.29 | 71.53 | -0.76 |
| dl_weight_initialization | 73.25 | 72.91 | 72.62 | -0.29 |

<redoc-highlight emoji="cuo" fillColor="red">
**Takeaway**：这三个样例都没有支持“轻量 SFT 已经提升 proposal quality”。更准确的说法是：SFT 学到了格式和常见术语，但还没有学会提出稳健、细粒度、可超过 strong baseline 的机制。
</redoc-highlight>

## Workflow 可视化

```plaintext
V3 strict TeX target SFT checkpoint
        |
        v
MLS task packet
        |
        v
master proposal
        |
        +--> inspect proposal: idea granularity / mechanism / risk
        |
        v
Claude worker implementation
        |
        +--> inspect editable_region.py: faithful or drifted?
        |
        v
MLS benchmark result
        |
        +--> inspect eval.log: stable training, NaN, timeout, or real metric?
```

这个 demo 的判断逻辑是：如果 worker 不忠实，那么问题主要在 worker interface；如果 worker 忠实但结果差，那么问题更可能在 master proposal 的机制质量。

---

## Case 1: Activation Function

### Motivation

dl_activation_function 是一个比较干净的 task：worker 只需要替换 activation function，不涉及复杂训练循环修改。因此它适合检查 master 是否能提出一个可直接落地的函数形式。

### Method

<redoc-highlight emoji="tuding" fillColor="orange">
**Master proposal**：提出一个 learnable smooth non-monotonic activation，形式接近 x * tanh(beta * softplus(x))，其中 beta 是可学习参数。
</redoc-highlight>

Worker 最终实现：

```python
class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * torch.tanh(self.beta * F.softplus(x))
```

### Result

| Item | Value |
| --- | --- |
| Benchmark baseline | 92.97 |
| Pass line | 93.35 |
| SFT proposal result | 92.67 |
| Passed | false |

训练日志显示过程稳定，没有 NaN，但最终 accuracy 低于 pass line，也低于 benchmark baseline。

```plaintext
epoch=180 test_acc=92.14
epoch=190 test_acc=92.45
epoch=200 test_acc=92.65
TEST_METRICS: test_acc=92.67
```

### Takeaway

<redoc-highlight emoji="dui" fillColor="green">
**Worker 忠实度高**：proposal 中的核心公式被直接实现，worker 没有明显偏离 master。
</redoc-highlight>

<redoc-highlight emoji="cuo" fillColor="red">
**但 proposal 质量有限**：这个 idea 更像 Mish / softplus-tanh 的小变体。它没有说明为什么该函数适合 ResNet-20 + CIFAR-10，也没有给出稳定性、初始化、参数约束或与 BatchNorm / residual path 的交互分析。
</redoc-highlight>

---

## Case 2: Regularization

### Motivation

dl_regularization 用来检查 master 是否能提出“训练 loss 附加项”级别的具体机制。这个 task 的风险是 regularization 很容易压过 cross entropy，导致训练失败。

### Method

<redoc-highlight emoji="tuding" fillColor="orange">
**Master proposal**：组合 adaptive orthogonal regularization 和 confidence penalty。直觉是让权重矩阵更正交以改善梯度流，同时惩罚过度自信预测以减轻 overfitting。
</redoc-highlight>

Worker 最终可信 retry 实现：

```python
orth_reg = torch.tensor(0.0, device=device)

for name, param in model.named_parameters():
    if 'weight' in name and param.dim() >= 2:
        w = param.view(param.size(0), -1)
        wt = torch.mm(w, w.t())
        identity = torch.eye(wt.size(0), device=device)
        orth_reg = orth_reg + torch.norm(wt - identity)

prob = F.softmax(outputs, dim=1)
entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
conf_penalty = -entropy.mean()

reg_loss = 1e-4 * orth_reg + 0.1 * conf_penalty
return reg_loss
```

### Result

| Item | Value |
| --- | --- |
| Benchmark baseline | 72.99 |
| Pass line | 73.00 |
| SFT proposal result | 71.53 |
| Passed | false |

过程里有一个重要细节：早期实现尝试出现过 shape mismatch，retry 后改成 W @ W.T，并加入小系数，训练才跑完。

```plaintext
early attempt: RuntimeError shape mismatch in orthogonal term
retry: epoch=180 test_acc=70.73
retry: epoch=190 test_acc=71.51
retry: TEST_METRICS: test_acc=71.53
```

### Takeaway

<redoc-highlight emoji="dui" fillColor="green">
**Worker 实现了主体机制**：orthogonal regularization 和 confidence penalty 都落到了代码里，并且 worker 主动加了小系数，避免正则项完全吞掉 CE loss。
</redoc-highlight>

<redoc-highlight emoji="cuo" fillColor="red">
**忠实度不是满分**：master 明确说 adaptive / epoch-dependent，但最终代码没有使用 epoch schedule。更重要的是，proposal 没有给出正则强度的量级、warmup、annealing 或 fail-safe，这在 regularization task 上非常关键。
</redoc-highlight>

---

## Case 3: Weight Initialization

### Motivation

dl_weight_initialization 适合检查 master 是否能提出“训练前一次性设置”的机制。相比 regularization，这类 task 更容易忠实实现，但也更容易把多个 known trick 生硬拼接。

### Method

<redoc-highlight emoji="tuding" fillColor="orange">
**Master proposal**：联合 He/Kaiming initialization、orthogonal initialization 和 Fixup-like depth scaling，并针对 residual branch、BatchNorm、Linear head 分别处理。
</redoc-highlight>

Worker 实现片段：

```python
num_blocks = sum(1 for m in model.modules() if isinstance(m, BasicBlock))
fixup_scale = max(num_blocks, 1) ** (-0.5)

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        if id(m) in residual_first_convs:
            nn.init.orthogonal_(m.weight)
        elif id(m) in residual_last_convs:
            nn.init.orthogonal_(m.weight, gain=fixup_scale)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        std = 1.0 / math.sqrt(depth)
        nn.init.normal_(m.weight, 0, std)

if 'resnet' in arch:
    for m in model.modules():
        if isinstance(m, BasicBlock):
            nn.init.constant_(m.bn2.weight, 0)
            nn.init.constant_(m.bn2.bias, 0)
```

### Result

| Item | Value |
| --- | --- |
| Benchmark baseline | 73.25 |
| Pass line | 73.26 |
| SFT proposal result | 72.62 |
| Passed | false |

训练过程稳定，但没有超过 baseline。

```plaintext
epoch=180 test_acc=72.39
epoch=190 test_acc=72.31
epoch=200 test_acc=72.54
TEST_METRICS: test_acc=72.62
```

### Takeaway

<redoc-highlight emoji="dui" fillColor="green">
**Worker 忠实度高**：He、orthogonal、Fixup-like scaling、BatchNorm / Linear handling 都被明确实现了。
</redoc-highlight>

<redoc-highlight emoji="cuo" fillColor="red">
**机制组合不够谨慎**：Fixup 是为无 normalization 或特定 residual scaling 场景设计的，把 zero residual branch 思路直接叠到 BatchNorm ResNet 上，不一定合理。proposal 缺少为什么这个组合会优于 standard Kaiming 的分析。
</redoc-highlight>

---

## Cross-case 分析

### 1. Worker 忠实度不是主要瓶颈

| Task | Proposal 是否被实现 | 主要偏差 |
| --- | --- | --- |
| dl_activation_function | 是 | 几乎无偏差 |
| dl_regularization | 部分是 | 缺少 adaptive schedule |
| dl_weight_initialization | 是 | 几乎无偏差 |

<redoc-highlight emoji="dui" fillColor="green">
**进展**：worker interface 可以支持 proposal-grounded implementation。至少在这三个 case 里，worker 不是主要噪声源。
</redoc-highlight>

### 2. Master proposal 的粒度仍然不够

| 需要的 proposal 粒度 | 当前表现 |
| --- | --- |
| 要说明具体公式 / module / loss term | 基本做到 |
| 要说明初始化、约束、scale、schedule | 经常缺失 |
| 要说明为什么适合当前 architecture + dataset | 经常泛化叙述 |
| 要预测 failure mode 并给出 sanity check | 基本缺失 |

<redoc-highlight emoji="cuo" fillColor="red">
**缺陷**：当前 proposal 经常停在“把几个相关术语组合起来”的层面。它们看起来像 research idea，但没有足够的 implementation-level guardrail。
</redoc-highlight>

### 3. 下一步训练目标应更强调 “可实现细节”

<redoc-highlight emoji="tuding" fillColor="orange">
**方法建议**：后续训练 target 不应只奖励 XML 完整和 research-like 叙述，还要显式奖励：baseline-equivalent initialization、bounded parameters、coefficient ranges、warmup / annealing plan、NaN guard、short sanity check、以及 task-specific rationale。
</redoc-highlight>

## 数据来源

主要 artifact：

```plaintext
autoresearch_idea_harness/runs/reports/v3_base_vs_v1sem_cot_sft_eval_partial.json

autoresearch_idea_harness/runs/v3_precomputed_worker_eval/
  dlc_mls10_v1sem_cot_sft_16k_guarded/
  dlc_mls10_v1sem_cot_sft_16k_guarded_reg_retry/
```

重点样例路径：

```plaintext
dl_activation_function__resnet20-cifar10__qwen25_32b_v1sem_cot_sft_16k__worker_eval
dl_regularization__resnet56-cifar100__qwen25_32b_v1sem_cot_sft_16k__worker_eval
dl_weight_initialization__resnet56-cifar100__qwen25_32b_v1sem_cot_sft_16k__worker_eval
```

## Final takeaway

<redoc-highlight emoji="dengpao" fillColor="cyan-blue">
这三个 demo 给出的判断是：V3 SFT checkpoint 已经能产出格式完整、领域相关、worker 可执行的 proposal；但它还没有学会提出足够稳健的 scientific mechanism。下一轮训练或 RL 需要把 reward 从“像 proposal”推进到“能指导 worker 写出稳定、合理、可超过 baseline 的实现”。
</redoc-highlight>
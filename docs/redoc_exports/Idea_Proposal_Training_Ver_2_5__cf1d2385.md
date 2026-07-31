> 本报告记录 Idea Proposal Training Ver 3 的四个核心工作包：高质量 arXiv/TeX cache、V1-semantics prompt rebuild、32B base vs V1-like CoT SFT 实验、以及 harness 可靠性修复。颜色约定：<font color="#F97316">橙色代表方法重点</font>，<font color="#16A34A">绿色代表进展</font>，<font color="#DC2626">红色代表 negative result 或缺陷</font>。

<redoc-highlight emoji="tuding" fillColor="yellow">
**一句话结论**：Ver 3 把训练数据和评测链路整理到可复现状态，cache 和 prompt 质量显著提升；但 32B 上的 V1-like CoT SFT 没有带来 pass-rate gain，说明当前轻量 SFT 不足以训练出更强的 idea proposal model。
</redoc-highlight>

## 0. 总览

| 模块 | Motivation | Method 核心 | Result | Takeaway |
| --- | --- | --- | --- | --- |
| 1. Strict929 Cache | abstract 太抽象，训练 target 缺少可执行细节 | <font color="#F97316">为 target paper 和 refs 建 compact abstract、research question、TeX detail snippets、六种 prompt cache</font> | <font color="#16A34A">929/929 pass，25383 refs，problem_count 0</font> | 数据底座可用于正式训练，但需要继续监控 ref universe 与 V1 差异 |
| 1. V1-semantics Rebuild | V3 cache 要能和 V1 训练/评测逻辑对齐 | <font color="#F97316">复刻 V1 conditioning：full_refs seed shuffle、top_k index、related-work narrative、with_research_question conditioned on full refs</font> | <font color="#16A34A">929/929 hard checks pass</font> | 可以公平比较 V1-style training，同时保留 V3 长问题和 XML schema |
| 1. 32B Base vs CoT SFT | 需要判断 SFT 是否真的比 base 更会 propose idea | <font color="#F97316">用 928 条 V1-semantics CoT target 训练 Qwen2.5-32B，并在 10 个 MLS task 上 worker eval</font> | <font color="#DC2626">base 1/10 pass，SFT 0/10 pass</font> | 轻量 SFT 当前不是 promising win；后续应转向更强数据、更强模型或 RL/PRS |
| 1. Harness Reliability | 评测链路本身如果不可靠，会污染结论 | <font color="#F97316">修 DLC 双执行、worker premature termination、report merge 逻辑</font> | <font color="#16A34A">最终两组 eval 都 10/10 complete，0 error</font> | 现在的 negative result 比之前更可信 |

### Pipeline 视图

| 阶段 | 输入 | 处理 | 输出 | 质量门 |
| --- | --- | --- | --- | --- |
| A. Paper cache | arXiv metadata + TeX + refs | <font color="#F97316">抽取 target/ref detail，生成 compact abstract 与 research question</font> | strict929 article cache | <font color="#16A34A">929 articles, 25383 refs, 0 problems</font> |
| B. Prompt rebuild | strict929 cache + V1 prompt semantics | <font color="#F97316">重建六类 conditioning prompt</font> | V1-semantics prompt cache | <font color="#16A34A">929/929 verifier pass</font> |
| C. CoT target synthesis | V1-semantics prompts | <font color="#F97316">生成包含 thinking 与 proposal XML 的 target</font> | 928 SFT rows | <font color="#DC2626">1 篇因 safety refusal excluded</font> |
| D. SFT | 928 CoT rows | <font color="#F97316">32B LoRA/SFT，max_seq_length 16k</font> | final checkpoint | <font color="#16A34A">PAI job succeeded in about 49 min</font> |
| E. Evaluation | base/SFT proposals + MLS tasks | <font color="#F97316">worker implements proposal, benchmark returns signed metric</font> | comparison report | <font color="#DC2626">SFT pass rate did not improve</font> |

那 1 篇是 2504.07137。它不是因为没有 TeX，也不是训练脚本坏了，而是在用 Opus 合成 V3 CoT target 的时候，API 对这篇文章的 malware / cybersecurity 相关内容触发了safety refusal，没有返回可用的 &lt;thinking&gt; + &lt;proposal&gt; 训练目标。

所以我们当时的处理是：严格数据集不做强行补齐，直接 exclude。结果就是 strict929 文章缓存里有 929 篇，但可用于 V1-semantics CoT SFT 的 schema-valid target 是 928 篇。

## 1. Strict929 Cache：把 abstract-only 训练升级成 detail-grounded 训练

### Motivation

V1 的核心问题是训练目标过度依赖 abstract。Abstract 往往只描述 high-level claim，不说明 algorithm、implementation、training recipe、evaluation protocol。这样的 target 会把模型训练成“会写漂亮摘要”，但不会稳定提出能让 worker 实现并超过 baseline 的 idea。

因此 Ver 3 的第一步不是马上训练，而是先把 arXiv TeX 数据变成一个 detail-grounded paper bank，让每篇高质量样本都能提供：target paper 细节、reference paper 细节、research question、以及五类以上可组合的 conditioning prompt。

### Method

<redoc-highlight emoji="tuding" fillColor="orange">
**方法重点**：对每篇文章构建 article-level cache，而不是临时在训练时拼字符串。cache 中保存 target compact abstract、ref compact abstract、research question、TeX snippet、top-ref selection 和六种 prompt cache。这样后续 SFT、RL、dashboard inspection 可以复用同一个数据事实源。
</redoc-highlight>

| Cache 组件 | 内容 | 设计原因 |
| --- | --- | --- |
| abstract prompt | 原始 abstract 或 compact abstract | 保留 V1 最弱条件，作为 ablation anchor |
| full_refs prompt | seed-42 shuffled refs | 模拟“看过较多相关工作”的 setting |
| top_k_refs prompt | top-5 ref index + compact refs | 限制 context，同时保留最相关 ref |
| related_work prompt | V1-style narrative + References title list | 让模型看到领域结构而不只是 isolated refs |
| top_k_related_work prompt | top-k refs + related-work narrative | 精简但语义更强的 research context |
| with_research_question prompt | full refs + long research question | 让模型知道要解决什么 open problem |
| TeX snippets | method, implementation, training/data recipe, evaluation, result | 避免 target 缺少可执行细节 |

### Result

| Metric | Value | Status |
| --- | --- | --- |
| Article caches | 929 | <font color="#16A34A">complete</font> |
| Prompt caches per strategy | abstract/full_refs/top_k_refs/related_work/top_k_related_work/with_research_question = 929 each | <font color="#16A34A">complete</font> |
| Ref rows | 25383 | <font color="#16A34A">complete</font> |
| Top-k index rows | 929 | <font color="#16A34A">complete</font> |
| Research question length | min 156, avg 171.2, max 187 words | <font color="#16A34A">not truncated</font> |
| Target compact abstract length | min 116, avg 179.9, max 198 words | <font color="#16A34A">bounded</font> |
| Ref compact abstract length | min 45, avg 179.4, max 200 words | <font color="#16A34A">bounded</font> |
| Audit problems | 0 | <font color="#16A34A">clean</font> |

### Takeaway

<redoc-highlight emoji="dui" fillColor="green">
Ver 3 的数据底座已经从“abstract-only”升级到“target/ref/detail grounded”。这使后续训练可以清楚地区分：模型到底是在复述 abstract，还是在利用 ref-side evidence 和 TeX details 形成可执行 proposal。
</redoc-highlight>

## 2. V1-semantics Prompt Rebuild：保留公平可比性，同时修正明显不利于训练的点

### Motivation

Ver 3 cache 比 V1 丰富，但如果 prompt 语义和 V1 完全不一致，就很难判断后续训练结果来自模型能力提升，还是来自 conditioning distribution 改了。因此我们把 V1 prompt behavior 设为 alignment baseline：凡是决定实验语义的地方对齐 V1；明显不利于训练 idea proposal model 的地方，则记录为 deliberate rationalization。

### Method

<redoc-highlight emoji="tuding" fillColor="orange">
**方法重点**：V1 定义 conditioning semantics，V3 提供更好的 cache substrate。也就是说，prompt 的 reference ordering、top-k index 和 related-work narrative 尽量复刻 V1；但 target schema、长 research question、compact ref summaries 使用 V3 版本。
</redoc-highlight>

| Strategy | V1 对齐点 | V3 保留/改进点 |
| --- | --- | --- |
| full_refs | seed-42 shuffled full-ref set | 使用 V3 compact ref text，减少截断与空 abstract |
| top_k_refs | V1 top_k_5_index | duplicate top-k index 做 dedupe，记录 allowed deviation |
| related_work | V1 cached narrative block + References title list | article cache 可追踪来源 |
| top_k_related_work | V1 top-related narrative | top-k ref detail 可独立 inspect |
| with_research_question | conditioned on full_refs block | 使用 V3 long research question，而不是 V1 过短问题 |
| target schema | 保留 proposal training objective | 使用 XML schema，包含 thinking 与 structured proposal |

### Result

| Verifier item | Value | Status |
| --- | --- | --- |
| Articles checked | 929 | <font color="#16A34A">pass</font> |
| with_research_question equals full-ref conditioning | 929 | <font color="#16A34A">pass</font> |
| related_work refs present | 929 | <font color="#16A34A">pass</font> |
| top_related refs present | 929 | <font color="#16A34A">pass</font> |
| long research questions | 929 | <font color="#16A34A">pass</font> |
| V1 related narrative | 929 | <font color="#16A34A">pass</font> |
| V1 top-related narrative | 929 | <font color="#16A34A">pass</font> |
| Hard problems | 0 | <font color="#16A34A">clean</font> |

| Known caveat | Count | Interpretation |
| --- | --- | --- |
| Duplicate V1 top-k indices deduped by V3 | 7 articles | <font color="#F97316">Allowed deviation: avoids duplicated context</font> |
| V3 refs differ from V1 train refs | 36 articles | <font color="#DC2626">Inherited ref-universe mismatch; document when comparing strictly to V1</font> |
| top_k_related_work V1 title-list differences | 56 articles | <font color="#DC2626">Related-work title list can reflect V1 cache universe, not always current article refs</font> |

### Takeaway

最准确的描述是：<font color="#16A34A">V1 conditioning cache plus V3 compact refs/schema/long-question supplements</font>。这让后续训练既能和 V1 公平对齐，又不会把明显不完整的 V1 artifact 强行复制进训练数据。

## 3. 32B Base vs V1-like CoT SFT：一次可信但不正向的训练结果

### Motivation

我们需要回答一个直接问题：如果用 V1-semantics cache 合成 CoT target，再对 Qwen2.5-32B 做 SFT，是否能比没有 finetune 的 32B base 更会提出可执行 idea？之前的 SFT 数据没有完全生成好，也不一定包含 CoT，因此这次重新做一版更严格的训练和评测。

### Method

<redoc-highlight emoji="tuding" fillColor="orange">
**方法重点**：训练输入来自 V1-semantics strict929 prompt cache；target 优先使用 cot_impl_proposal，要求包含 &lt;thinking&gt; 和结构化 proposal XML；SFT 后在 10 个 MLS task 上生成 proposal，再由 worker 实现并跑 benchmark。
</redoc-highlight>

| Step | Configuration | Result |
| --- | --- | --- |
| Target synthesis | V1-semantics strict929 cache | <font color="#16A34A">928 schema-valid CoT rows</font> |
| Exclusion | 2504.07137 | <font color="#DC2626">Opus safety refusal on malware/cybersecurity survey synthesis</font> |
| SFT dataset | all 928 rows in train split | <font color="#16A34A">all rows include thinking + complete proposal XML</font> |
| Model | Qwen2.5-32B, 16k sequence length | <font color="#16A34A">DLC job succeeded in about 49 minutes</font> |
| Evaluation | 10 MLS tasks, proposal + worker implementation | <font color="#16A34A">base and SFT both completed 10/10 with 0 final errors</font> |

### Result：总体

| Model arm | Planned | Completed | Pass | Error | Pass rate | Main reading |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B base | 10 | 10 | <font color="#16A34A">1</font> | 0 | 10.0% | base has one real pass |
| V1-semantics CoT SFT | 10 | 10 | <font color="#DC2626">0</font> | 0 | 0.0% | SFT did not improve pass rate |

### Result：task-level matrix

| Task | Pass line | Base metric | Base pass | SFT metric | SFT pass | SFT minus base |
| --- | --- | --- | --- | --- | --- | --- |
| cv_classification_loss | 73.48 | 72.72 | <font color="#DC2626">fail</font> | 72.14 | <font color="#DC2626">fail</font> | <font color="#DC2626">-0.58</font> |
| cv_data_augmentation | 93.76 | 92.13 | <font color="#DC2626">fail</font> | 93.46 | <font color="#DC2626">fail</font> | <font color="#16A34A">+1.33</font> |
| cv_multitask_loss | 68.79 | 67.87 | <font color="#DC2626">fail</font> | 68.50 | <font color="#DC2626">fail</font> | <font color="#16A34A">+0.63</font> |
| cv_pooling_aggregation | 72.19 | 72.63 | <font color="#16A34A">pass</font> | <font color="#CC3030">**1.00**</font> | <font color="#DC2626">fail</font> | <font color="#DC2626">-71.63</font> |
| cv_sample_weighting | 74.44 | 72.15 | <font color="#DC2626">fail</font> | 72.87 | <font color="#DC2626">fail</font> | <font color="#16A34A">+0.72</font> |
| dl_activation_function | 93.35 | 92.79 | <font color="#DC2626">fail</font> | 92.67 | <font color="#DC2626">fail</font> | <font color="#DC2626">-0.12</font> |
| dl_lr_schedule | 93.19 | 90.82 | <font color="#DC2626">fail</font> | 90.96 | <font color="#DC2626">fail</font> | <font color="#16A34A">+0.14</font> |
| dl_regularization | 73.00 | 72.29 | <font color="#DC2626">fail</font> | 71.53 | <font color="#DC2626">fail</font> | <font color="#DC2626">-0.76</font> |
| dl_residual_connection | 92.96 | 92.32 | <font color="#DC2626">fail</font> | 92.96 | <font color="#DC2626">fail</font> | <font color="#16A34A">+0.64</font> |
| dl_weight_initialization | 73.26 | 72.91 | <font color="#DC2626">fail</font> | 72.62 | <font color="#DC2626">fail</font> | <font color="#DC2626">-0.29</font> |

#### cv_pooling_aggregation 的 failure mode

model proposal:

```plaintext
<proposal>
<title>Adaptive Spatial-Aware Pooling for Improved Image Classification</title>
<problem>Current global pooling methods like GAP and GMP either discard spatial information (GAP) or focus solely on the strongest activations (GMP). This limits their ability to capture nuanced spatial statistics in feature maps.</problem>
<gap>The need exists for a pooling mechanism that can dynamically adapt to the spatial characteristics of feature maps, potentially leading to more informative representations for classification tasks.</gap>
<core_idea>Implement an adaptive spatial-aware pooling mechanism that combines learnable spatial weighting with a mixture of average and maximum pooling strategies. This approach aims to balance capturing the overall distribution of activations and the most salient features, while adapting to the unique spatial characteristics of each feature map.</core_idea>
<implementation_plan>Modify the `CustomPool` class to incorporate a spatially weighted averaging mechanism that learns to emphasize different regions of the feature map. Additionally, integrate an element-wise combination of average and maximum pooling to capture both mean-field and peak statistics.</implementation_plan>
<algorithm_or_system>
```python
class CustomPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_weights = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.avg_max_weights = nn.Parameter(torch.ones(2))

    def forward(self, x):
        spatial_weights_expanded = self.spatial_weights.expand_as(x)
        weighted_x = x * spatial_weights_expanded

        avg_pool = F.adaptive_avg_pool2d(weighted_x, 1).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool2d(weighted_x, 1).view(x.size(0), -1)

        avg_max_combined = self.avg_max_weights[0] * avg_pool + self.avg_max_weights[1] * max_pool
        return avg_max_combined
```
</algorithm_or_system>
<training_or_data_recipe>Train the network with the same setup as specified in the task description: SGD optimizer with learning rate of 0.1, momentum of 0.9, weight decay of 5e-4, cosine annealing schedule, and 200 epochs. Data augmentation includes RandomCrop(32, pad=4) and RandomHorizontalFlip.</training_or_data_recipe>
<evaluation_plan>Evaluate the performance of the modified pooling layer on the ResNet-56 model trained on CIFAR-100. Measure the test accuracy and compare it against the pass threshold of 72.19000000000001.</evaluation_plan>
<expected_results>Expect the adaptive spatial-aware pooling to improve the test accuracy due to its ability to capture more nuanced spatial information and adapt to varying spatial distributions within the feature maps. Anticipate a significant improvement over the baseline GAP approach, surpassing the pass metric.</expected_results>
<risks_and_limitations>There is a risk that the added complexity of learnable spatial weights and the combined pooling strategy might not provide a substantial benefit over simpler methods. Additionally, the learnable parameters may lead to overfitting on the training set without generalizing well to the test set. The effectiveness of this approach also depends heavily on the architecture and dataset, and may not generalize well across different settings.</risks_and_limitations>
</proposal>
```

关键现象是 eval 从 epoch 1 开始：

```plaintext
  train_loss=nan train_acc=1.00 test_loss=nan test_acc=1.00
  ...
  TEST_METRICS: test_acc=1.00
```

  worker 根据 SFT proposal 写了一个不稳定的 pooling：

```plaintext
  self.spatial_weights = nn.Parameter(torch.ones(1, 1, 1, 1))
  self.avg_max_weights = nn.Parameter(torch.ones(2))
  ...
  return self.avg_max_weights[0] * avg_pool + self.avg_max_weights[1] * max_pool
```

问题在于它不是稳定的 convex mixture，也没有 normalization / sigmoid / softmax 约束；初始就是 avg + max，会显著改变 feature scale。再加上spatial_weights 是无约束可学习标量，LR 0.1 下很容易把 ResNet-56/CIFAR-100 的训练推到 NaN。最后evaluator 仍然写出了 val_metric=1.0，所以表里显示成 1/72.19/fail。

<font color="#CC3030" backgroundColor="rgba(255, 0, 0, 0.18)">**这是一个 worker implementation failure**</font>，不是 SFT 模型能力提升或真实差异。需要加两个 guard：

  1. eval log 出现 nan/inf loss 时，直接标记为 invalid/error，不当成正常结果。

  2. worker prompt 对 pooling 类 task 明确要求 stable initialization：从 GAP 等价初始化、convex gate、bounded weights，并先跑 short sanity check，NaN 必须修复。

### Result：SFT delta 分布

| Direction | Tasks | Count | Reading |
| --- | --- | --- | --- |
| <font color="#16A34A">SFT metric higher than base</font> | cv_data_augmentation, cv_multitask_loss, cv_sample_weighting, dl_lr_schedule, dl_residual_connection | 5 | SFT learned something useful on several tasks, but not enough to cross pass lines |
| <font color="#DC2626">SFT metric lower than base</font> | cv_classification_loss, cv_pooling_aggregation, dl_activation_function, dl_regularization, dl_weight_initialization | 5 | Regressions include a severe pooling failure at 1.00 |

### Takeaway

<redoc-highlight emoji="cuo" fillColor="red">
**主要 negative result**：这版 V1-like CoT SFT 没有提升 pass rate。Base 是 1/10 pass，SFT 是 0/10 pass。虽然 SFT 在 data augmentation、sample weighting、multitask loss、lr schedule、residual connection 上提升 raw metric，但提升幅度不足，且在 pooling aggregation 上出现严重退化。
</redoc-highlight>

这说明当前 SFT 目标可能只教会模型输出更像 training target 的 proposal，而<font color="#CC3030" backgroundColor="rgba(255, 0, 0, 0.18)">**没有稳定提升“提出能被 worker 实现并超过 strong baseline 的 idea”的能力**</font>。下一阶段更适合做 PRS/RL、提高 proposal quality filtering、使用更强 master model，或把 proposal generation 与 worker dialogue 共同纳入训练目标。

## 4. Harness Reliability：让 negative result 可信

### Motivation

在自动科研 harness 中，评测链路本身很容易制造假结论。例如同一个 DLC job 在 aimaster 和 master 上重复执行，会导致 sample 重复启动；worker 写出 error result.json 后如果 runner 过早杀掉 Claude，会阻止 worker 自修复；report 如果优先读取旧 error row，也会掩盖 retry 的有效 metric。

### Method

<redoc-highlight emoji="tuding" fillColor="orange">
**方法重点**：把 harness 当成实验系统的一部分来 debug。不是只看模型分数，而是修复 sample lifecycle、DLC pod behavior、worker termination、report merge priority，确保每一个 result 都能追溯到唯一 proposal、唯一 worker run、唯一 signed metric。
</redoc-highlight>

| Reliability issue | Symptom | Fix | Impact |
| --- | --- | --- | --- |
| DLC aimaster/master double execution | 一个 sample 被启动两次，日志和 workspace 被污染 | <font color="#F97316">加入 aimaster skip guard</font> | <font color="#16A34A">guarded eval 每个 sample 只启动一次</font> |
| Worker premature termination | error result.json 出现后 Claude 被杀，无法自修复 | <font color="#F97316">只在 metric-bearing result 出现时提前终止</font> | <font color="#16A34A">regularization retry 能继续训练并产出 metric</font> |
| Report stale error priority | 早期失败 row 可能覆盖后续 retry metric | <font color="#F97316">report merge 优先使用带 metric 的 retry row</font> | <font color="#16A34A">最终统计 10/10 completed, 0 final errors</font> |
| Base worker blocking | worker 已写 result 但还在等后台 benchmark，GPU batch 被卡住 | <font color="#F97316">runner 发现 fresh result 后终止 worker process</font> | <font color="#16A34A">减少无效等待，补跑 pending tasks</font> |

### Result

| Artifact | Meaning | Status |
| --- | --- | --- |
| scripts/prepare_v3_precomputed_worker_eval_run.py | worker eval prep and batching | <font color="#16A34A">records correct module/model ids and exits nonzero on worker failure</font> |
| watch_v3_sft_and_launch_eval.py | post-SFT eval launch | <font color="#16A34A">idempotent path to proposal generation and worker eval</font> |
| guarded DLC command skeletons | avoid duplicate pod execution | <font color="#16A34A">fixed after detecting contamination</font> |
| render_v3_eval_comparison.py | final base-vs-SFT report | <font color="#16A34A">merges signed metrics and retry roots correctly</font> |

### Takeaway

Ver 3 的实验结论虽然是负面的，但它比早期结果更可信：最终不是因为 worker crash、DLC duplicate execution 或 report merge bug 而失败，而是在修复这些链路后仍然没有 pass-rate improvement。

## 5. 最终判断与下一步

| Question | Current answer | Evidence |
| --- | --- | --- |
| 数据是否足够干净，可以继续训练？ | <font color="#16A34A">基本可以</font> | strict929 cache problem_count 0，V1-semantics verifier pass |
| V1-like CoT SFT 是否 promising？ | <font color="#DC2626">当前不 promising</font> | SFT 0/10 pass，base 1/10 pass |
| 负结果是否可信？ | <font color="#16A34A">可信度比之前高</font> | runner/report/DLC guard 已修复，最终 10/10 complete |
| 下一步应该直接继续 SFT 吗？ | <font color="#DC2626">不建议只加同类轻量 SFT</font> | raw metric 有些改善但 pass line 没过，且有严重 regression |
| 更合理的方向 | <font color="#F97316">PRS/RL、proposal filtering、更强 master、训练 master-worker dialogue</font> | pass/fail 稀疏，单纯 imitation 不够 |

<redoc-highlight emoji="dengpao" fillColor="cyan-blue">
**建议路线**：把 Ver 3 的 strict929/V1-semantics cache 保留为数据底座；不要把这次 SFT 当成成功模型，而应把它当作可靠 baseline。下一阶段应把 reward 与 outcome 结合，优先探索 PRS/RL 或更强 master 的 proposal selection/self-improvement。
</redoc-highlight>

## 6. Artifact 索引

| Artifact | Path | 用途 |
| --- | --- | --- |
| strict929 merged cache | autoresearch_idea_harness/runs/v3_quality_cache/strict929_v4_sharded/merged | article cache, prompt cache, refs, TeX snippets |
| strict completion audit | .../merged/completion_audit_strict.json | cache completeness proof |
| V1-semantics validation | .../merged/prompt_rebuild_v1_semantics_validation.json | prompt rebuild proof |
| SFT train data | autoresearch_idea_harness/runs/training_data/v3_0_sft_qwen25_32b_v1sem_cot_strict929_schema_all_train_all/train.jsonl | 928 CoT rows |
| Eval report JSON | autoresearch_idea_harness/runs/reports/v3_base_vs_v1sem_cot_sft_eval_partial.json | machine-readable task matrix |
| Eval report Markdown | autoresearch_idea_harness/runs/reports/v3_base_vs_v1sem_cot_sft_eval_partial.md | human-readable local report |
| Main memory log | agent-memory/checkins.md | chronological decisions and status |
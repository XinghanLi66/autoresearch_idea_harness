<redoc-highlight>
V2.2 的重点不再是单独训练一个 proposal model，而是先把一个可检查、可复现、可结算的 autoresearch harness 搭起来：输入 MLS/MLE task，master model 生成 proposal，expert models 做 prediction market，gate 通过后由 worker 实现并跑 benchmark，最后把所有中间过程、LLM calls、worker logs、eval result 和 market settlement 都落盘，并可通过 dashboard 检查。

今天已经完成了这个 end-to-end MVP，并在 dl_activation_function / resnet20-cifar10 上跑了一次真实 worker smoke。结果没有超过 baseline，但 pipeline 本身完整闭环。
</redoc-highlight>

## 为什么要做 V2.2

V2.1 主要解决 training target 的问题：abstract 太抽象，模型即使学会对齐 abstract，也很容易输出缺少 implementation detail 的 idea。因此我们收集 full TeX source，合成 TeX-grounded target，并进一步意识到 reference papers 也需要 detail，否则 condition 里没有足够 evidence，却要求模型输出细节是不合理的。

V2.2 的判断是：如果最终目标是 autoresearch，idea proposal 不应只是一个离线 SFT/RL 模型，而应该处在一个 harness 里。这个 harness 要同时管理 evidence、proposal、market scoring、implementation、evaluation 和 feedback。训练可以之后发生，但第一步应该是把真实科研循环的接口和可观测性搭出来。

## 今天完成的系统

新增 standalone 项目：autoresearch_idea_harness/。

核心能力包括：

- self-contained task/benchmark interface：从 proposal_rl 复制小型必要接口和 assets，不依赖远端代码跳转；大 cache 和大 checkout 仍通过 path config 外挂。

- task packet：把 task、subtask、baseline、pass line、metric direction、worker constraints、eval command 和 evidence context 固化为 task_packet.json。

- master proposal generation：默认 master 使用 Opus 4.7，经 Google Anthropic rawPredict 路径调用。

- prediction market：专家模型目前为 Opus 4.7 和 GPT 5.5，分别给 proposal 成功概率和 rationale。

- gate and settlement：默认 equal-weight mean probability &gt;= 0.5 才通过；worker 结果出来后用 Brier score 和 log score 结算 expert forecast。

- worker implementation loop：worker 实现 proposal；如果第一次失败或遇到歧义，可向 master 请求 advice，advice 作为辅助信息重新尝试。

- detailed logging：所有事件、LLM prompt/response、proposal、market、worker prompt、worker log、dialogue、eval log、result 和 report 都落盘。

- read-only dashboard：scripts/dashboard_tui.py 可以浏览 Runs、Overview、Task Packet、Proposals、Market、Gate、Worker Prompt、Worker Log、Dialogue、Eval Log、Result 和 Report。

## End-to-end pipeline

当前 V2.2 pipeline 是：

1. 输入 task：例如 dl_activation_function / resnet20-cifar10。

2. 构建 task packet：记录 baseline 92.97、pass metric 93.27、editable constraints 和 eval command。

3. Master 生成 proposal：Opus 4.7 读 task packet，输出可实现的 research idea。

4. 匿名化 proposal：expert 只看到 proposal label，不看到 generator identity。

5. Expert prediction market：Opus 4.7 和 GPT 5.5 各自预测 proposal 成功概率，并写 rationale、strengths、weaknesses、baseline risk。

6. Gate：若 mean probability 达到阈值则运行 worker；smoke 模式下如果没有 proposal 通过，可 forced run top-1。

7. Worker implementation：Claude Code worker 在 sandbox workspace 中编辑目标文件并运行 benchmark。

8. Master-worker dialogue：如果初次结果失败，worker 可以请求 master advice，再尝试一次。

9. Benchmark eval：写出 signed result.json，记录 eval.log 和 worker.log。

10. Settlement：根据 pass/fail 结算 expert Brier/log score。

11. Final report：生成 summary.json 和 report.md，dashboard 自动可读。

<br/>

<redoc-text-draw remoteTemplate="flowchart TD
    A[Input: task + subtask] --> B[Build task_packet.json]
    B --> C[Master: Opus 4.7 generates N proposals]
    C --> D[Anonymize proposals for experts]
    D --> E[Experts: Opus 4.7 + GPT 5.5 forecast success probability]
    E --> F[Market gate: mean p >= 0.5]
    F -->|pass| G[Worker implements selected proposal]
    F -->|no pass, smoke only| G2[Forced top-1 worker run]
    G2 --> G
    G --> H[Benchmark eval writes signed result.json]
    H --> I{Passed threshold?}
    I -->|fail + advice budget| J[Master gives implementation advice]
    J --> G
    I --> K[Market settlement: Brier/log score]
    K --> L[summary.json + report.md + dashboard]" remoteView="chart-only"/>

## Artifacts 设计

每个 run 都放在：

```plaintext
runs/end_to_end/<task>_<subtask>_<run_id>/
```

关键文件：

```plaintext
events.jsonl
任务和状态事件流，append-only。

task_packet.json
完整 task/evidence/baseline/eval context。

llm_calls/<call_id>/request.json
llm_calls/<call_id>/response.json
所有 LLM 调用的 sanitized prompt、response、usage、latency 和错误信息。

proposals/proposal_private.jsonl
带 generator identity 的内部 proposal 记录。

proposals/proposal_expert.jsonl
匿名 expert-facing proposal。

market/forecasts.jsonl
专家概率、rationale 和 usage。

market/gate_summary.json
gate 决策、选中 proposal、是否 forced smoke。

worker_runs/<proposal_id>/worker_prompt.txt
worker_runs/<proposal_id>/worker.log
worker_runs/<proposal_id>/master_worker_dialogue.jsonl
worker_runs/<proposal_id>/eval.log
worker_runs/<proposal_id>/result.json
worker 实现和评测的完整证据链。

summary.json
report.md
最终结构化结果和人类可读报告。
```

所有日志都避免写 API key、auth header 和 .env 原始内容。

## 验证结果

今天做了三层验证。

### 1. Offline smoke

使用 mock master、mock experts 和 fixture worker，验证 pipeline 控制流、匿名 proposal、gate、fixture result、settlement 和 dashboard 都能跑通。Fixture result 设为 93.28，超过 pass metric，因此用于确认 pass path 没问题。

### 2. Live LLM skip-worker smoke

使用真实 Opus 4.7 生成 proposal，真实 Opus 4.7 和 GPT 5.5 做 forecast，但跳过 worker。这个测试确认两个模型 API path 可用，LLM call logs 正常落盘。

### 3. Live worker smoke

真实运行路径：

```plaintext
autoresearch_idea_harness/runs/end_to_end/live_worker_activation_20260603_144845
```

任务：dl_activation_function / resnet20-cifar10。

结果摘要：

| 项目 | 结果 |
| --- | --- |
| Run ID | 9ab4b322 |
| Baseline | 92.97 |
| Pass metric | 93.27 |
| Proposal | SoftSwish-style learnable activation |
| Opus 4.7 forecast | 0.36 |
| GPT 5.5 forecast | 0.38 |
| Mean probability | 0.37 |
| Gate | 未通过；smoke forced top-1 |
| Worker attempt 0 | learnable SoftSwish-style activation，92.11 |
| Master advice | 简化为 plain SiLU，移除 learnable parameters |
| Worker attempt 1 | plain SiLU，92.05 |
| Final improvement | -0.92 vs baseline |
| Final status | failed |

Market settlement：

| Expert | Probability | Outcome | Brier score | Log score |
| --- | --- | --- | --- | --- |
| Opus 4.7 | 0.36 | failed | 0.1296 | 0.446287 |
| GPT 5.5 | 0.38 | failed | 0.1444 | 0.478036 |

这次结果说明专家判断是保守且方向正确的：两个专家都没有让 proposal 过 gate；最终 forced smoke 的 worker 结果也确实失败。

## 关键观察

第一，pipeline 的闭环已经成立。proposal、market、worker、master advice、eval、settlement 和 report 都有真实 artifact，而不是只停留在 prompt demo。

第二，prediction market 在这次 smoke 中起到了过滤作用。两个专家都认为成功概率低于 0.5，最终 benchmark 也验证了这个低置信判断。

第三，worker/master dialogue 的机制可用。第一次实现失败后，系统触发 master advice，并完成第二次尝试。虽然第二次仍失败，但这个路径证明了 worker 不是一次性黑盒，而可以纳入带日志的 iterative loop。

第四，activation function 任务比看起来更难。SiLU/Swish-like activation 通常只是在 GELU 附近波动，要稳定超过 93.27 需要更强机制，而不是简单的 smooth activation variant。

第五，V2.2 的主要价值不是今天这个 proposal 成功，而是提供了一个能明确回答“哪里失败”的实验系统：proposal 本身风险偏高、market 没放行、forced smoke 后 worker 也没过 benchmark。

## 当前限制

- 这次 live worker 只跑了 1 个 proposal，不能代表 proposal distribution。

- Expert market 目前只有两个模型，且 settlement 只是记录 Brier/log score，还没有长期 reliability weighting。

- Worker 的实现能力仍可能被 task sandbox、训练时间、seed variance 和 benchmark budget 限制。

- Master advice 目前是 advisory text，还没有结构化成可控 patch plan。

- Dashboard 是 read-only TUI，够检查 artifact，但还不是完整 web dashboard。

- Prediction market 目前没有真实人的经济激励，只是用 proper scoring rule 记录模型 forecast 质量；之后可以接入 human experts 或 prediction-market-style reward。

## 下一步建议

1. 对 activation task 跑 n_proposals &gt; 1，不要只看单 proposal；先让 market 选出真正过 0.5 的 proposal，再运行 worker。

2. 在 master prompt 中显式要求“为什么该方法可能超过 GELU baseline”，并惩罚只提出 Swish/GELU 邻近变体的 proposal。

3. 把 expert forecast 历史累计到 expert_reliability.jsonl，逐步从 equal-weight market 切到 reliability-weighted market。

4. 扩展到 MLE task，但先以 prompt/proposal/market preview 为主，不急着跑长 worker。

5. 继续强化 dashboard，让每一步的 prompt、response、gate reason、worker diff 和 eval tail 都能快速复制和审计。

6. 把 paper bank/ref evidence packet 接入 task packet，使 idea proposal 不只依赖 task description，而能依赖更丰富的 reference-side detail。

## 主要命令

运行 end-to-end：

```bash
cd autoresearch_idea_harness
python scripts/run_end_to_end.py --task dl_activation_function --subtask resnet20-cifar10 --n-proposals 3 --max-workers 1
```

查看 dashboard：

```bash
cd autoresearch_idea_harness
python scripts/dashboard_tui.py --runs-root runs/end_to_end
```

本次 live run 位置：

```plaintext
autoresearch_idea_harness/runs/end_to_end/live_worker_activation_20260603_144845
```

## V2.2 的定位

V2.2 是从“训练一个 proposal model”转向“搭建 autoresearch proposal harness”的第一版。它先把数据、模型、专家评估、worker 实现和 benchmark 评测放进同一个可观测 loop。后续训练应该服务于这个 loop，而不是绕开这个 loop。

## Workflow 细节补充

这一节把 V2.2 harness 的实际运行细节展开，重点说明 task_packet 如何构建、master proposal 的 setting 和 V1 的关系，以及 expert 在 prediction market 中到底能看到什么。

### Workflow 图

```plaintext
Input: task + subtask
        |
        v
Build task_packet.json
  - task adapter 读取 benchmark/task metadata
  - 固化 baseline、pass line、metric、task context、frontline papers、worker constraints
        |
        v
Master proposal generation
  - Opus 4.7
  - Google Anthropic rawPredict
  - temperature = null (API的调用平台拒绝temperature参数)
  - input = full task_packet
  - output = implementation-ready XML proposal
        |
        v
Anonymize proposals
  - private file 保留 generator/model/call_id
  - expert file 只保留 proposal_id、label、text
        |
        v
Prediction market expert scoring
  - Opus 4.7 expert
  - GPT 5.5 expert
  - each returns success_probability + rationale + strengths/weaknesses + baseline_risk
        |
        v
Gate
  - equal-weight mean probability
  - default gate threshold = 0.5
  - smoke mode: if none pass, force top-1 only for debugging pipeline
        |
        v
Worker implementation
  - worker receives selected proposal + task context + constraints
  - MLS: modify editable_region.py only
  - MLE: modify baseline.py and produce submission.csv
        |
        v
Benchmark eval
  - run bash run.sh result.json
  - result.json must contain signed val_metric
        |
        v
Optional master-worker dialogue
  - if ambiguity, failure, or under-threshold result
  - master gives one concise implementation/debugging advice
  - worker retries within advice budget
        |
        v
Settlement + report
  - compute Brier/log score for expert forecasts
  - write summary.json, report.md, dashboard-readable artifacts
```

### task_packet 是怎么构建的

代码入口是 autoresearch_idea_harness/src/autoresearch_idea_harness/benchmarking.py。

实际流程：

1. run_end_to_end.py 接收 task 和 subtask。

2. EndToEndRunner.run() 调用 get_task(cfg, task, subtask)。

3. get_task 从 registry 中选具体 task adapter，例如 DlActivationFunctionTask 或某个 MleBenchTask。

4. task adapter 调用 task.task_packet(cfg)，生成统一 JSON。

5. runner 补上 run_id 和 subtask，写入当前 run 目录下的 task_packet.json。

task_packet 的核心字段：

```json
{
  "task": "dl_activation_function",
  "task_type": "mls",
  "metric_name": "test_acc",
  "lower_is_better": false,
  "baseline_metric": 92.97,
  "pass_threshold": 0.30,
  "pass_metric": 93.27,
  "task_context": "...",
  "frontline_papers": [...],
  "worker_constraints": {...},
  "run_id": "...",
  "subtask": "resnet20-cifar10"
}
```

MLS task 的 task_context 来自 MLS-Bench task description，并附带：

- 当前 subtask，例如 resnet20-cifar10。

- reference baseline，例如 activation task 的 gelu = 92.97。

- pass line，例如 93.27。

- 可编辑区域代码，即 editable_region.py 的初始内容。

- 固定运行命令：bash run.sh result.json。

MLS task 的 worker_constraints 明确 worker 只能改 editable_region.py，并记录原始文件、起止行和 entrypoint。

MLE task 的 task_context 则来自 competition description、sample submission、baseline starting point，以及当前 baseline normalized score。MLE worker 的约束是修改 baseline.py，产生 submission.csv，再由 mle_grade.py 写 signed result.json。

frontline_papers 当前来自 harness assets：

- MLS: autoresearch_idea_harness/assets/mls_tasks/&lt;task&gt;/frontline_papers.txt

- MLE: autoresearch_idea_harness/assets/mle_tasks/&lt;task&gt;/frontline_papers.txt 和 frontline_metadata.json

注意：当前 end-to-end task packet 还没有把 V2 paper bank 里的 ref-side TeX snippets 完整接进来。它已经有 task-level frontline paper metadata，但更丰富的 reference-side detail 仍是下一步要接入的 evidence layer。

### Master proposal setting 和 V1 是否一致

结论：不一致。V2.2 继承了 V1 的“结构化 proposal”目标，但 generation setting、输入信息和 output schema 都已经换成 benchmark-aware harness setting。

V1 的 proposal generation 是 proposal_rl 里的 prompt-builder 逻辑，核心是从 paper reference list 出发：

- condition strategy 有 full_refs、top_k_refs、related_work、top_k_related_work、with_research_question。

- 输入主要是 paper refs、related-work narrative 或 research question。

- 输出 schema 偏 paper-proposal：problem、gap、key_insight、approach、expected_contributions。

- 训练和 RL reward 主要围绕 PRS/FAS/PPL/cosine similarity 等 paper-level alignment 信号。

V2.2 的 master proposal generation 是 end-to-end harness 逻辑：

- master model：Opus 4.7。

- endpoint：Google Anthropic rawPredict。

- temperature：null，因为 Vertex Claude 对该模型不接受 temperature。

- max tokens：默认 2600。

- 输入：完整 task_packet.json，包含 benchmark task context、baseline、pass line、worker constraints 和 frontline evidence。

- 输出 schema 偏 implementation-ready benchmark proposal：problem、gap、core_idea、implementation_plan、evaluation_plan、expected_results、risks_and_limitations。

因此，V1 更像“从文献阅读列表生成未来论文方向”；V2.2 更像“从 benchmark context 生成一个 worker 可以立刻实现并评测的 research idea”。两者不是同一个 prompt，只是在研究方向上连续：V1 训练 proposal 习惯，V2.2 把 proposal 放进真实 implementation/evaluation loop。

### Expert 能看到什么

Expert prompt 在 autoresearch_idea_harness/src/autoresearch_idea_harness/prompts.py 的 end_to_end_forecast_prompt()。

每个 expert 看到的是：

1. success definition：预测一个 competent worker 按 proposal 实现后，是否能产出通过 task threshold 的 signed benchmark result。

2. 完整 task_packet：包括 task context、baseline、pass metric、metric direction、frontline papers、worker constraints。

3. 匿名 proposal：只包含 proposal_id、label 和 proposal text。

Expert 不会看到：

- proposal 是哪个 model 生成的。

- generator_id、model_id、call_id。

- 其他 expert 的 forecast。

- worker result、worker log、eval log。

- API key、auth header 或 .env 内容。

Expert 返回 compact JSON：

```json
{
  "success_probability": 0.0,
  "rationale": "one concise paragraph",
  "strengths": ["short point 1", "short point 2"],
  "weaknesses": ["short point 1", "short point 2"],
  "baseline_risk": "one sentence"
}
```

当前 market 的 aggregation 很简单：对同一 proposal 的 expert probabilities 做 equal-weight mean，默认不低于 0.5 才通过 gate。worker 跑完后，系统会用实际 pass/fail 对 expert 进行 Brier score 和 log score settlement，并把结果写入 expert_reliability.jsonl。后续可以把这个历史 reliability 用作 weighted market，而不是永远 equal-weight。

### 为什么这套可观测性重要

这套 workflow 的重点不是让每个 proposal 都成功，而是让失败变得可定位：

- 如果 expert 没放行，说明 proposal 在实施前就被判断为低胜率。

- 如果 expert 放行但 worker 失败，说明 expert calibration、worker ability 或 proposal implementation detail 有问题。

- 如果 worker 能实现但 benchmark 不过，说明 idea 本身或者 pass line 设置需要重新分析。

- 如果 worker 卡住，可以查看 worker prompt、worker log、eval log 和 master-worker dialogue，而不是只看到一个 timeout。

这也是 V2.2 相比 V1 的核心变化：proposal 不再只是一个文本 target，而是进入一个可执行、可审计、可结算的 autoresearch loop。
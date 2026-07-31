## Summary

V2.3 的目标是把 autoresearch_idea_harness 推进到 formal sweep 阶段：<font color="#F06A1D">**10 个 MLS tasks、9 个 proposal modules、每格 20 samples，并把 expert reliability 作为主要评估对象。**</font>所有输出必须写到 V2 harness 目录，不写入 V1 proposal_rl/runs/benchmark。

## Task Suite

已有 3 个 MLS tasks 使用 260605 worker-only ceiling 阈值：

| Task | Pass Metric |
| --- | --- |
| dl_lr_schedule | 93.19 |
| dl_activation_function | 93.35 |
| cv_data_augmentation | 93.76 |

<font color="#F06A1D">**新增 7 个 MLS tasks**</font> 先跑 worker-only calibration，再把 threshold 设为<font color="#F06A1D">** best_worker_score + 0.01pp：**</font>

| Task | Reason |
| --- | --- |
| dl_weight_initialization | 研究味强，单函数 initialization 策略 |
| cv_classification_loss | loss design，易审计、运行较快 |
| cv_sample_weighting | long-tail reweighting，科研问题明确 |
| cv_pooling_aggregation | feature aggregation，单 module 可编辑 |
| cv_multitask_loss | multi-task loss balancing，科研味强 |
| dl_regularization | regularization design，单函数可编辑 |
| dl_residual_connection | block design，有结构创新空间 |

## Proposal Modules

正式 sweep 使用 9 个 proposal modules：

| Module | Source |
| --- | --- |
| empty_worker_only | no proposal control arm |
| opus47_master | V2 task-packet Opus 4.7 master |
| qwen25_7b_base_with_research_question | V1-style base checkpoint + strategy |
| exp09_top_k_related_work | latest exp09 RL checkpoint + strategy |
| exp11_top_k_related_work | latest exp11 RL checkpoint + strategy |
| exp12_with_research_question | latest exp12 RL checkpoint + strategy |
| exp13_top_k_refs | latest exp13 RL checkpoint + strategy |
| exp16_top_k_refs | latest exp16 RL checkpoint + strategy |
| exp17_with_research_question | latest exp17 RL checkpoint + strategy |

Checkpoint weights remain under proposal_rl/runs/exps/*/rl/final; V2 only reads them.

## Reliable Expert Objective

V2.3 不只看 pass rate，也要判断 expert 是否可靠。每个 expert forecast 必须完整落盘：

```plaintext
expert_prompts/<expert_id>.json
expert_responses/<expert_id>.json
expert_forecasts.jsonl
settlement.json
```

每条 forecast 至少包含：

```json
{
  "expert_id": "opus47 or gpt55",
  "expert_model_id": "...",
  "success_probability": 0.0,
  "feasibility": 0.0,
  "novelty_over_worker_best": 0.0,
  "expected_delta": 0.0,
  "risk": 0.0,
  "confidence": 0.0,
  "rationale": "...",
  "usage": {},
  "latency_s": 0.0,
  "error": null
}
```

Worker result 出来后结算：

- <font color="#F06A1D">**Brier score （均方误差，用来衡量该export的准确度）**</font>

- log score

- outcome pass/fail

- raw metric and improvement

- calibration bucket

- rank correlation inputs

- append-only expert_reliability.jsonl

## Run Workflow

1. Implement V2-only task adapter and generic MLS evaluator.

2. Calibrate new 7 tasks with empty_worker_only n=20.

3. Freeze thresholds in V2 calibration artifact.

4. Run formal sweep: 10 tasks × 9 modules × 20 samples.

5. Keep dashboard-readable registry/index.

6. Generate final reports:

  - module_matrix.json

  - task_matrix.json

  - expert_reliability.jsonl

  - expert_calibration_report.md

  - formal_sweep_report.md

Run root:

```plaintext
autoresearch_idea_harness/runs/formal_sweeps/v2_3_mls10_modules9/
```

## Agent Routing

- coder/cc103: implement only in autoresearch_idea_harness/; V1 is read-only reference.

- benchmarker/cc102: run calibration/sweep, monitor errors, update agent-memory/benchmarker/evals.md.

- exp-manager/cc101: supervise GPU/machine allocation and long-running process health.

- Codex meta: coordinate, audit artifacts, update memory and REDoc.

## Required Per-Sample Artifacts

```plaintext
task_packet.json
proposal.txt or empty_control.json
expert_forecasts.jsonl
expert_prompts/<expert_id>.json
expert_responses/<expert_id>.json
worker_prompt.txt
worker.log
eval.log
result.json or error.json
settlement.json
```

## Test Plan

- python -m py_compile all changed V2 Python files.

- Instantiate all 10 task packets.

- Dry-run proposal generation for all 9 modules on one task.

- Run one fixture worker per new task.

- Run one real empty_worker_only sample on a fast task.

- Verify expert logs contain no secrets and include prompt, response, parsed forecast, usage, latency, and settlement fields.

- Verify dashboard can inspect task, module, proposal, expert forecast, worker log, eval log, result, and settlement.

## Report Update: Worker-Only Baselines

截至 2026-06-08，V2.3 已经从纯 plan 进入 report 阶段。正式 sweep 的 pass line 不再使用旧的主观阈值，而是以 worker-only control 的实际能力作为参照：每个 task 先看 worker 在没有 proposal 时能达到的最好结果，再把 pass line 设为 worker-only best + 0.01pp。这样后续 proposal module 必须超过 worker 自己能想到的最好方案，才算有明确增量。

前三个 MLS task 使用 260605 worker-only 报告中的 sandbox 结果：

| **Task** | Subtask | Reference Baseline | Worker-Only Best | **Pass Line** | Source |
| --- | --- | --- | --- | --- | --- |
| **dl_lr_schedule** | resnet20-cifar10 | warmup_cosine 92.71 | 93.18 | **93.19** | 260605 worker-only report |
| **dl_activation_function** | resnet20-cifar10 | GELU 92.97 | 93.34 | **93.35** | 260605 worker-only report |
| **cv_data_augmentation** | resnet20-cifar10 | Cutout 93.67 | 93.75 | **93.76** | 260605 worker-only report |

新增七个 MLS task 使用 V2.3 calibration run，全部为 empty_worker_only，n=20：

| **Task** | Subtask | N | Worker-Only Mean | Worker-Only Median | Worker-Only Best | **Pass Line** | Best Sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **dl_weight_initialization** | resnet56-cifar100 | 20 | 72.5885 | 72.58 | 73.25 | **73.26** | s04 |
| **cv_classification_loss** | resnet56-cifar100 | 20 | 70.6050 | 72.37 | 73.47 | **73.48** | s02 |
| **cv_sample_weighting** | resnet32-cifar10lt | 20 | 71.6860 | 71.785 | 74.43 | **74.44** | s05 |
| **cv_pooling_aggregation** | resnet56-cifar100 | 20 | 71.0255 | 70.995 | 72.18 | **72.19** | s06 |
| **cv_multitask_loss** | resnet20-cifar100mt | 20 | 66.8340 | 66.30 | 68.78 | **68.79** | s02 |
| **dl_regularization** | resnet56-cifar100 | 20 | 72.1325 | 72.11 | 72.99 | **73.00** | s08 |
| **dl_residual_connection** | resnet20-cifar10 | 20 | 92.5075 | 92.575 | 92.95 | **92.96** | s00 |

当前 V2.3 的主要比较基准因此是：proposal module 是否能超过对应 task 的 worker-only pass line，而不是是否超过一个容易或过难的历史 baseline。这个设定会使 pass 更稀疏，但更适合评估 master/proposal 是否真的提供了 worker 自身没有的科研增量。

相关本地 artifacts：

- autoresearch_idea_harness/configs/v2_3_thresholds.json

- autoresearch_idea_harness/runs/formal_sweeps/v2_3_calibration_worker_only/

- 260605 Worker-Only 基准实验报告

## 260608 Report Update: First-Batch Formal Sweep Snapshot

2026-06-08 16:35 的 first-batch progress 已压缩归档。那一版只覆盖 57 个 materialized samples、24 个 completed samples，且几乎全部来自 empty_worker_only，因此不再作为当前判断依据。完整本地报告仍保留在 autoresearch_idea_harness/runs/formal_sweeps/v2_3_mls10_modules9/analysis_reports/first_batch_1635.md；当前正文以后面的 latest progress matrix 和 expert analysis 为准。

## 260609 Report Update: Latest Formal Sweep Progress Matrix

更新时间：2026-06-09 12:23 CST。这个 section 更新并替代前面的 first-batch progress snapshot。矩阵中每个单元格格式为 running/pass/fail，也就是当前正在跑的数量、已经 pass 的数量、已经 fail 的数量。加粗单元格表示当前看起来比较好的 setting：要么 pass 数量已经不少，要么在已完成样本里的 pass-rate 较高。

总体进度：

| Metric | Value |
| --- | --- |
| Planned total | 1800 samples |
| Materialized | 649 |
| Done | 617 |
| Running | 32 |
| Pass | 91 |
| Fail | 526 |
| Current terminal errors | 0 |

当前 sweep 已经完成 worker-only、opus47_master 两个完整模块；qwen-base 接近完成；exp09 已经开始产出；exp11 刚开始 materialize。后面的 exp12、exp13、exp16、exp17 还没有开始生成可比较结果。

### 10 x 9 Progress Matrix

| Task | worker-only | opus47 | qwen-base | exp09 | exp11 | exp12 | exp13 | exp16 | exp17 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dl_lr_schedule | 0/0/20 | 0/3/17 | 0/0/20 | 5/1/6 | 1/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| dl_activation_function | 0/2/18 | 0/1/19 | 0/0/20 | 4/0/8 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| cv_data_augmentation | 0/1/19 | 0/0/20 | 1/2/17 | 3/0/6 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| dl_weight_initialization | 0/0/20 | 0/1/19 | 1/1/18 | 2/0/3 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| cv_classification_loss | 0/1/19 | 0/0/20 | 1/0/18 | 3/0/4 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| cv_sample_weighting | **0/7/13** | 0/0/20 | 0/3/14 | 1/1/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| cv_pooling_aggregation | 0/3/17 | **0/6/14** | 0/3/16 | 0/0/3 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| cv_multitask_loss | **0/17/3** | **0/5/15** | **0/10/8** | **0/3/1** | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| dl_regularization | 0/0/20 | 0/1/19 | 4/0/13 | 1/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| dl_residual_connection | **0/8/12** | 0/1/19 | **3/9/8** | 2/1/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |

### Module-Level Snapshot

| Module | Running | Pass | Fail | Done | Pass Rate Among Done |
| --- | --- | --- | --- | --- | --- |
| empty_worker_only | 0 | 39 | 161 | 200 | 19.5% |
| opus47_master | 0 | 18 | 182 | 200 | 9.0% |
| qwen25_7b_base_with_research_question | 10 | 28 | 152 | 180 | 15.6% |
| exp09_top_k_related_work | 21 | 6 | 31 | 37 | 16.2% |
| exp11_top_k_related_work | 1 | 0 | 0 | 0 | - |
| exp12_with_research_question | 0 | 0 | 0 | 0 | - |
| exp13_top_k_refs | 0 | 0 | 0 | 0 | - |
| exp16_top_k_refs | 0 | 0 | 0 | 0 | - |
| exp17_with_research_question | 0 | 0 | 0 | 0 | - |

### Early Readout

- cv_multitask_loss 的 pass 很多，包括 worker-only 也很高，说明这个 task 的 worker-only pass line 可能相对容易被随机/worker variation 超过。它对 expert reliability 有用，但不一定是最强的 proposal-quality 区分任务。

- dl_residual_connection 上 qwen-base 当前是 3/9/8，已经比 worker-only 的 0/8/12 略强，但还有 3 个样本在跑，最终要等完整 20 个样本再判断。

- cv_pooling_aggregation 上 opus47 是 0/6/14，比 worker-only 的 0/3/17 更好，是目前一个值得关注的正信号。

- cv_data_augmentation 的 qwen-base 出现了当前最高单样本 93.80，但该 cell 目前 pass 数是 2/19 completed，整体还不算稳定。

- 当前没有 terminal error；events 里的历史 sample_error 主要是中间 parse/retry/resume 事件，目前没有样本以 error 终态停住。

## Report Update: Latest Expert Analysis

更新时间：2026-06-09 12:23 CST。当前 expert analysis 已经可以初步看 calibration，但还不能下最终结论：completed samples 主要覆盖 worker-only、opus47、qwen-base 和刚开始的 exp09，后续 exp11、exp12、exp13、exp16、exp17 还没有足够 settled samples。

当前 forecast 覆盖情况：

| Metric | Value |
| --- | --- |
| Forecast rows | 1227 |
| Samples with forecasts | 649 |
| Common settled samples with both experts | 546 |
| Passes in common settled set | 91 |
| Empirical pass rate in common settled set | 16.7% |

### Expert Calibration Snapshot

按各自 settled samples 计算：

| Expert | Settled N | Pass | Empirical Pass | Mean P | Mean Confidence | Brier | Log Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt55 | 546 | 91 | 16.7% | 0.395 | 0.626 | 0.2100 | 0.6044 |
| opus47 | 617 | 91 | 14.7% | 0.396 | 0.581 | 0.2216 | 0.6350 |

在 common settled set 上，gpt55 和 opus47 的 mean P 几乎一样，都是约 0.39-0.40，但真实 pass rate 只有 16.7%。这说明当前两个 expert 都明显偏乐观。gpt55 的 Brier 略好，但差距不大，且当前样本分布还不均衡。

### Calibration Buckets

gpt55：

| P Bucket | N | Mean P | Empirical Pass |
| --- | --- | --- | --- |
| 0.00-0.25 | 123 | 0.161 | 9.8% |
| 0.25-0.40 | 141 | 0.322 | 17.0% |
| 0.40-0.55 | 134 | 0.465 | 23.1% |
| 0.55-0.70 | 143 | 0.592 | 16.8% |
| 0.70-1.00 | 5 | 0.724 | 0.0% |

opus47：

| P Bucket | N | Mean P | Empirical Pass |
| --- | --- | --- | --- |
| 0.00-0.25 | 97 | 0.166 | 11.3% |
| 0.25-0.40 | 208 | 0.321 | 18.8% |
| 0.40-0.55 | 181 | 0.435 | 18.8% |
| 0.55-0.70 | 94 | 0.573 | 7.4% |
| 0.70-1.00 | 37 | 0.778 | 0.0% |

目前 bucket 并不单调，尤其是高概率 bucket 反而没有更高 pass-rate。这是一个重要信号：expert 现在更像是在给“听起来合理”的 proposal 较高分，而不是稳定预测最终是否超过 worker-only pass line。

### Module-Level Expert Behavior

| Expert | Module | N | Empirical Pass | Mean P | Brier |
| --- | --- | --- | --- | --- | --- |
| gpt55 | empty_worker_only | 195 | 20.0% | 0.475 | 0.2328 |
| gpt55 | opus47_master | 137 | 13.1% | 0.481 | 0.2674 |
| gpt55 | qwen-base | 177 | 15.8% | 0.270 | 0.1542 |
| gpt55 | exp09 | 37 | 16.2% | 0.257 | 0.1449 |
| opus47 | empty_worker_only | 200 | 19.5% | 0.475 | 0.2739 |
| opus47 | opus47_master | 200 | 9.0% | 0.429 | 0.2313 |
| opus47 | qwen-base | 180 | 15.6% | 0.290 | 0.1606 |
| opus47 | exp09 | 37 | 16.2% | 0.302 | 0.1840 |

一个初步现象是：两个 expert 都对 empty_worker_only 和 opus47_master 给了偏高概率，但这两个模块的真实 pass-rate 并没有对应升高。相反，qwen-base 和 exp09 的 mean P 更低，Brier 暂时更好。这可能说明 expert 对“proposal 文本质量/语言可信度”的偏好和最终 benchmark gain 之间存在偏差。

### Confidence Readout

self-reported confidence 目前不能直接当成 calibrated confidence。gpt55 的 mean confidence 约 0.626，opus47 约 0.581，都是中等偏高；但两者 mean P 约 0.40，而真实 pass rate 约 0.15-0.17。也就是说，expert 并不是极端自信，但仍然系统性高估了 pass probability。

更重要的是，confidence 目前还没有证明能区分“会成功”和“不会成功”。后续应该把 confidence 作为 feature 进入 expert calibration model，而不是直接用来加权 market。

### Action Items For Reliable Expert

- 继续收集后续模块的 settled samples，尤其是 exp11/exp12/exp13/exp16/exp17，避免只根据前四个模块判断 expert。

- 在 report 中同时保留 raw expert probability、confidence、Brier、log score 和 bucket calibration，不只看平均 P。

- 对 expert 做 post-hoc calibration：至少需要 temperature scaling、isotonic calibration 或 bucket-based reliability weighting。

- 把 “proposal text 看起来合理” 和 “最终 benchmark 真正超过 worker-only” 分开评估；当前数据已经显示二者可能不一致。

- 后续 market gate 不应直接使用 raw P，应该使用 calibrated P 或 reliability-weighted P。
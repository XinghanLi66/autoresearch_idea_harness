_Living tracker — 持续更新。owner: cc001 (benchmarker)。最近更新 2026-07-20。_

<redoc-highlight emoji="dengpao" fillColor="orange">
**V3 一句话**：训练 LLM 产出「被 worker 实现后能真实提升 ML 任务指标」的 research idea —— **execution-grounded idea quality**，而非只靠 LLM-judge。这是相对其他 ideation 工作的核心壁垒。当前算力：**GB200**（训练）+ **128×L20Z farm**（PAI DLC ws 262162 / quota1shcr2h7uae；评测执行农场，当前约 106/128 占用）。
</redoc-highlight>

## ✅ 已完成 (Done)

<redoc-highlight emoji="dui" fillColor="green">
Benchmark 管线已收敛到「github MLS-Bench 原生 CLI」；preference data 已建；eval 全链路 smoke 通过。
</redoc-highlight>

| 项 | 详情 |
| --- | --- |
| Benchmark 选定 | 官方 **MLS-Bench-Lite 30 tasks**（12 domain），源=github Imbernoulli/MLS-Bench；**弃用 HuggingFace**（已停维护）。清单 docs/eval/mls_bench_lite_tasks.json |
| 管线极简化 | 删除 27 个 bespoke/legacy 文件；新管线只用**原生 mlsbench CLI**（run_mls_lite_eval.py + _mls_lite_common.py），Δ-over-baseline + 多 seed CI |
| Eval 全链路 smoke | 我的 driver 端到端跑通：proposal → --extra-context → mlsbench agent（fable-5 worker，shim :18791）→ leaderboard → Δ |
| Eval 环境 (cc002) | D0 conda runtime（无 docker），10 个 cheap task 就绪（cv/dl/ml/rl/opt/ts）；proposal 注入 patch 已合入 |
| Preference data | **2,437 对**：trivialize 1,626（margin-5 强）+ route 811（margin-3 弱辅助）；<font color="#F06A1D">**供 DPO/IPO + Bradley-Terry RM；可以考虑大模型/小模型 rollout做DPO**</font> |
| Model-zoo SFT recipe | 6 arm、逐 arm 超参、Qwen3 non-thinking 要求已定稿（docs/plan/model_zoo_sft_recipe.md）；anchored 数据 1,626/85 已定位 |
| Keys | fable-5 / opus-4.8 / opus-4.6 可用；sonnet-4.6、gpt-5.5 key 已失效 |
| 先前 failure-mode 调查 | worker 忠实；SFT 修好 proposal **形式** 而非 novelty；RL reward 有 bug；**benchmark ceiling 是主导限制** → 催生本轮 easier-benchmark + zoo + RL-v2 |

## 🔧 进行中 & 主计划 (In progress / plans)

| track | owner | 状态 |
| --- | --- | --- |
| Model-zoo SFT（Qwen3 8B/14B/32B FT + 30B-A3B/235B LoRA + DeepSeek） | cc000 exp-manager | **全并行启动中**（S1 先跑做 recipe 校验，28 idle GB200） |
| L20Z serving fabric（worker + judge + proposal-gen） | 待派（cc002/coder 方向） | **待启动** —— E1/E2/E6 的前置 |
| RL-v2（DPO/IPO 先，GRPO-BT 后；drop PPO） | cc001 备数据 → cc000 训 | preference data 就绪；on-policy 负样本随 SFT ckpt 补 |
| Phase-2 eval（base vs SFT，30 Lite，多 seed Δ+CI） | cc001 | 待 SFT ckpt + serving |
| 报告 / RedDoc / arXiv | cc003 reporter | 独立线 |

## 🧪 实验议程 (Experiment agenda)

<redoc-highlight emoji="tuding" fillColor="yellow">
用户 2026-07-17 **greenlit：E1 + E2 + E3 + E4 + E6**（+ 建 serving fabric）。E5/E7/E8/E9 暂缓、可回访。
</redoc-highlight>

| id | 问题 | 算力 | 状态 |
| --- | --- | --- | --- |
| **E1** | idea 质量的 **scaling**（size × series × structure，闭环 Δ + rubric） | GB200 训 + L20Z 评 | greenlit |
| **E2** | ideation 的 **test-time scaling**（best-of-N / pass@k，Δ-vs-N） | L20Z serving | greenlit |
| **E3** | **RL-from-rubric** 消融（DPO/IPO/KTO/GRPO-BT/scalar）+ **reward-hacking / diversity-collapse** 审计（用 grounded Δ 而非 reward） | GB200 | greenlit |
| **E4** | **novelty-by-retrieval**（vs 7,441 arxiv 语料）；novelty 与 Δ 相关性 | CPU/L20Z | greenlit |
| **E6** | **judge/worker 泛化**（多个 served open model 当评委/worker，去 opus-judge-opus 循环偏差） | L20Z serving | greenlit |
| E5 | RL 下 idea **多样性 / mode-collapse** 量化 | GB200 | 暂缓 |
| E7 | **agentic revise loop**（propose→implement→feedback→revise vs 单发） | L20Z | 暂缓 |
| E8 | **公开 MLS-Bench-Lite leaderboard** 定位（vs frontier + human） | L20Z | 暂缓 |
| E9 | **data 消融**（teacher 强度、persona 条件、count vs diversity） | GB200 | 暂缓 |

## ❓ 开放问题 & 待定消融 (Open questions / deferred)

<redoc-highlight emoji="gantanhao" fillColor="red">
这些**暂不做**但值得回访 —— 记录以免遗忘。
</redoc-highlight>

- **arxiv/MLE easy-negative track**（用户 07-17 定为 open direction）：只有 reformat 进我们模板才可用（raw text = genre shortcut）；arxiv 语料在（7,441），MLE report 本地无源；配对有 setup/topic confound。

- **SFT 数据量是否够**：现用 1,626 跑 zoo，若大 model 过拟合/退化再扩（teacher-upgrade / 加 route）。判据：val + creativity eval 是否随 size 退化。

- **2-epoch full-FT** OOM（fp32 AdamW）→ 需 8-bit optimizer 才能 2 epoch。

- **MLAgentBench** port（≥10%-over-baseline 主基准）：计划内、尚未实现。

- **RL reward v2 细节**：k3 KL、去饱和 format 项、pairwise creativity、rollout≥32（E3 内验证）。

- **DeepSeek arm tag** 未定（latest trainable distill vs V4-Flash LoRA）—— 待与 cc000 确认。

## 🗂 基础设施 & 资产 (Infra / assets)

- **Compute**：QS2 queue 532（GB200，48 张，28 idle）；research_agent quota 上 ~100 L20Z。

- **Agents**：cc000 exp-manager（QS 训练）· cc001 benchmarker（本 doc / eval / preference / 分析）· cc002 coder（eval 环境 + serving）· cc003 reporter（文档）· cx001（QS env）。

- **数据**：anchored SFT runs/training_data/v3_researcher_cot_anchored（1626/85）；preference runs/researcher_cot/preference/pairs_&#123;trivialize,route&#125;.jsonl；arxiv 语料 data/arxiv_tex_classified/v1/papers.jsonl（7,441）。

- **MLS-Bench**：/newcpfs/lxh/MLS-Bench（github clone）；worker 走 shim http://127.0.0.1:18791，模型串用 claude-fable-5。

- **关键脚本**：run_mls_lite_eval.py、_mls_lite_common.py、build_v3_preference_pairs.py、ablation_failure_mode.sh、verify_v3_qs_checkpoint.py。plan：docs/plan/&#123;model_zoo_and_easier_benchmarks,model_zoo_sft_recipe&#125;.md。

## 📓 变更日志 (Changelog)

- **2026-07-17**：pivot 到 easier benchmark（官方 30-task Lite）+ model zoo；管线换原生 mlsbench；preference data 建成（2,437 对）；eval smoke 通过；greenlit E1/E2/E3/E4/E6 + L20Z serving；SFT 交 cc000 全并行。

## 📓 追加更新 2026-07-17 (b)

- **hparam-swap 已接线**：run_mls_lite_eval.py --hparam-mode fixed&#124;swap（fixed = idea-only 归因，为主；swap = 允许 worker 调 lr/batch/epochs，作为消融）。

- **swap 消融 → waitlist**（暂缓，稍后回访）；每个 eval 主跑用 fixed，swap 二次跑复核 robustness。

- **compute-soak 训练队列**：scripts/gen_training_queue.py → runs/qs_training_queue/submit_all.sh（10 个 priority-0 lr-sweep 训练变体，backfill 空闲 GB200）；已交 cc000（待其 stage Qwen3 权重后 fire）。

- **DPO/IPO 数据集就绪**：runs/researcher_cot/preference/pairs_all.jsonl（2,437 对 = trivialize 1,626 + route 811）。RL-v2 可在现有 32B SFT checkpoint-101 上先起。

## 🖥 Eval serving & 算力估计 2026-07-17

**架构**（三分量分开）：proposer（我们的 ckpt，vLLM on L20Z）· worker（Claude fable-5 via shim :18791）· **task-execution（resnet/gpt/robomimic 训练 = 真瓶颈）→ L20Z 集群并行，一 GPU 一 task-run**。**GB200 = 训练；L20Z = 评测执行农场 + 轻量 serving**。MLAgentBench 复用同农场。

**单 checkpoint 评测耗时**（30 Lite tasks，per-task budget 合计 73 GPU-h/seed，长尾：6 个重任务占 58% —— llm-pretrain 12h / robomimic 8h / jepa·robo-diff·llm-rl 6h）：

| tier | tasks | GPU-h/seed | wall（≥30 L20Z 并行） |
| --- | --- | --- | --- |
| fast | 24 | ~31 | ~2h |
| full | 30 | ~73 | ~12h（长尾主导） |

**策略**：fast-24 × 3 seed 每 ckpt（~2-3h，全 zoo 快速对比）；6 个重任务只在 top arm 上 1 seed / proxy 缩短。已把「L20Z serving + 并行执行农场 + tier 拆分」交给 cc002（Request 2）。

## ⏸ 追加 2026-07-17 (c) — MAB 明确暂缓

**MLAgentBench port 暂缓（用户指示：先做完 MLS-lite）。** MAB = 独立 harness（13 tasks，≥10%-over-baseline），未 clone；将来用 option-A（把每个 task 包成 mlsbench schema）复用同一 L20Z farm。当前全力：跑通 MLS-lite 全 zoo 评测。

## 🟢 追加 2026-07-17 (d) — MLS-lite code-complete + S1 就绪

- **cc000: S1 Qwen3-8B SFT 完成**（eval 2.087，non-thinking，无 &lt;think&gt;）；S2/S3/M1/D1(R1-0528-Qwen3-8B)+lr-sweep 训练中。

- **DPO entrypoint** scripts/train_v3_researcher_cot_dpo.py（trl DPOTrainer，LoRA on frozen SFT，ref=adapter-off，sigmoid&#124;ipo&#124;kto_pair，读 pairs_all.jsonl）已交 cc000 在现有 32B checkpoint-101 上起（待 stage pairs_all + confirm 路径）。

- **MLS-lite 30-task packets** build_mls_lite_packets.py → packets.jsonl 完成。MLS-lite 代码侧齐活（packets+driver+hparam-mode+DPO）；**剩执行**：cc002 farm/served-proposer + 从 ckpt 生成 proposals → 全 zoo Δ 评测。

## ⏸ Deferred（goal 2026-07-17，记录以免遗忘，稍后回访）

明确暂缓、待 base/SFT/RL 训练报告完成后回访：**(1) MLAgentBench port**（第二基准，option-A 复用同 farm）；**(2) teacher upgrade**（opus-4.8 重生成 CoT 数据以攻 novelty 天花板）；**(3) frontier ceiling**（opus-4.8/fable 作为 proposer 上界对照）。当前目标：跑完 model-zoo SFT + RL + MLS-lite 评测 → /redoc 训练报告（rescaled 50=baseline 表 + Δ + takeaway + next step）。

## 🟢 追加 2026-07-17 (e) — goal: base/SFT/RL 训练报告（rescaled 50=baseline）

- **S1 Qwen3-8B + S2 Qwen3-14B 已验证 CLEARED**（format 1.00 / anchor 3/3 / in-voice；creativity 0.65 / fp 0.07-0.12）→ 可入 Phase-2。S3(32B)/M1(30B-A3B) staging 中；D1(R1-0528) tokenization 失败（DeepSeek template，诊断中，单 arm 不阻塞）；multi-seed S1/S2(43/44) 已起（供 Δ 方差）。

- **报告链路就绪并验证**：rescaling = mlsbench score &lt;task&gt; --format json 的 task_score ×100（best-baseline 校准到 0.5 → **50 = strong baseline**，100 = 理论上界）；scripts/mls_lite_report.py 产表（每 task base/SFT/RL + Δ）+ RedDoc viz + takeaway，已用 stand-in arms 跑通。

- **待执行**（gated on cc000 训练完成 + cc002 farm/proposer + arm-distinct leaderboard tags）：从 ckpt 生成 proposals → 30-task 多 seed 评测 → 出报告。DPO 待 pairs_all.jsonl(25MB) gzip+chunk-stage（已给 cc000 call，不走 git）。

## 🟢 追加 2026-07-18 — 评测链路端到端跑通（真实 farm 数据）

<redoc-highlight emoji="dui" fillColor="green">
**整条 deliverable 链路已验证**：served proposer → dispatcher → L20Z farm → arm-tagged leaderboard → mlsbench score×100（50=strong baseline）→ 报告表。validation（exp09rl，2 task）：dl-activation-function **55.2**、ml-clustering-algorithm **68.8**、均值 **62.0**（2/2 &gt;50，超强 baseline）。
</redoc-highlight>

- **base arm 就绪**：30 个 Qwen3-8B base proposals（/completions + 1-shot）。cc002 已起 Qwen3-8B-base proposer + heavy-6 envs 全建好（仅 nanoGPT tokenize gate llm-pretrain-optimizer）+ 15 个 mid-tier pkg envs 建设中（~14 task 现可跑）。

- **当前瓶颈**：cc000 把 S1/S2 ckpt 从 /mnt/3fs 传到 /newcpfs（尚未，gates SFT arm）；RL arm 待 DPO。base arm 一旦配齐 S1 即可跑 base-vs-SFT。

## 🟢 追加 2026-07-20 — 基准设施定稿 + model-zoo/RL 设施就绪 + eval sweeps 进行中

### ✅ 已完成 (newly done)

- **基准设施定稿**：native mlsbench CLI 流水线（run_mls_lite_eval.py），MLS-Bench-Lite 收敛到 **29 任务**（robo-humanoid-sim2real-algo 因 IsaacGym Preview-4 在 L20Z/Ada 上 segfault 已剔除）；L20Z farm dispatcher 上线（PAI DLC ws 262162，quota quota1shcr2h7uae，128×L20Z）。

- **Model-zoo SFT**：S1/S2/S3 = 从**已后训练的** Qwen3-8B/14B/32B 全参微调（注意 **Qwen3-32B 无 base 模型**，只有 chat/post-trained）；已上传 ModelScope（endpoint 用 .ai）。另建 Qwen2.5-32B 三元组作为跨系列对照。

- **偏好数据 + RL 设施**：pairs_all.jsonl（**2,437 对** = trivialize margin-5 + route）；Bradley-Terry 奖励头（train pairwise acc **0.864**）；DPO/IPO/KTO（trl DPOTrainer + LoRA）与 GRPO-BT trainer 就绪。

- **修正 base-arm 对照错误**：base 臂改用**每个 size 对应的已后训练 Qwen3**（base8b / base14b / base32bq3），与 S1/S2/S3 的 SFT 起点一致；此前误用 Qwen3-8B-Base 预训练模型，**相关结论已作废**。

- **基础设施整改 (2026-07-20)**：

  - 杀掉一个空转的 vLLM proposer serve（dlcfutmvegcwehqn / mls-serve-qwen25-32b-base）—— 生成完 base32b proposals 后仍占用 4 GPU、0% util 达 ~42h（vllm serve 不会自终止）。

  - dispatcher 崩溃修复：JobId 追踪失败时 warn+continue 而非 raise（原先会拖垮整批 sweep）。

  - heavy-seed 裁剪：run_mls_lite_eval.py 现对 6 个 HEAVY 任务只跑 **1 seed**，cheap 任务保留 **3 seed**（farm 是 GPU-quota-bound，heavy 尾巴主导 wall-clock）；并停掉 20 个已在跑的 heavy s2/s3 冗余任务（含数个卡住 40h 的 jepa-planning）。

  - 已请 exp-manager (cc000) 搭建 PAI DLC idle-job 监控（idle-GPU / 超时 / 重复提交检测），spec 见 agent-memory/exp-manager/pai_idle_job_monitor_request.md。

### 🔧 进行中 (in progress)

- **MLS-lite 评测 sweeps 正在跑**：base8b / base14b / base32bq3、base32b(Qwen2.5)、sft8b / sft14b / sft32b、rl，逐步补齐到 ~28-29/29 覆盖；farm GPU-quota-bound（当前约 **106/128** 占用）。worker = Claude **fable-5**（经每 pod 自带的 MaaS shim）。**评测仍在跑，暂无最终评分表**（覆盖率填充中，勿引用任何 MLS-lite 分数）。

- **RL 消融 (cc000)**：DPO / IPO / KTO 与 GRPO-BT。

### 🚀 主要计划 / 下一步

- **汇总训练报告（/redoc，符合可视化要求）**：按任务给出 base/SFT/RL 分数表，**使用 MLS-bench 的 rescaling**（**50 = strong baseline，100 = 理论上界，&lt;50 = 不如 baseline**），含 ΔSFT / ΔRL 列、takeaway 与「面向科学发现训练更强模型」的下一步。脚注：robo-humanoid 剔除；臂族（Qwen3 per-size 梯队 vs Qwen2.5-32B 三元组）。

- **消融**：LoRA vs 全参（8B）、persona-off、DPO vs GRPO-BT。

### ⏸ 推迟但不要忘记 (deferred, parked — 勿删)

明确暂缓、待 base/SFT/RL 训练报告完成后回访：

- **MAB（multi-armed / MLAgentBench）port**（第二基准，option-A 复用同 L20Z farm）。

- **teacher upgrade**（用 opus-4.8 重新生成数据，攻 novelty 天花板）。

- **frontier ceiling**（前沿模型作为 proposer 上界对照测量）。

- **30B-A3B / 235B LoRA + DeepSeek 模型 zoo**（D1，parked）。

<br/>

<br/>

下一步计划

- master model固定为训出来的r1-8b, worker换各种东西

- 可能要测一下mle之类的

- 补全30个task，18个arm
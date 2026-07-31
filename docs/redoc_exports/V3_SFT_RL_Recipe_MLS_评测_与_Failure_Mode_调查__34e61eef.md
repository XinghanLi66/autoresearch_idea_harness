## (abandoned draft — see the finalized doc: [https://docs.xiaohongshu.com/doc/b6b0310b597cc5c26a49a5f1a6fb0b60](https://docs.xiaohongshu.com/doc/b6b0310b597cc5c26a49a5f1a6fb0b60) )

## 4. 结论 & 解决方案

<redoc-highlight emoji="dui" fillColor="green">
**判定：worker 与 proposal-形式 环节健康；bug 集中在 RL reward、creativity 数据信号，而 MLS 评测本身的 ceiling/方差是当前最大限制。**
</redoc-highlight>

| 环节 | 解决方案 |
| --- | --- |
| RL reward | KL 改 **k3** 估计器；**去掉/降权** 饱和的 format 项；creativity 改 **pairwise / Bradley-Terry**（绝对分饱和）；加 novelty-via-dissimilarity；rollout 16→≥32；步数加大 |
| 数据 creativity 信号 | 用 **pairwise** creativity 标注（不饱和）；挖更难/更分化的 case |
| Eval / benchmark（最高优先） | 每 proposal **多 seed（n≥3）** 压方差；每 task **多 proposal（k≥3）**；把 --free-hparams 设为默认；报告 **相对 baseline 的 Δ + 置信区间**，而非贴近 baseline 的硬阈值 pass/fail；或换 headroom 更大的 task |
| Recipe / model | 先修 reward+eval（当前是 binding constraint）；再考虑 2-epoch（换 optimizer 解 OOM）/更大 model |

## 5. 可复现

- 消融编排脚本：scripts/ablation_failure_mode.sh（stage：gen-qs / frontier / judge / worker / worker-freehp），QS 出 proposal + 本地跑 worker。

- 关键脚本：generate_mls_proposals_3arm.py、judge_mls_proposals.py、run_mls_eval_3arm.py（--free-hparams）、verify_v3_qs_checkpoint.py。

- 数据 receipts：runs/researcher_cot/mls_eval/（proposals、proposal_judge、worker_runs、worker_runs_freehp、ablation_subset.json）。均在分支 **V3**（GitHub 同步）。
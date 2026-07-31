## Prompt for your agent

Build a simple, human-readable [localhost](localhost) dashboard for post-training runs.

Include these core panels:

1. Run summary: run name/id, model type, base model, recipe/stage (SFT or GRPO), dataset version, global step/epoch, last update time, status.

2. Training curves: train loss, GRPO reward/advantage stats if applicable, KL if applicable, grad norm, learning rate, tokens/samples seen.

3. Validation/test curves: val loss, main eval metric, benchmark pass rate/accuracy, best checkpoint marker.

4. GRPO rollout examples: show positive and negative rollouts with prompt, model output, score/reward, and short rule-based failure tags.

5. SFT example panel: show outputs on a fixed prompt set, with prompt, model output, and reference/label when available.

6. Failure case panel: recent bad examples and repeated failure modes such as NaN, malformed output, truncation, refusal, format error, tool error, repetition/collapse.

7. Checkpoint panel: latest checkpoint, best checkpoint, save times, eval score by checkpoint.

8. Data / generation stats: data source proportions, filtered/rejected samples, rollout acceptance rate, average completion length, stop reason distribution, format-valid rate, empty-output rate, truncated-output rate.

```bash
(this part is optional)
Also add a few lightweight but useful extras:
- Fixed probe set to visualize drift over time.
- Reward breakdown by component, not just total reward.
- Simple rule-based error taxonomy.
- Train-vs-test gap.
- Preference margin / chosen-vs-rejected gap if relevant.
- Sample efficiency trend.
- Distribution shift across recent windows.
- Recent changes log for hyperparameters/code/config.
```

Priorities:

- readable under debugging pressure

- compact and information-dense

- useful for spotting failures quickly

- deterministic / rule-based, not AI-generated summaries

Keep the first version minimal and operational, not pretty.

<br/>

<br/>

## 参考

rednote-hilab/TELL dashboard

[https://docs.xiaohongshu.com/doc/cb164751db16e638ef751a39d2d604b4](https://docs.xiaohongshu.com/doc/cb164751db16e638ef751a39d2d604b4)

<br/>

## 效果

![](https://xhs-doc.xhscdn.com/104004dg31uvp2rff3m0e5bos4s?redoc-w=950&redoc-h=531)

<br/>
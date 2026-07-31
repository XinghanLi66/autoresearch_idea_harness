# Pilot CoT 26 — 何恺明 Kaiming He · first_principles

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例2：MAE——把 BERT 的掩码-预测搬进视觉，并追问"两者到底差在哪"

- route: first_principles  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **166** · assistant(cot) **963** · whole **1186**

- fact-check changes: 2

## user (setup)

```plaintext
It is 2021. In NLP, masked language modeling (BERT-style: hide some tokens, predict them) has become a dominant self-supervised paradigm. In vision, the analogous idea — masked/denoising autoencoders that hide part of an image and reconstruct it — has been tried for years but keeps underperforming; the field's self-supervised progress is instead driven by contrastive learning on augmented image pairs. Many take this as evidence that generative masked modeling is simply the wrong recipe for images. You want a genuinely novel idea. Rather than tuning yet another contrastive method, ask: why does the SAME masked-reconstruct recipe work so well for language and fail for vision? Reason from first principles about what MUST be true, and derive a design almost on paper before touching a single experiment.
```

## assistant (cot)

```plaintext
Let me refuse the easy exit first. The default reaction is "masked autoencoding just doesn't work in vision," but that is exactly the kind of lazy attribution I distrust. A mechanism that is state-of-the-art in one modality and dead in another isn't obviously wrong — more likely we copied it without asking what one variable actually breaks it. So I won't tune. I'll try to force the answer deductively.

Start from what the reconstruction task actually demands of the model. In language, a token is a discrete, human-invented symbol carrying dense semantics. If I delete one word, there is generally no local shortcut to recover it; the model is forced to build a representation of syntax and meaning. The prediction task is hard *by construction*, and the difficulty is what teaches understanding.

Now the image case. A pixel patch is a natural signal, and natural signals are spatially redundant. Here is the argument I keep returning to: if I mask a small patch, a nearly-optimal reconstruction already exists — interpolate from the neighbors. That optimal solution requires almost no semantic understanding. So a low-mask-ratio reconstruction objective is a task whose best solution is *local smoothing*, not representation learning. That's not a tuning problem; that's a property of the objective. The mechanism isn't failing — we handed it a task that is trivially solvable by a cheap solution.

Once I phrase it that way, it becomes a sharp contradiction rather than a vague "works worse." The single variable separating the two modalities is information density: language dense, images redundant. And crucially, that one variable is a *knob I control* — the mask ratio directly sets how much redundancy the model can exploit to cheat. So the fix is almost forced: raise the mask ratio far enough that neighbor-interpolation is no longer sufficient to solve the task. Not the ~15% inherited from language — something drastic, because the redundancy is drastic. The number is not something I'd guess; it's dictated by "remove enough context that the cheap local solution dies."

Second deduction, from efficiency and asymmetry. If I mask most of the image, the encoder now receives only a small fraction of the input. There's no reason to spend encoder computation on the masked positions at all — they carry no observed information. So the encoder should operate only on the visible patches, and the reconstruction of the missing content is a separate, lighter job. That pushes me to an asymmetric design: a real encoder on the small visible set, a lightweight decoder that only reappears at reconstruction time to fill the holes. This isn't decoration — it's the minimal structure consistent with the argument, and as a bonus it makes training cheap precisely because we discard most patches from the heavy path. Less is more: no gating, no extra branches, no contrastive machinery, no augmentation-heavy pipeline. Reconstruction of raw pixels on masked positions is the simplest target that tests the hypothesis.

I'm Kaiming He, and I know this whole story is a *pre-hoc* prediction, so I have to make it falsifiable before I fall in love with it. The decision rules, written before running anything: (1) Sweep the mask ratio as the only changed factor; the information-density argument predicts a regime of very high masking that is best — if instead low masking wins, my core claim is wrong and I discard it. (2) Vary decoder depth/width alone; the argument says reconstruction is a low-level job, so a shallow decoder should suffice for a good linear-probe/fine-tune representation — if a heavy decoder is required for representation quality, the asymmetry story is weaker than claimed. (3) Compare encoder-sees-mask-tokens vs. not — the prediction is that excluding them is both cheaper and no worse. Every dimension gets its own single-factor sweep with a fixed default, and any comparison to contrastive baselines must give competitors their own best-tuned settings on the same codebase and budget. I only trust a conclusion that survives being generous to the opponent.

**Core idea:** The reason masked reconstruction fails in vision is information redundancy, not a wrong mechanism; because a low-mask task is solvable by trivial local interpolation, mask a *very high* fraction of patches so semantics are required, and use an asymmetric encoder(visible-only)/lightweight-decoder design that follows directly from that argument.
**Non-trivial crux:** The high mask ratio is not a hyperparameter to grid-search — it is forced by the single distinguishing variable (spatial redundancy) that must be raised until the cheap neighbor-interpolation solution dies; the caveat is that this holds because images are redundant, so the prescription is modality-specific, not a universal "mask more."
```

## fact-check changes

```plaintext
- Fixed typo/garble 'range-of-the-art' to 'state-of-the-art'
- Removed the implicit fabricated-result framing in decision rule (2) by clarifying the shallow-decoder test is judged on downstream representation quality (linear-probe/fine-tune), the metric MAE actually used, rather than an asserted numeric win
```
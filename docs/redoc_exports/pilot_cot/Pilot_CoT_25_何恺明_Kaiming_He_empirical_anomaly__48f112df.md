# Pilot CoT 25 — 何恺明 Kaiming He · empirical_anomaly

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例2：MAE——把 BERT 的掩码-预测搬进视觉，并追问"两者到底差在哪"

- route: empirical_anomaly  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **160** · assistant(cot) **1070** · whole **1287**

- fact-check changes: 3

## user (setup)

```plaintext
It is 2021. In vision self-supervised learning, contrastive methods dominate the leaderboards. Meanwhile, in NLP, masked language modeling (BERT-style: hide some tokens, predict them) has become the reigning pretraining paradigm. The obvious cross-modal move — mask patches of an image and reconstruct them (denoising / masked autoencoding) — has been tried repeatedly over the years, yet keeps underperforming contrastive learning by a clear margin. The community has mostly treated this as evidence that masked reconstruction is just weaker for images. You have a strong ViT backbone, a clean codebase, and a habit of not letting a stubborn gap slide. Propose a genuinely novel idea for why masked autoencoding lags in vision — and what to change about it.
```

## assistant (cot)

```plaintext
Let me start from the one fact that annoys me: the same recipe — hide part of the input, predict it — is a revolution in language and a persistent disappointment in vision. People shrug and say 'reconstruction just doesn't learn good visual features.' That's the lazy explanation, the equivalent of blaming overfitting. I don't want to accept it until I've ruled out the trivial stories. Is it the architecture? ViT is now the same Transformer BERT uses, so 'CNN inductive bias' is no longer an excuse. Is it scale, augmentation, optimization? Those are knobs, not a mechanism. If I keep turning knobs I'll never see the actual disease.

So I reframe it as a question, not a method: what makes masked autoencoding different between vision and language? I want the difference to be one variable, something I can name in a single word and then falsify.

Here's what I keep staring at. In language, if I mask one token, the model is forced into real understanding — syntax, semantics, world knowledge — because a word is a human-invented, information-dense symbol; you cannot cheaply interpolate it from its neighbors. In an image, a masked patch is usually recoverable from the patches right next to it. Pixels are a natural signal with heavy spatial redundancy. So when I ask the network to fill in a masked patch at the typical BERT-like ratio (~15%), I've handed it a task it can solve by low-level interpolation. It never needs to build a high-level representation to win. The pretext is too cheap.

That single word — information density — is my candidate essence. And I like it because it immediately forces a prediction rather than a story. If cheapness is the disease, then the cure is to make the task genuinely hard: remove so much of the image that neighborhood interpolation is no longer enough, so the only way to reconstruct is to understand objects and their layout. That points to a mask ratio far beyond NLP's 15% — something aggressive, on the order of most of the image gone. This is a construction argument in spirit: I'm not fine-tuning the old task, I'm reasoning from 'what would make the trivial solution impossible' to a design parameter directly.

Now the 'less is more' angle. Once I commit to hiding most of the image, a second thing falls out for free. If the encoder only ever sees the small visible fraction, and I never feed mask tokens into it, the encoder does far less work — I push the mask tokens and the reconstruction burden into a lightweight decoder that only exists during pretraining. That asymmetry isn't just decoration; it's motivated by the same insight, and as a bonus it makes training on a small visible subset much cheaper. Reconstruction to pixels is plausibly a low-level job, so the decoder might be shallow and narrow while the representation I actually care about lives in the encoder — but that is a hypothesis to test, not a given. I'd resist the temptation to add gating, extra losses, contrastive terms on top. The whole point is a structure so simple its every piece can be independently ablated.

How do I keep myself honest — because I know I'm capable of overfitting a nice story to whichever runs look good. I write the predictions and decision rules down before I run. Pre-hoc claim: accuracy should be non-monotonic in mask ratio and peak far higher than the NLP regime; if the optimum sits near 15%, my information-density thesis is wrong and I should say so. Second pre-hoc claim: the decoder can be made very shallow with little loss — if depth matters a lot, the 'reconstruction is low-level' argument is weak. I sweep one factor per table, mark the default row, change one line at a time: mask ratio, decoder depth/width, whether mask tokens enter the encoder, reconstruction target, augmentation dependence. And the comparison to contrastive methods must be apples-to-apples — same backbone, same budget, competitors tuned to their best, not run at defaults. A result only counts if it holds when the opponent is fed to the fullest, and if it survives a change of downstream task.

The caveat I keep: this is an argument about a difference in signal statistics, not a proof that reconstruction is universally superior. If the redundancy story were false — say the ablation refuses to show a high optimum — I'd prefer the simpler conclusion that the mechanism, not my variable, is the issue.

**Core idea:** The reason masked autoencoding underperforms in vision is information density — images are spatially redundant, so a normal-ratio masking task is solvable by interpolation and never forces high-level understanding; the fix is to mask a very large fraction of patches and pair it with an asymmetric encoder (sees only visible patches) and a lightweight decoder.

**Non-trivial crux:** The high mask ratio and the asymmetric, mask-token-free encoder are not two tricks but one consequence of a single essence variable; the claim only stands if an ablation shows the accuracy optimum sitting far above the NLP-like ratio under strictly matched, competitor-favoring comparisons — otherwise the redundancy thesis is falsified.
```

## fact-check changes

```plaintext
- (a) Removed hindsight framing that the decoder 'can be shallow and narrow' as a settled fact; recast as a hypothesis to be tested, consistent with the stated pre-hoc claim.
- (a) Changed 'A win only counts' to 'A result only counts' and added 'decision rules' to keep the honest, not-yet-run framing.
- (e) Replaced the English section header 'Less is More' phrasing kept but rendered as an English descriptive phrase ('the less is more angle'); noted the asymmetric encoder's efficiency benefit as a bonus rather than pure design necessity, which is historically accurate.
```
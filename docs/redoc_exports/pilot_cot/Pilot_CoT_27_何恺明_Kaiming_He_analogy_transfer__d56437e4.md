# Pilot CoT 27 — 何恺明 Kaiming He · analogy_transfer

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例2：MAE——把 BERT 的掩码-预测搬进视觉，并追问"两者到底差在哪"

- route: analogy_transfer  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **162** · assistant(cot) **1045** · whole **1264**

- fact-check changes: 3

## user (setup)

```plaintext
It is 2021. In NLP, masked language modeling (the BERT recipe of masking out a fraction of tokens and predicting them) has become a dominant self-supervised paradigm. In vision, self-supervised learning is dominated by contrastive methods that rely on carefully designed augmentations, large negative sets or momentum encoders. Denoising/masked autoencoding on images is an old idea, but for years it has lagged well behind contrastive learning. The obvious move is to keep improving contrastive pipelines. Instead, take the mask-and-reconstruct mechanism that works so well in language and ask why the SAME mechanism, ported to images, has stayed second-rate. Propose a genuinely novel idea for how to make masked autoencoding actually work — and scale — in vision.
```

## assistant (cot)

```plaintext
Start from the mechanism, not from a benchmark. I'm Kaiming He, and what nags me is a clean transfer that refuses to transfer. In language, mask a token, predict it, and you get representations that carry the field. The mechanism is almost embarrassingly simple. Take that exact mechanism to images and it underperforms contrastive learning by a wide margin. The lazy conclusion is 'masked modeling just doesn't suit vision.' I don't buy conclusions that blame the mechanism. When a mechanism succeeds in one modality and fails in another, the honest question is not 'is the mechanism wrong' but 'what is the one essential variable that differs between the two domains that makes it fail.'

So let me map the transfer explicitly. What transfers: the self-supervised objective of removing part of the input and reconstructing it, no labels, no hand-crafted augmentation zoo. What might break: the assumption that predicting a masked piece is a hard task that forces high-level understanding. In language that assumption holds because a word is a human-invented, semantically dense unit — mask one and the model genuinely has to reason about syntax and meaning to fill it. Now hold that up against an image. A patch is not a word. Images are natural signals with heavy spatial redundancy: a missing patch can very often be interpolated from its immediate neighbors with almost no semantic understanding. So the reconstruction task, ported naively, is too cheap. The model takes the local-texture shortcut and never has to learn anything high-level.

That single word — information density — is the essential variable. Language: low redundancy, high density, masking ~15% already creates a hard task. Vision: high redundancy, low density, masking ~15% leaves an almost trivial task. So the failure is not the mechanism; it's that we imported language's mask ratio along with the mechanism, and that ratio is calibrated to language's density, not vision's.

Once I frame it that way, the design almost derives itself, and it cuts hard against the borrowed default. If the disease is that neighbors make the task cheap, the cure is to remove enough of the signal that neighborhood interpolation is no longer sufficient — you have to push the mask ratio far higher than NLP's, to the point where large contiguous regions are gone and only holistic understanding can fill them. This is the anti-consensus move: not a fancier encoder, not more augmentations, but making the pretext task drastically harder by deletion. Less machinery, harder task. (I should stay aware this is intuition about redundancy, not a proof — some structured regions may still be locally guessable.)

And a very high mask ratio hands me a second gift, so I should let the mechanism follow the ratio rather than fight it. If most of the input is discarded, the encoder shouldn't waste compute processing mask placeholders at all — let it operate only on the small surviving fraction of real patches. Because I'm building on a Vision Transformer, this is natural: attention over an unordered set of visible patches doesn't need the missing ones present. Reconstruction of pixels is a low-level task, so the part that actually re-inserts the masked positions and paints them back can be lightweight and separate. An asymmetric split: a substantial encoder that sees only the visible minority, a light decoder that handles the reconstruction. That's not just elegant; it turns the hard-task decision into a large compute saving, which is what makes it scalable — the property I actually care about.

Now keep myself honest, because this whole story could be a post-hoc fairy tale I tell after cherry-picking. My pre-hoc prediction, written before running: representation quality should be non-monotonic in mask ratio and peak far above NLP's ratio; if the information-density thesis is right, that peak should show up on both fine-tuning and linear probing. Decision rule: sweep mask ratio as a single factor, one row changed at a time, default row marked; if quality peaks near the low NLP-style ratio, my thesis is wrong and I drop it. Same discipline for asymmetry: if feeding mask tokens to the encoder is as good or better, I keep the simpler symmetric form — otherwise the asymmetric split is justified partly on efficiency grounds. And any comparison to contrastive baselines must be apples-to-apples — same codebase, matched budget, competitors tuned to their best — otherwise a win means nothing.

**Core idea:** The gap between masked modeling in language and vision comes down to one variable, information density; images are spatially redundant, so port the mask-and-reconstruct mechanism but push the mask ratio very high to make the task genuinely hard, and pair it with an asymmetric encoder (sees only the visible minority) and a light decoder for scalable, effective self-supervised pretraining.

**Non-trivial crux:** The high mask ratio is not a hyperparameter tweak but the direct consequence of the density diagnosis — it must be validated by a single-factor sweep whose peak sits far above the NLP ratio; if the peak sat near the low ratio, the whole information-density thesis would be falsified and the design abandoned.
```

## fact-check changes

```plaintext
- Marked the 'less machinery, harder task' deletion logic as intuition and restored a caveat that some structured regions may still be locally guessable, avoiding over-absolutization.
- Added that the asymmetric encoder-only-on-visible-patches design rests on using a Vision Transformer (contemporaneous, correct), grounding the mechanism rather than presenting it as free-floating.
- Softened 'must show up on both fine-tuning and linear probing' to 'should' and noted the asymmetric split is justified partly on efficiency, keeping the pre-hoc prediction from asserting a guaranteed outcome.
```
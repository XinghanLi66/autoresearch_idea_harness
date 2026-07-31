# Pilot CoT 14 — Andrej Karpathy · first_principles

- researcher: **Andrej Karpathy**  ·  style: 工程实证·偏执debug·打开黑盒·Software2.0

- case: Karpathy 案例 1：被"静默失败"折磨 → 逼出神经网络训练的六步流水线

- route: first_principles  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **211** · assistant(cot) **1137** · whole **1405**

- fact-check changes: 5

## user (setup)

```plaintext
It's around 2019. Training deep nets has become deceptively easy on the surface: libraries advertise five-line snippets that fit a ResNet and 'conquer the world.' Yet in practice, people who ship these snippets often get models that are quietly worse than they should be. The catastrophic failures are not the loud ones (syntax errors, exceptions) but the silent ones: a mislabeled channel, an off-by-one in an autoregressive mask that lets timestep t peek at t+1, a reshape that smears information across the batch dimension, a loss that eats softmaxed probabilities instead of logits. In all these cases the code runs, the loss goes down, nothing throws — and the model just learns worse. You want to lay out a genuinely disciplined recipe for training neural networks that does not depend on 'run it and eyeball the result.' Reason from first principles about what should be true of a correct training pipeline, and derive a methodology from those invariants. Give me the idea before appealing to any experiment.
```

## assistant (cot)

```plaintext
Let me start from the property of the object I'm actually dealing with, not from what the marketing says it is. A neural net pipeline is a leaky abstraction: correctness of syntax does not imply correctness of behavior, and the error surface is logical, not syntactic. So my prior should be adversarial — assume it is silently broken until proven otherwise. That's not pessimism, that's just conditioning on the space of ways it can fail: the loud failures throw exceptions and get caught; the ones that survive are precisely the ones that don't throw. So by construction, the bugs that reach me are the invisible ones. I — Karpathy — want a method that makes those visible by force, before I've burned two GPU-days.

Here's the deductive move. If I can't trust 'loss went down' as a signal (a broken pipeline can also make loss go down), then I need signals whose expected value I can COMPUTE ahead of time, and which a bug would visibly violate. A quantity I can predict before running and can be checked after running is the only trustworthy anchor when there's no external ground truth. That's the invariant-first principle. Let me enumerate what should be true.

One: at initialization, before the net has learned anything, a correctly wired softmax + cross-entropy over n classes should output near-uniform probabilities, so the loss should be close to –log(1/n). This isn't a heuristic; it follows from a well-initialized last layer producing roughly symmetric logits. If the measured init loss disagrees, something in the loss or the final-layer init is likely wrong — no training needed to know that.

Two: a model has capacity that should be able to memorize a handful of examples. So if I take two-to-four samples and train only on them, the achievable loss should go essentially to zero and the predictions should match. This cleanly separates 'idea is wrong' from 'implementation is wrong': if you can't overfit four points, it's almost certainly a bug, and there's no point scaling. Failure to overfit a tiny batch is in tension with 'the model has capacity and the data reaches it.'

Three — the fingerprint move — comes from thinking about what independence should hold. In a correctly batched model, the output for example i is a function of input i alone (barring intentional cross-sample layers like batch norm). This is a statement about the Jacobian: ∂ output_i / ∂ input_j should be zero for j ≠ i. And I have a machine that computes exactly that: backprop. So I construct a fake loss that depends only on output i, run the backward pass, and check the input gradient is nonzero only on row i. Any leak from view/reshape/broadcast or a bad mask — including the autoregressive t-sees-t+1 disaster — shows up as a nonzero gradient where the invariant says zero. This is why it's decisive: leaks often make metrics look BETTER, so no amount of eyeballing loss will catch them, but the gradient probe catches them structurally.

Four, the counterintuitive one: zero the inputs. If the model genuinely uses its input, an input-independent baseline should be strictly worse. If feeding zeros does as well as feeding real data, the data isn't actually flowing into the net — the model is just memorizing label priors. This falsifies the seductive illusion that 'the model is learning.'

Why does this cut against the common practice? Because the field's default is fast-and-furious: change architecture, lr, augmentation, and loss all at once, then tune hard. That approach has no falsifiable signal at any step, so a regression can't be attributed to any single change. My claim is the opposite: advance only when a pre-computable invariant lights up, change one variable at a time, and treat patience and attention to detail — not cleverness — as the qualities that correlate with success. Don't be a hero: reproduce a verified baseline first so the only moving part is your hypothesis.

How do I keep myself honest without fabricating outcomes? I don't assert what will happen; I assert the DECISION RULE. Init loss off by more than a small tolerance → stop, fix loss/init. Can't drive a tiny batch to ~zero → stop, there's likely a bug. Nonzero cross-example gradient → stop, there's a leak. Zero-input baseline not clearly worse → stop, the input isn't wired in. Each gate is a check I can reason about before any big experiment earns the right to run. The caveat: some checks are relaxable when reusing a heavily verified config, or skippable for a trivially simple feedforward net with no masking — but the moment something 'should have worked and didn't,' I drop back into strict serial mode.

**Core idea:** Treat training as a paranoid, serially-gated investigation: at each step establish a quantity you can COMPUTE before running and that a bug would visibly violate (init loss ≈ –log(1/n), overfit-one-batch → ~0, backprop Jacobian nonzero only on the i-th input, zeroed-input baseline strictly worse), and only advance when the invariant holds.

**Non-trivial crux:** The cross-example gradient probe is the linchpin — information leaks tend to make metrics look better, not worse, so they're invisible to loss-watching; only checking the derivative structure that should be zero exposes them — while noting intentional cross-sample layers (e.g., batch norm) are the legitimate exception.
```

## fact-check changes

```plaintext
- Softened over-absolutized 'MUST/PROVABLY/forced by symmetry/full stop/decisive/contradiction' language to 'should/likely/in tension with', since these are diagnostic heuristics, not formal proofs.
- Restored the legitimate exception for the batch-independence invariant by naming batch norm as an example of intentional cross-sample layers (in setup crux and body).
- Reframed setup from 'propose a genuinely novel discipline' to a 'disciplined recipe', since Karpathy's 'Recipe for Training Neural Networks' consolidates known practice rather than claims novelty.
- Changed 'leaks make metrics look BETTER' absolute claim to 'often/tend to' to avoid overstatement.
- Removed the specific claim '–log(1/n)' being 'almost exactly' matched, softened to 'close to', consistent with initialization not being perfectly symmetric.
```
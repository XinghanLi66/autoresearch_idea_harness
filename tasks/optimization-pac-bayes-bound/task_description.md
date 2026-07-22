# Task: PAC-Bayes Generalization Bound Optimization

## Research Question
Design a tighter PAC-Bayes generalization bound by optimizing the bound formulation, prior/posterior parameterization, and KL divergence estimation for stochastic neural networks.

## Background
PAC-Bayes theory provides non-vacuous generalization bounds for stochastic classifiers. Given a prior distribution P over hypotheses (chosen before seeing data) and a posterior Q (learned from data), PAC-Bayes bounds certify that with high probability (1-delta), the true risk of a stochastic classifier sampled from Q is bounded.

The key components of a PAC-Bayes bound are:
- **Empirical risk**: estimated loss of the stochastic predictor on training data
- **KL divergence**: KL(Q||P) measuring complexity of the posterior relative to the prior
- **Bound formula**: how these terms combine to yield the final certificate

Standard bounds include:
- **McAllester/Maurer**: risk + sqrt(KL_term / (2n)) -- simple but loose
- **Catoni/Lambda**: risk/(1-lam/2) + KL_term/(n*lam*(1-lam/2)) -- tighter with tuned lambda
- **Quadratic**: (sqrt(risk + KL_term) + sqrt(KL_term))^2 -- better at low risk

The bound can be further tightened through:
- Optimizing the bound functional form (beyond classical inequalities)
- Better training objectives that minimize the bound directly
- Improved risk certificate evaluation (e.g., PAC-Bayes-kl inversion)
- Data-dependent prior construction
- Tighter KL estimation or alternative divergence measures

## What to Implement
Implement the `BoundOptimizer` class in `custom_pac_bayes.py`. You must implement:
1. `compute_bound(empirical_risk, kl, n, delta)`: The PAC-Bayes bound formula
2. `train_step(model, data, target, device, n_bound, delta)`: Training objective
3. `compute_risk_certificate(model, bound_loader, device, delta, mc_samples)`: Final certificate evaluation

## Interface
- `model(x, sample=True/False)`: stochastic forward pass (sample=True) or posterior mean (sample=False)
- `get_total_kl(model)`: sum of KL divergence across all probabilistic layers
- `inv_kl(q, c)`: binary KL inversion -- find p such that KL(Ber(q)||Ber(p)) = c
- `compute_01_risk(model, loader, device, mc_samples)`: MC estimate of 0-1 risk
- Available losses: F.nll_loss, F.cross_entropy on log_softmax outputs

## Evaluation
The bound optimizer is tested on three settings:
1. **MNIST-FCN**: 4-layer fully connected network (784-600-600-600-10) on MNIST
2. **MNIST-CNN**: 4-layer CNN (2 conv + 2 fc) on MNIST
3. **FashionMNIST-CNN**: Same CNN architecture on FashionMNIST

**Primary metric**: `risk_certificate` (0-1 loss PAC-Bayes bound) -- **lower is better** (tighter bound).

Training uses data-dependent priors: 50% of training data trains a deterministic prior, 50% evaluates the bound.


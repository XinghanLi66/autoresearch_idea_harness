"""Optimization-bilevel scaffold for MLS-Bench.

The fixed driver reproduces the numerical verification and data hyper-cleaning
experiments from Shen and Chen, "On Penalty-based Bilevel Gradient Descent
Method" (ICML 2023 / Mathematical Programming 2025) while exposing only the
method choice and official hyperparameters as editable strategy hooks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_DATA_ROOT = os.environ.get("DATA_ROOT", "/data")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
RHG_ROOT = ROOT / "RHG"
if str(RHG_ROOT) not in sys.path:
    sys.path.insert(0, str(RHG_ROOT))

import hypergrad as hg


# =====================================================================
# FIXED: Benchmark configuration
# =====================================================================
@dataclass(frozen=True)
class ToyProblemConfig:
    x_lower: float = 0.0
    x_upper: float = 3.0
    init_x_lower: float = 0.0
    init_x_upper: float = 3.5
    init_y_lower: float = -5.0
    init_y_upper: float = 8.5
    num_runs: int = 1000
    stationarity_tol: float = 1e-5
    residual_tol: float = 5e-2
    max_steps_per_gamma: int = 20_000


@dataclass(frozen=True)
class HypercleanConfig:
    dataset_name: str = "MNIST"
    dataset_root: str = _DATA_ROOT + "/mnist"
    train_size: int = 5000
    val_size: int = 5000
    test_size: int = 10000
    pollute_rate: float = 0.5


@dataclass(frozen=True)
class ToyStrategy:
    method: str
    gams: tuple[float, ...]
    alpha0: float

    def validate(self) -> None:
        if self.method not in {"v_pbgd", "g_pbgd"}:
            raise ValueError(f"Unsupported toy method: {self.method}")
        if not self.gams:
            raise ValueError("Toy strategy must contain at least one penalty value.")
        if any(gam <= 0.0 for gam in self.gams):
            raise ValueError(f"Penalty values must be positive, got {self.gams}")
        if self.alpha0 <= 0.0:
            raise ValueError("alpha0 must be positive.")


@dataclass(frozen=True)
class HypercleanStrategy:
    method: str
    lrx: float = 0.0
    lry: float = 0.0
    lr_inner: float = 0.0
    gamma_init: float = 0.0
    gamma_max: float = 0.0
    gamma_argmax_step: int = 1
    outer_itr: int = 0
    inner_itr: int = 1
    lr: float = 0.0
    T: int = 0
    K: int = 0
    reg: float = 0.0
    eval_interval: int = 10

    def validate(self) -> None:
        if self.method not in {"v_pbgd", "g_pbgd", "rhg", "t_rhg"}:
            raise ValueError(f"Unsupported hyper-cleaning method: {self.method}")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive.")
        if self.reg < 0.0:
            raise ValueError("reg must be non-negative.")
        if self.method in {"v_pbgd", "g_pbgd"}:
            if self.lrx <= 0.0 or self.lry <= 0.0:
                raise ValueError("lrx and lry must be positive for PBGD variants.")
            if self.outer_itr <= 0:
                raise ValueError("outer_itr must be positive for PBGD variants.")
            if self.gamma_init < 0.0 or self.gamma_max < 0.0:
                raise ValueError("gamma values must be non-negative.")
            if self.gamma_argmax_step <= 0:
                raise ValueError("gamma_argmax_step must be positive.")
            if self.method == "v_pbgd" and (self.lr_inner <= 0.0 or self.inner_itr <= 0):
                raise ValueError("V-PBGD requires positive lr_inner and inner_itr.")
        else:
            if self.lr <= 0.0 or self.lr_inner <= 0.0:
                raise ValueError("RHG/T-RHG require positive lr and lr_inner.")
            if self.outer_itr <= 0 or self.T <= 0 or self.K <= 0:
                raise ValueError("RHG/T-RHG require positive outer_itr, T, and K.")
            if self.K > self.T:
                raise ValueError("K cannot be larger than T.")


@dataclass
class HypercleanEval:
    step: int
    train_loss: float
    val_loss: float
    test_accuracy: float
    f1_score: float
    cleaner_precision: float
    cleaner_recall: float
    aux_value: float
    runtime_sec: float


DEFAULT_TOY = ToyProblemConfig()
DEFAULT_HYPERCLEAN = HypercleanConfig()


class HypercleanSplit:
    def __init__(self, data: torch.Tensor, target: torch.Tensor, polluted: bool = False, rho: float = 0.0):
        data = data.float()
        self.data = data / max(float(data.max().item()), 1.0)
        if polluted:
            self.clean_target = None
            self.dirty_target = target.clone()
            self.clean = torch.zeros(target.shape[0], dtype=torch.float32)
        else:
            self.clean_target = target.clone()
            self.dirty_target = None
            self.clean = torch.ones(target.shape[0], dtype=torch.float32)
        self.polluted = polluted
        self.rho = rho
        self.label_set = set(int(v) for v in target.tolist())

    def pollute(self, rho: float) -> None:
        if self.polluted or self.dirty_target is not None:
            raise ValueError("Split has already been polluted.")
        number = self.data.shape[0]
        number_list = list(range(number))
        random.shuffle(number_list)
        self.dirty_target = self.clean_target.clone()
        for index in number_list[: int(rho * number)]:
            dirty_set = set(self.label_set)
            dirty_set.remove(int(self.clean_target[index].item()))
            # Match the released official implementation exactly.
            self.dirty_target[index] = random.randint(0, len(dirty_set))
            self.clean[index] = 0.0
        self.polluted = True
        self.rho = rho

    def flatten(self) -> None:
        self.data = self.data.view(self.data.shape[0], -1)

    def to_device(self, device: torch.device) -> None:
        self.data = self.data.to(device)
        self.clean = self.clean.to(device)
        if self.clean_target is not None:
            self.clean_target = self.clean_target.to(device)
        if self.dirty_target is not None:
            self.dirty_target = self.dirty_target.to(device)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def sum_squared_norm(parameters) -> torch.Tensor:
    total: torch.Tensor | None = None
    for tensor in parameters:
        term = torch.sum(tensor * tensor)
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return total


def write_json(path: Path, payload: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# =====================================================================
# EDITABLE: select supported methods and paper-style hyperparameters only
# =====================================================================
def get_toy_strategy() -> ToyStrategy:
    return ToyStrategy(
        method="v_pbgd",
        gams=(10.0,),
        alpha0=0.1,
    )


def get_hyperclean_strategy(net: str) -> HypercleanStrategy:
    if net == "linear":
        return HypercleanStrategy(
            method="v_pbgd",
            lrx=0.1,
            lry=0.1,
            lr_inner=0.01,
            gamma_init=0.0,
            gamma_max=0.2,
            gamma_argmax_step=30_000,
            outer_itr=40_000,
            inner_itr=1,
            reg=0.0,
            eval_interval=10,
        )
    if net == "mlp":
        return HypercleanStrategy(
            method="v_pbgd",
            lrx=0.1,
            lry=0.01,
            lr_inner=0.01,
            gamma_init=0.0,
            gamma_max=0.1,
            gamma_argmax_step=10_000,
            outer_itr=80_000,
            inner_itr=1,
            reg=0.0,
            eval_interval=10,
        )
    raise ValueError(f"Unsupported network: {net}")


# =====================================================================
# FIXED: numerical verification setup
# =====================================================================
def toy_f(x: float, y: float) -> float:
    return math.cos(4.0 * y + 2.0) / (1.0 + math.exp(2.0 - 4.0 * x)) + 0.5 * math.log((4.0 * x - 2.0) ** 2 + 1.0)


def toy_df(x: float, y: float) -> tuple[float, float]:
    exp_term = math.exp(2.0 - 4.0 * x)
    return (
        4.0 * exp_term * math.cos(4.0 * y + 2.0) / (1.0 + exp_term ** 2)
        + (16.0 * x - 8.0) / ((4.0 * x - 2.0) ** 2 + 1.0),
        -4.0 * math.sin(4.0 * y + 2.0) / (1.0 + exp_term),
    )


def toy_g(x: float, y: float) -> float:
    return (x + y) ** 2 + x * math.sin(x + y) ** 2


def toy_dg(x: float, y: float) -> tuple[float, float]:
    sin_term = math.sin(x + y)
    cos_term = math.cos(x + y)
    return (
        2.0 * (x + y + 0.5 * sin_term ** 2 + x * sin_term * cos_term),
        2.0 * (x + y + x * sin_term * cos_term),
    )


def toy_gpbgd_penalty_grad(x: float, y: float) -> tuple[float, float]:
    sin_term = math.sin(x + y)
    cos_term = math.cos(x + y)
    gy = 2.0 * (x + y + x * sin_term * cos_term)
    dgy_dx = 2.0 * (1.0 + sin_term * cos_term + x * (cos_term ** 2 - sin_term ** 2))
    dgy_dy = 2.0 * (1.0 + x * (cos_term ** 2 - sin_term ** 2))
    return gy * dgy_dx, gy * dgy_dy


def toy_penalized_gradient(x: float, y: float, method: str, gamma: float) -> tuple[float, float, float, float]:
    upper = toy_f(x, y)
    lower = toy_g(x, y)
    grad_fx, grad_fy = toy_df(x, y)
    if method == "v_pbgd":
        grad_gx, grad_gy = toy_dg(x, y)
    elif method == "g_pbgd":
        grad_gx, grad_gy = toy_gpbgd_penalty_grad(x, y)
    else:
        raise ValueError(f"Unsupported toy method: {method}")
    return upper, lower, grad_fx + gamma * grad_gx, grad_fy + gamma * grad_gy


def project_x(x: float, config: ToyProblemConfig) -> float:
    return min(max(x, config.x_lower), config.x_upper)


def run_toy(seed: int, output_dir: Path, label: str) -> dict[str, float]:
    config = DEFAULT_TOY
    strategy = get_toy_strategy()
    strategy.validate()
    rng = random.Random(seed)
    start_time = time.perf_counter()

    convergence_steps: list[int] = []
    residuals: list[float] = []
    projected_grads: list[float] = []
    objectives: list[float] = []
    successes = 0

    for run_idx in range(config.num_runs):
        x = rng.uniform(config.init_x_lower, config.init_x_upper)
        y = rng.uniform(config.init_y_lower, config.init_y_upper)
        total_steps = 0
        upper_value = float("nan")
        residual = float("inf")
        projected_grad = float("inf")
        success = False

        for gamma in strategy.gams:
            alpha = strategy.alpha0 / gamma
            for _ in range(config.max_steps_per_gamma):
                upper_value, _, grad_x, grad_y = toy_penalized_gradient(x, y, strategy.method, gamma)
                x_next = project_x(x - alpha * grad_x, config)
                y_next = y - alpha * grad_y
                projected_grad = math.hypot((x - x_next) / alpha, (y - y_next) / alpha)
                residual = abs(x_next + y_next)
                x, y = x_next, y_next
                total_steps += 1
                if projected_grad <= config.stationarity_tol:
                    success = True
                    break
            if not success:
                total_steps = len(strategy.gams) * config.max_steps_per_gamma
                break
            if success:
                break

        successes += int(success)
        convergence_steps.append(total_steps)
        residuals.append(residual)
        projected_grads.append(projected_grad)
        objectives.append(upper_value)
        print(
            "TRAIN_METRICS "
            f"run={run_idx} step={total_steps} objective={upper_value:.6f} "
            f"residual={residual:.6f} projected_grad={projected_grad:.6f} success={int(success)}",
            flush=True,
        )

    total_runtime = time.perf_counter() - start_time
    metrics = {
        "convergence_steps": float(sum(convergence_steps) / len(convergence_steps)),
        "median_steps": float(sorted(convergence_steps)[len(convergence_steps) // 2]),
        "final_residual": float(sum(residuals) / len(residuals)),
        "final_projected_grad": float(sum(projected_grads) / len(projected_grads)),
        "success_rate": float(successes / len(convergence_steps)),
        "runtime_sec": float(total_runtime),
        "score": float(sum(convergence_steps) / len(convergence_steps)),
    }
    print(
        "FINAL_METRICS " + " ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in metrics.items()
        ),
        flush=True,
    )
    write_json(output_dir / f"{label}_metrics.json", metrics)
    return metrics


# =====================================================================
# FIXED: data hyper-cleaning setup
# =====================================================================
def resolve_dataset_root(config: HypercleanConfig) -> str:
    preferred = Path(config.dataset_root)
    if preferred.exists():
        return str(preferred)
    for candidate in (Path("/tmp/mnist"), Path("./data/mnist")):
        if candidate.exists():
            return str(candidate)
    return str(preferred)


def load_hyperclean_splits(seed: int, device: torch.device) -> tuple[HypercleanSplit, HypercleanSplit, HypercleanSplit]:
    set_global_seed(seed)
    config = DEFAULT_HYPERCLEAN
    dataset = datasets.MNIST(root=resolve_dataset_root(config), train=True, download=False)
    number_list = list(range(dataset.targets.shape[0]))
    random.shuffle(number_list)

    tr_end = config.train_size
    val_end = tr_end + config.val_size
    test_end = val_end + config.test_size

    train = HypercleanSplit(dataset.data[number_list[:tr_end], :, :], dataset.targets[number_list[:tr_end]])
    val = HypercleanSplit(dataset.data[number_list[tr_end:val_end], :, :], dataset.targets[number_list[tr_end:val_end]])
    test = HypercleanSplit(dataset.data[number_list[val_end:test_end], :, :], dataset.targets[number_list[val_end:test_end]])

    train.pollute(config.pollute_rate)
    train.flatten()
    val.flatten()
    test.flatten()
    train.to_device(device)
    val.to_device(device)
    test.to_device(device)
    return train, val, test


def make_model(net: str, device: torch.device) -> nn.Module:
    if net == "linear":
        return nn.Sequential(nn.Linear(784, 10)).to(device)
    if net == "mlp":
        return nn.Sequential(nn.Linear(784, 300), nn.Sigmoid(), nn.Linear(300, 10)).to(device)
    raise ValueError(f"Unsupported network: {net}")


def compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1, keepdim=True)
    return 100.0 * pred.eq(target.view_as(pred)).sum().item() / len(target)


def compute_cleaner_metrics(x: torch.Tensor, clean_indicator: torch.Tensor, rho: float) -> tuple[float, float, float]:
    x_bi = (x >= 0).float()
    clean = x_bi * clean_indicator
    precision = clean.mean() / (x_bi.mean() + 1e-8)
    recall = clean.mean() / (1.0 - rho + 1e-8)
    f1 = 100.0 * 2.0 * precision * recall / (precision + recall + 1e-8)
    return scalar_to_float(precision), scalar_to_float(recall), scalar_to_float(f1)


def make_eval_record(
    step: int,
    train_loss: torch.Tensor | float,
    val_loss: torch.Tensor | float,
    test_accuracy: float,
    f1_score: float,
    cleaner_precision: float,
    cleaner_recall: float,
    aux_value: torch.Tensor | float,
    runtime_sec: float,
) -> HypercleanEval:
    return HypercleanEval(
        step=step,
        train_loss=scalar_to_float(train_loss),
        val_loss=scalar_to_float(val_loss),
        test_accuracy=float(test_accuracy),
        f1_score=float(f1_score),
        cleaner_precision=float(cleaner_precision),
        cleaner_recall=float(cleaner_recall),
        aux_value=scalar_to_float(aux_value),
        runtime_sec=float(runtime_sec),
    )


def update_best_by_accuracy(best: HypercleanEval | None, current: HypercleanEval) -> HypercleanEval:
    if best is None:
        return current
    if current.test_accuracy > best.test_accuracy + 1e-12:
        return current
    if abs(current.test_accuracy - best.test_accuracy) <= 1e-12 and current.f1_score > best.f1_score:
        return current
    return best


def update_best_by_f1(best: HypercleanEval | None, current: HypercleanEval) -> HypercleanEval:
    if best is None:
        return current
    if current.f1_score > best.f1_score + 1e-12:
        return current
    if abs(current.f1_score - best.f1_score) <= 1e-12 and current.test_accuracy > best.test_accuracy:
        return current
    return best


def summarize_hyperclean_metrics(
    best_accuracy: HypercleanEval,
    best_f1: HypercleanEval,
    total_runtime: float,
) -> dict[str, float]:
    return {
        'test_accuracy': best_accuracy.test_accuracy,
        'f1_score': best_accuracy.f1_score,
        'cleaner_precision': best_accuracy.cleaner_precision,
        'cleaner_recall': best_accuracy.cleaner_recall,
        'best_step': float(best_accuracy.step),
        'runtime_sec': best_accuracy.runtime_sec,
        'total_runtime_sec': total_runtime,
        'best_accuracy_test_accuracy': best_accuracy.test_accuracy,
        'best_accuracy_f1_score': best_accuracy.f1_score,
        'best_accuracy_cleaner_precision': best_accuracy.cleaner_precision,
        'best_accuracy_cleaner_recall': best_accuracy.cleaner_recall,
        'best_accuracy_step': float(best_accuracy.step),
        'best_accuracy_runtime_sec': best_accuracy.runtime_sec,
        'best_f1_test_accuracy': best_f1.test_accuracy,
        'best_f1_f1_score': best_f1.f1_score,
        'best_f1_cleaner_precision': best_f1.cleaner_precision,
        'best_f1_cleaner_recall': best_f1.cleaner_recall,
        'best_f1_step': float(best_f1.step),
        'best_f1_runtime_sec': best_f1.runtime_sec,
        'score': best_accuracy.test_accuracy,
    }


def mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    return mean, 1.96 * std / math.sqrt(len(values))


def aggregate_metric_dicts(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError('Cannot aggregate empty hyper-cleaning records.')
    metrics: dict[str, float] = {'num_runs': float(len(records))}
    for key in records[0]:
        values = [float(record[key]) for record in records if key in record]
        mean, ci95 = mean_ci95(values)
        metrics[f'{key}_mean'] = mean
        metrics[f'{key}_ci95'] = ci95
    if 'test_accuracy_mean' in metrics:
        metrics['test_accuracy'] = metrics['test_accuracy_mean']
        metrics['score'] = metrics['test_accuracy_mean']
    if 'f1_score_mean' in metrics:
        metrics['f1_score'] = metrics['f1_score_mean']
    return metrics


def yforward(params: list[torch.Tensor], data: torch.Tensor, net: str) -> torch.Tensor:
    if net == 'linear':
        weight, bias = params
        return F.linear(data, weight, bias)
    if net == 'mlp':
        w1, b1, w2, b2 = params
        out = torch.sigmoid(F.linear(data, w1, b1))
        return F.linear(out, w2, b2)
    raise ValueError(f'Unsupported network: {net}')


def run_v_pbgd(
    seed: int,
    net_name: str,
    strategy: HypercleanStrategy,
    output_dir: Path,
    label: str,
    device: torch.device,
) -> dict[str, float]:
    train, val, test = load_hyperclean_splits(seed, device)
    model = make_model(net_name, device)
    x = torch.zeros(train.data.shape[0], device=device, requires_grad=True)
    y_opt = torch.optim.SGD(model.parameters(), lr=strategy.lry)
    x_opt = torch.optim.SGD([x], lr=strategy.lrx)

    if strategy.gamma_init > strategy.gamma_max:
        gamma_max = strategy.gamma_init
    else:
        gamma_max = strategy.gamma_max
    gamma = strategy.gamma_init
    gamma_step = (gamma_max - strategy.gamma_init) / strategy.gamma_argmax_step

    net_inner = make_model(net_name, device)
    net_inner.load_state_dict(model.state_dict())
    net_inner.train()
    inner_opt = torch.optim.SGD(net_inner.parameters(), lr=strategy.lr_inner)

    best_accuracy: HypercleanEval | None = None
    best_f1: HypercleanEval | None = None
    start_time = time.perf_counter()

    for step in range(strategy.outer_itr):
        model.train()
        with torch.no_grad():
            sigx = torch.sigmoid(x)

        iter_start = time.perf_counter()
        for _ in range(strategy.inner_itr):
            inner_opt.zero_grad(set_to_none=True)
            logits_inner = net_inner(train.data)
            ce_inner = F.cross_entropy(logits_inner, train.dirty_target, reduction='none')
            inner_loss = (sigx * ce_inner).mean() + strategy.reg * sum_squared_norm(net_inner.parameters())
            inner_loss.backward()
            inner_opt.step()

        y_opt.zero_grad(set_to_none=True)
        x_opt.zero_grad(set_to_none=True)

        logits_val = model(val.data)
        logits_train = model(train.data)
        fy = F.cross_entropy(logits_val, val.clean_target)
        ce_train = F.cross_entropy(logits_train, train.dirty_target, reduction='none')
        gxy = (torch.sigmoid(x) * ce_train).mean() + strategy.reg * sum_squared_norm(model.parameters())
        ce_inner_eval = F.cross_entropy(net_inner(train.data), train.dirty_target, reduction='none').detach()
        vx = (torch.sigmoid(x) * ce_inner_eval).mean() + strategy.reg * sum_squared_norm(net_inner.parameters()).detach()

        lr_decay = min(1.0 / (gamma + 1e-8), 1.0)
        objective = lr_decay * (fy + gamma * (gxy - vx))
        objective.backward()
        x_opt.step()
        y_opt.step()
        gamma = min(gamma_max, gamma + gamma_step)
        iter_time = time.perf_counter() - iter_start

        if step % strategy.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                logits_test = model(test.data)
                test_accuracy = compute_accuracy(logits_test, test.clean_target)
                precision, recall, f1_score = compute_cleaner_metrics(x, train.clean, train.rho)
                current = make_eval_record(
                    step=step,
                    train_loss=gxy,
                    val_loss=fy,
                    test_accuracy=test_accuracy,
                    f1_score=f1_score,
                    cleaner_precision=precision,
                    cleaner_recall=recall,
                    aux_value=gxy - vx,
                    runtime_sec=time.perf_counter() - start_time,
                )
                best_accuracy = update_best_by_accuracy(best_accuracy, current)
                best_f1 = update_best_by_f1(best_f1, current)
                print(
                    'TRAIN_METRICS '
                    f'step={step} train_loss={current.train_loss:.6f} val_loss={current.val_loss:.6f} '
                    f'test_accuracy={current.test_accuracy:.3f} f1_score={current.f1_score:.3f} '
                    f'cleaner_precision={current.cleaner_precision:.6f} cleaner_recall={current.cleaner_recall:.6f} '
                    f'penalty_gap={current.aux_value:.6f} iter_time={iter_time:.6f}',
                    flush=True,
                )

    if best_accuracy is None or best_f1 is None:
        raise RuntimeError('V-PBGD did not emit any evaluation records.')
    total_runtime = time.perf_counter() - start_time
    metrics = summarize_hyperclean_metrics(best_accuracy, best_f1, total_runtime)
    print(
        'FINAL_METRICS ' + ' '.join(
            f'{key}={value:.6f}' if isinstance(value, float) else f'{key}={value}'
            for key, value in metrics.items()
        ),
        flush=True,
    )
    write_json(output_dir / f'{label}_metrics.json', metrics)
    return metrics


def run_g_pbgd(
    seed: int,
    net_name: str,
    strategy: HypercleanStrategy,
    output_dir: Path,
    label: str,
    device: torch.device,
) -> dict[str, float]:
    train, val, test = load_hyperclean_splits(seed, device)
    model = make_model(net_name, device)
    x = torch.zeros(train.data.shape[0], device=device, requires_grad=True)
    y_opt = torch.optim.SGD(model.parameters(), lr=strategy.lry)
    x_opt = torch.optim.SGD([x], lr=strategy.lrx)

    if strategy.gamma_init > strategy.gamma_max:
        gamma_max = strategy.gamma_init
    else:
        gamma_max = strategy.gamma_max
    gamma = strategy.gamma_init
    gamma_step = (gamma_max - strategy.gamma_init) / strategy.gamma_argmax_step

    best_accuracy: HypercleanEval | None = None
    best_f1: HypercleanEval | None = None
    start_time = time.perf_counter()

    for step in range(strategy.outer_itr):
        model.train()
        iter_start = time.perf_counter()
        y_opt.zero_grad(set_to_none=True)
        x_opt.zero_grad(set_to_none=True)

        logits_val = model(val.data)
        logits_train = model(train.data)
        fy = F.cross_entropy(logits_val, val.clean_target)
        ce_train = F.cross_entropy(logits_train, train.dirty_target, reduction='none')
        gxy = (torch.sigmoid(x) * ce_train).mean() + strategy.reg * sum_squared_norm(model.parameters())
        dgdy = torch.autograd.grad(gxy, tuple(model.parameters()), create_graph=True)
        dgdy_norm = torch.sqrt(sum_squared_norm(dgdy) + 1e-12)

        lr_decay = min(1.0 / (gamma + 1e-8), 1.0)
        objective = lr_decay * (fy + 0.5 * gamma * sum_squared_norm(dgdy))
        objective.backward()
        x_opt.step()
        y_opt.step()
        gamma = min(gamma_max, gamma + gamma_step)
        iter_time = time.perf_counter() - iter_start

        if step % strategy.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                logits_test = model(test.data)
                test_accuracy = compute_accuracy(logits_test, test.clean_target)
                precision, recall, f1_score = compute_cleaner_metrics(x, train.clean, train.rho)
                current = make_eval_record(
                    step=step,
                    train_loss=gxy,
                    val_loss=fy,
                    test_accuracy=test_accuracy,
                    f1_score=f1_score,
                    cleaner_precision=precision,
                    cleaner_recall=recall,
                    aux_value=dgdy_norm,
                    runtime_sec=time.perf_counter() - start_time,
                )
                best_accuracy = update_best_by_accuracy(best_accuracy, current)
                best_f1 = update_best_by_f1(best_f1, current)
                print(
                    'TRAIN_METRICS '
                    f'step={step} train_loss={current.train_loss:.6f} val_loss={current.val_loss:.6f} '
                    f'test_accuracy={current.test_accuracy:.3f} f1_score={current.f1_score:.3f} '
                    f'cleaner_precision={current.cleaner_precision:.6f} cleaner_recall={current.cleaner_recall:.6f} '
                    f'grad_penalty={current.aux_value:.6f} iter_time={iter_time:.6f}',
                    flush=True,
                )

    if best_accuracy is None or best_f1 is None:
        raise RuntimeError('G-PBGD did not emit any evaluation records.')
    total_runtime = time.perf_counter() - start_time
    metrics = summarize_hyperclean_metrics(best_accuracy, best_f1, total_runtime)
    print(
        'FINAL_METRICS ' + ' '.join(
            f'{key}={value:.6f}' if isinstance(value, float) else f'{key}={value}'
            for key, value in metrics.items()
        ),
        flush=True,
    )
    write_json(output_dir / f'{label}_metrics.json', metrics)
    return metrics


def run_rhg_family(
    seed: int,
    net_name: str,
    strategy: HypercleanStrategy,
    output_dir: Path,
    label: str,
    device: torch.device,
) -> dict[str, float]:
    train, val, test = load_hyperclean_splits(seed, device)
    x = torch.zeros(train.data.shape[0], device=device, requires_grad=True)
    x_opt = torch.optim.SGD([x], lr=strategy.lr)

    def fp_map(params: list[torch.Tensor], hparams: list[torch.Tensor], dataset: HypercleanSplit = train) -> list[torch.Tensor]:
        x_param = hparams[0]
        logits = yforward(params, dataset.data, net_name)
        loss = (torch.sigmoid(x_param) * F.cross_entropy(logits, dataset.dirty_target, reduction='none')).mean()
        loss = loss + strategy.reg * sum_squared_norm(params)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        return [param - strategy.lr_inner * grad for param, grad in zip(params, grads)]

    def val_loss(params: list[torch.Tensor], hparams: list[torch.Tensor], dataset: HypercleanSplit = val) -> torch.Tensor:
        logits = yforward(params, dataset.data, net_name)
        return F.cross_entropy(logits, dataset.clean_target)

    def fresh_params() -> list[torch.Tensor]:
        model = make_model(net_name, device)
        return [param for param in model.parameters()]

    best_accuracy: HypercleanEval | None = None
    best_f1: HypercleanEval | None = None
    start_time = time.perf_counter()

    for step in range(strategy.outer_itr):
        params = fresh_params()
        new_params = params
        params_history = [new_params]
        step_start = time.perf_counter()
        for inner_step in range(strategy.T):
            new_params = fp_map(new_params, [x])
            if inner_step >= strategy.T - strategy.K:
                params_history.append(new_params)
        x_opt.zero_grad(set_to_none=True)
        hg.reverse(params_history, [x], [fp_map] * strategy.K, val_loss, set_grad=True)
        x_opt.step()
        step_time = time.perf_counter() - step_start

        with torch.no_grad():
            fx = val_loss(params_history[-1], [x], val)
            weighted_train = torch.tensor(float('inf'), device=device)
            best_test_acc = 0.0
            for params_t in params_history:
                logits_train = yforward(params_t, train.data, net_name)
                candidate_g = (torch.sigmoid(x) * F.cross_entropy(logits_train, train.dirty_target, reduction='none')).mean()
                weighted_train = torch.minimum(weighted_train, candidate_g)
                logits_test = yforward(params_t, test.data, net_name)
                best_test_acc = max(best_test_acc, compute_accuracy(logits_test, test.clean_target))
            precision, recall, f1_score = compute_cleaner_metrics(x, train.clean, train.rho)
            current = make_eval_record(
                step=step,
                train_loss=weighted_train,
                val_loss=fx,
                test_accuracy=best_test_acc,
                f1_score=f1_score,
                cleaner_precision=precision,
                cleaner_recall=recall,
                aux_value=weighted_train,
                runtime_sec=time.perf_counter() - start_time,
            )
            best_accuracy = update_best_by_accuracy(best_accuracy, current)
            best_f1 = update_best_by_f1(best_f1, current)
            print(
                'TRAIN_METRICS '
                f'step={step} train_loss={current.train_loss:.6f} val_loss={current.val_loss:.6f} '
                f'test_accuracy={current.test_accuracy:.3f} f1_score={current.f1_score:.3f} '
                f'cleaner_precision={current.cleaner_precision:.6f} cleaner_recall={current.cleaner_recall:.6f} '
                f'inner_objective={current.aux_value:.6f} iter_time={step_time:.6f}',
                flush=True,
            )

    if best_accuracy is None or best_f1 is None:
        raise RuntimeError('RHG family did not emit any evaluation records.')
    total_runtime = time.perf_counter() - start_time
    metrics = summarize_hyperclean_metrics(best_accuracy, best_f1, total_runtime)
    print(
        'FINAL_METRICS ' + ' '.join(
            f'{key}={value:.6f}' if isinstance(value, float) else f'{key}={value}'
            for key, value in metrics.items()
        ),
        flush=True,
    )
    write_json(output_dir / f'{label}_metrics.json', metrics)
    return metrics


def run_hyperclean(seed: int, net_name: str, output_dir: Path, label: str, device: torch.device) -> dict[str, float]:
    strategy = get_hyperclean_strategy(net_name)
    strategy.validate()
    if strategy.method == 'v_pbgd':
        return run_v_pbgd(seed, net_name, strategy, output_dir, label, device)
    if strategy.method == 'g_pbgd':
        return run_g_pbgd(seed, net_name, strategy, output_dir, label, device)
    if strategy.method in {'rhg', 't_rhg'}:
        return run_rhg_family(seed, net_name, strategy, output_dir, label, device)
    raise ValueError(f'Unsupported hyper-cleaning method: {strategy.method}')


def run_hyperclean_repeated(
    seed_base: int,
    num_runs: int,
    net_name: str,
    output_dir: Path,
    label: str,
    device: torch.device,
) -> dict[str, float]:
    records: list[dict[str, float]] = []
    for run_index in range(num_runs):
        run_seed = seed_base + run_index
        run_label = f'{label}_run{run_index:02d}'
        metrics = run_hyperclean(run_seed, net_name, output_dir, run_label, device)
        records.append(metrics)
        print(
            'RUN_METRICS '
            f'run={run_index} seed={run_seed} test_accuracy={metrics["test_accuracy"]:.6f} '
            f'f1_score={metrics["f1_score"]:.6f} best_f1_f1_score={metrics["best_f1_f1_score"]:.6f}',
            flush=True,
        )
    aggregate = aggregate_metric_dicts(records)
    print(
        'FINAL_METRICS ' + ' '.join(
            f'{key}={value:.6f}' if isinstance(value, float) else f'{key}={value}'
            for key, value in aggregate.items()
        ),
        flush=True,
    )
    write_json(output_dir / f'{label}_metrics.json', aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Optimization-bilevel benchmark scaffold')
    parser.add_argument('--mode', choices=['toy', 'hyperclean'], required=True)
    parser.add_argument('--net', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed-base', type=int, default=None, help='Base seed for multi-run paper-style reproduction.')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of independent hyper-cleaning runs to aggregate.')
    parser.add_argument('--label', type=str, default='eval')
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/mlsbench_optimization_bilevel'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.num_runs <= 0:
        raise ValueError('--num-runs must be positive.')
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'toy':
        if args.num_runs != 1:
            raise ValueError('--num-runs is only supported for hyperclean mode.')
        run_seed = args.seed if args.seed_base is None else args.seed_base
        run_toy(run_seed, args.output_dir, args.label)
        return
    run_seed = args.seed if args.seed_base is None else args.seed_base
    if args.num_runs == 1:
        run_hyperclean(run_seed, args.net, args.output_dir, args.label, device)
        return
    run_hyperclean_repeated(run_seed, args.num_runs, args.net, args.output_dir, args.label, device)


if __name__ == '__main__':
    main()

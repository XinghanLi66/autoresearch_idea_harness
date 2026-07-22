"""Training and evaluation script for pde-foundation-icl task.

This script trains a custom in-context operator model jointly on multiple PDE datasets,
then evaluates via autoregressive rollout on a specified dataset.

Usage:
    python custom_train_eval.py --eval-dataset NS2D --seed 42 --output-dir /path/to/output
"""

import sys
import os
import argparse
import errno
import json
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import time
from pathlib import Path
from omegaconf import OmegaConf

# VICON source is on PYTHONPATH
from dataset import all_datasets
from dataset_utils import InfiniteDataLooper
from trainer import Trainer
from train_utils import board_loss, print_error, cross_batch_process
import utils
import dataset as dataset_module


class _CyclingIterableDataset(torch.utils.data.IterableDataset):
    """Compatibility wrapper for iterable datasets without a native ``cycle()`` helper."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        while True:
            yield from iter(self.dataset)


def _cycle_dataset(dataset):
    """Use native ``cycle()`` when available, otherwise wrap the iterable dataset."""
    if hasattr(dataset, "cycle"):
        return dataset.cycle()
    return _CyclingIterableDataset(dataset)


def _normalize_type_batch(type_batch):
    """Convert per-loader type metadata to a plain list for downstream aggregation."""
    if isinstance(type_batch, list):
        return type_batch
    if isinstance(type_batch, tuple):
        return list(type_batch)
    try:
        return list(type_batch)
    except TypeError:
        return [type_batch]


_KNOWN_BAD_H5_BASENAMES = {
    "NavierStokes2D_train_496019_0.38774_32.h5",
}


def _should_exclude_dataset_file(path):
    """Filter out known-bad upstream files and accidental LFS pointer stubs."""
    basename = os.path.basename(path)
    if basename in _KNOWN_BAD_H5_BASENAMES:
        return True
    try:
        return os.path.getsize(path) < 1024
    except OSError:
        return True


def _patch_dataset_file_discovery():
    """Prune broken HDF5 shards before datasets build their file lists."""
    if getattr(dataset_module, "_mlsbench_safe_file_discovery", False):
        return

    original_all_in_one = dataset_module._get_all_in_one_path
    original_nested = dataset_module._get_nested_path

    def _safe_all_in_one(type_cfg, mode):
        return [path for path in original_all_in_one(type_cfg, mode) if not _should_exclude_dataset_file(path)]

    def _safe_nested(type_cfg, mode):
        return [path for path in original_nested(type_cfg, mode) if not _should_exclude_dataset_file(path)]

    dataset_module._get_all_in_one_path = _safe_all_in_one
    dataset_module._get_nested_path = _safe_nested
    dataset_module._mlsbench_safe_file_discovery = True


def _patch_dataset_iteration():
    """Skip corrupt dataset files with explicit errors instead of swallowing them silently."""
    original_iter = dataset_module.datasetBase.__iter__
    if getattr(dataset_module.datasetBase, "_mlsbench_safe_iter", False):
        return

    def _safe_iter(self):
        self._init_rng()
        worker_id, _worker_info = self._get_worker_id_and_info()
        worker_paths = self._get_nested_paths()[worker_id]
        if (self.mode != "test") and self.rng is not None:
            self.rng.shuffle(worker_paths)

        for path in worker_paths:
            try:
                f, data, num_trajs = self._load_dataset_file(path, self.mode)
            except Exception as exc:
                print(f"Skipping corrupt file {self.data_type} {path}: {exc}", flush=True)
                continue

            try:
                if self.limit_trajectories is not None and self.limit_trajectories != -1:
                    chunk_size = min(self.limit_trajectories, num_trajs)
                else:
                    chunk_size = num_trajs

                traj_ids = np.arange(num_trajs, dtype=int)
                if (self.mode != "test") and self.rng is not None:
                    self.rng.shuffle(traj_ids)
                worker_chunk_ids = traj_ids[:chunk_size]

                for idx in worker_chunk_ids:
                    c_mask = torch.tensor(np.array(self.cfg.types[self.data_type].c_mask), dtype=torch.float32)

                    if self.rollout and self.dropped_frames:
                        cur_data, dropped_mask = self._proc_data(data, idx)
                    else:
                        cur_data = self._proc_data(data, idx)

                    if not self.rollout:
                        cur_data, pairs, t_in, t_out, delta_t = dataset_module._augment_data(
                            cur_data,
                            self.rng,
                            self.tc_rng,
                            self.cfg,
                            self.split_scheme,
                            self.cfg.types[self.data_type].pad_mode,
                        )
                        pairs = list(pairs)
                        pairs.append(c_mask)
                        yield self.data_type, pairs, t_in, t_out, delta_t
                    else:
                        if self.dropped_frames:
                            yield (self.data_type, cur_data, dropped_mask, c_mask)
                        else:
                            yield (self.data_type, cur_data, c_mask)
            finally:
                try:
                    f.close()
                except Exception:
                    pass

    dataset_module.datasetBase.__iter__ = _safe_iter
    dataset_module.datasetBase._mlsbench_safe_iter = True


_patch_dataset_file_discovery()
_patch_dataset_iteration()


def get_data_from_looper(looper_per_type, tc_rng, cfg):
    """Compatibility copy of VICON's loader aggregation with tuple-safe type handling."""
    type_list = []
    pairs_list = []
    in_idx_list = []
    out_idx_list = []
    for _, looper in looper_per_type.items():
        tmp_types, tmp_pairs, tmp_in_idx, tmp_out_idx, _ = next(looper)
        type_list.append(_normalize_type_batch(tmp_types))
        pairs_list.append(tmp_pairs)
        in_idx_list.append(tmp_in_idx)
        out_idx_list.append(tmp_out_idx)

    t_types = sum(type_list, [])
    t_pairs = tuple(torch.cat([pair[i] for pair in pairs_list], dim=0) for i in range(len(pairs_list[0])))
    t_in_idx = torch.cat(in_idx_list, dim=0)
    t_out_idx = torch.cat(out_idx_list, dim=0)
    t_pairs = cross_batch_process(t_pairs, tc_rng, cfg.data_cross_batch)
    return type_list, pairs_list, in_idx_list, out_idx_list, t_types, t_pairs, t_in_idx, t_out_idx


def _pid_is_alive(pid):
    """Return True when ``pid`` is still running."""
    if pid <= 0:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if not proc_stat.exists():
        return False
    try:
        stat_fields = proc_stat.read_text().split()
    except OSError:
        return False
    if len(stat_fields) < 22:
        return False
    if stat_fields[2] == "Z":
        return False
    return True


def _process_start_time(pid):
    """Return the kernel start time for ``pid`` or ``None`` when unavailable."""
    proc_stat = Path(f"/proc/{pid}/stat")
    if not proc_stat.exists():
        return None
    try:
        stat_fields = proc_stat.read_text().split()
    except OSError:
        return None
    if len(stat_fields) < 22:
        return None
    return stat_fields[21]


def _read_lock_owner(lock_path):
    """Return ``(pid, start_time)`` from the lock file, tolerating legacy integer locks."""
    try:
        raw = Path(lock_path).read_text().strip()
    except OSError:
        return 0, None
    if not raw:
        return 0, None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            return int(raw), None
        except ValueError:
            return 0, None
    if isinstance(payload, int):
        return int(payload), None
    if not isinstance(payload, dict):
        return 0, None
    try:
        pid = int(payload.get("pid", 0) or 0)
    except (TypeError, ValueError):
        pid = 0
    start_time = payload.get("start_time")
    if start_time is not None:
        start_time = str(start_time)
    return pid, start_time


def _load_trainer_from_checkpoint(cfg, model, ckpt_path):
    """Load a shared checkpoint and rebuild the trainer wrapper."""
    print(f"Loading existing checkpoint: {ckpt_path}", flush=True)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return Trainer(
        model, cfg.model, cfg.opt, cfg.loss,
        trainable_mode=cfg.trainable_mode, amp=cfg.amp, multi_gpu=cfg.multi_gpu,
    )


def _train_or_wait_for_shared_checkpoint(cfg, model, ckpt_path, lock_path):
    """Train exactly once per seed/output-dir, other workers wait and reuse the checkpoint."""
    wait_logged = False

    while True:
        if os.path.exists(ckpt_path):
            return _load_trainer_from_checkpoint(cfg, model, ckpt_path)

        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

            lock_owner, lock_start_time = _read_lock_owner(lock_path)
            current_start_time = _process_start_time(lock_owner) if lock_owner else None
            if (
                lock_owner
                and (
                    not _pid_is_alive(lock_owner)
                    or (lock_start_time is not None and current_start_time != lock_start_time)
                )
            ):
                print(f"Removing stale training lock from PID {lock_owner}", flush=True)
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue

            if not wait_logged:
                print("Waiting for shared training checkpoint to be produced...", flush=True)
                wait_logged = True
            time.sleep(10)
            continue

        with os.fdopen(fd, "w") as lock_f:
            json.dump({"pid": os.getpid(), "start_time": _process_start_time(os.getpid())}, lock_f)

        try:
            if os.path.exists(ckpt_path):
                return _load_trainer_from_checkpoint(cfg, model, ckpt_path)

            print(f"Acquired training lock: {lock_path}", flush=True)
            trainer = train(cfg, model)
            save_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            torch.save(save_model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}", flush=True)
            return trainer
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


def make_rollout_cfg():
    """Create a minimal rollout config so dataset init doesn't crash."""
    return OmegaConf.create({
        "dropped": False,
        "strategy": "fixed_single_step",
        "gt_ref_steps": 10,
        "demo_num": 9,
        "max_stride": 1,
        "batch_size": 1,
        "save": 0,
        "error": {"scheme": "scale_by_std", "eps": 1e-4},
    })


def make_config(args):
    """Build OmegaConf config matching VICON's expected structure."""
    cfg = OmegaConf.create({
        "train_seed": args.seed,
        "test_seed": args.seed + 42,
        "board": False,
        "plot": False,
        "deterministic": True,
        "amp": 0,
        "multi_gpu": 0,
        "trainable_mode": "all",
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "loss_freq": 500,
        "save_freq": 999999,
        "plot_freq": 999999,
        "time_freq": 1000,
        "time_warm": 100,
        "data_cross_batch": "no",
        "dataset_workers": args.num_workers,
        "restore_dir": None,
        "dump_dir": args.output_dir,
        "project": "train",
        "datasets": {
            "num_samples_max": -1,
            "delta_t_range": [1, 5],
            "usevort": True,
            "usegrid": False,
            "demo_num": 10,
            "split_scheme": "split",
            "crop_len_in": 64,
            "crop_len_out": 16,
            "monotonic": True,
            "rotate_flip": "none",
            "train_batch_size": args.batch_size,
            "test_batch_size": args.batch_size,
            "types": {
                "COMPRESSIBLE2D": {
                    "folder": os.path.join(args.data_root, "2D_Train_Rand/"),
                    "pad_mode": "periodic",
                    "train_batch_size": args.batch_size,
                    "test_batch_size": args.batch_size,
                    "c_mask": [1, 1, 1, 1, 0, 0, 0],
                },
                "EULER2D": {
                    "folder": os.path.join(args.data_root, "2D_Train_Turb/"),
                    "pad_mode": "periodic",
                    "train_batch_size": args.batch_size,
                    "test_batch_size": args.batch_size,
                    "c_mask": [1, 1, 1, 1, 0, 0, 0],
                },
                "NS2D": {
                    "folder": os.path.join(args.data_root, "NavierStokes-2D-conditoned/"),
                    "pad_mode": "padding",
                    "train_batch_size": args.batch_size,
                    "test_batch_size": args.batch_size,
                    "c_mask": [0, 1, 1, 0, 0, 1, 0],
                },
            },
        },
        "model": {
            "transformer": {
                "dim_channel": 7,
                "dim_token": args.dim_token,
                "nhead": args.nhead,
                "dim_feedforward": args.dim_feedforward,
                "num_layers": args.num_layers,
                "dropout": 0.0,
                "norm_first": False,
            },
            "preprocess": {"scheme": "scale_to_eps", "eps": 1e-4},
            "type": "nocrop",
            "demo_num": 10,
            "patch_num_in": 8,
            "patch_num_out": 8,
            "patch_resolution": 16,
            "use_patch_pos_encoding": True,
            "use_func_pos_encoding": True,
        },
        "opt": {
            "peak_lr": args.lr,
            "end_lr": 1e-7,
            "warmup_steps": args.warmup_steps,
            "decay_steps": args.epochs * args.steps_per_epoch,
            "gnorm_clip": 1.0,
            "weight_decay": 1e-4,
            "gradient_accumulation_steps": 1,
        },
        "loss": {"min_ex": 5, "scale": "bc"},
    })
    return cfg


def train(cfg, model):
    """Train the model jointly on all datasets."""
    utils.set_seed(cfg.train_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.train_seed)
    train_worker_count = max(0, int(cfg.dataset_workers))
    eval_worker_count = 0 if train_worker_count == 0 else 1
    use_pin_memory = torch.cuda.is_available()

    # Create trainer
    trainer = Trainer(
        model, cfg.model, cfg.opt, cfg.loss,
        trainable_mode=cfg.trainable_mode, amp=cfg.amp, multi_gpu=cfg.multi_gpu,
    )

    # Create datasets
    rollout_cfg = make_rollout_cfg()
    train_datasets = all_datasets(cfg.datasets, cfg.dataset_workers, cfg.train_seed, "train", rollout_cfg)
    test_datasets = all_datasets(cfg.datasets, cfg.dataset_workers, cfg.test_seed, "test", rollout_cfg)

    train_loaders = {
        k: torch.utils.data.DataLoader(
            _cycle_dataset(v), batch_size=cfg.datasets.types[k].train_batch_size,
            num_workers=train_worker_count, pin_memory=use_pin_memory,
        )
        for k, v in train_datasets.items()
    }
    test_loaders = {
        k: torch.utils.data.DataLoader(
            _cycle_dataset(v), batch_size=cfg.datasets.types[k].test_batch_size,
            num_workers=eval_worker_count, pin_memory=use_pin_memory,
        )
        for k, v in test_datasets.items()
    }

    train_loopers = {k: InfiniteDataLooper(v) for k, v in train_loaders.items()}
    test_loopers = {k: InfiniteDataLooper(v) for k, v in test_loaders.items()}

    total_steps = cfg.epochs * cfg.steps_per_epoch
    print(f"Total training steps: {total_steps}", flush=True)

    for step in range(total_steps + 1):
        data = get_data_from_looper(train_loopers, tc_rng, cfg)
        train_type_list, train_pairs_list = data[0], data[1]
        train_types, train_pairs = data[4], data[5]

        trainer.model.eval()

        # Log metrics periodically
        if (trainer.train_step % cfg.loss_freq == 0) or \
           (trainer.train_step % (cfg.loss_freq // 10) == 0 and trainer.train_step <= cfg.loss_freq):
            with torch.inference_mode():
                train_loss = trainer.get_loss(train_pairs)
                print(f"TRAIN_METRICS step={trainer.train_step} train_loss={train_loss:.6f}", flush=True)

            test_data = get_data_from_looper(test_loopers, tc_rng, cfg)
            test_pairs = test_data[5]
            test_loss = trainer.get_loss(test_pairs)
            print(f"TRAIN_METRICS step={trainer.train_step} test_loss={test_loss:.6f}", flush=True)

        trainer.model.train()
        trainer.iter(train_pairs)

    return trainer


def evaluate_rollout(cfg, model, eval_dataset, trainer):
    """Evaluate the model via autoregressive rollout on a specific dataset."""
    print(f"\n=== Evaluating rollout on {eval_dataset} ===", flush=True)

    utils.set_seed(cfg.test_seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(cfg.test_seed)
    eval_worker_count = 0 if int(cfg.dataset_workers) <= 0 else 1
    use_pin_memory = torch.cuda.is_available()

    model.eval()

    # Load test data for the specific dataset
    rollout_cfg = make_rollout_cfg()
    test_datasets = all_datasets(cfg.datasets, eval_worker_count, cfg.test_seed, "test", rollout_cfg)

    if eval_dataset not in test_datasets:
        print(f"ERROR: Dataset {eval_dataset} not found. Available: {list(test_datasets.keys())}")
        return

    test_ds = test_datasets[eval_dataset]
    test_loader = torch.utils.data.DataLoader(
        _cycle_dataset(test_ds), batch_size=cfg.datasets.types[eval_dataset].test_batch_size,
        num_workers=eval_worker_count, pin_memory=use_pin_memory,
    )
    test_looper = InfiniteDataLooper(test_loader)

    # Evaluate over multiple batches
    n_eval_batches = 20
    all_errors = []

    with torch.inference_mode():
        for batch_idx in range(n_eval_batches):
            try:
                data = next(test_looper)
                pde_types, pairs = data[0], data[1]

                # Get predictions and compute errors
                error_mean_dict, error_std_dict = trainer.get_error(pde_types, pairs)

                for key in error_mean_dict:
                    err = error_mean_dict[key]  # shape: (timesteps, channels)
                    # Average over channels, take last timestep and all-average
                    last_step = err[-1].mean()
                    all_avg = err.mean()
                    all_errors.append({
                        "last_step": last_step,
                        "all_avg": all_avg,
                    })
            except Exception as e:
                print(f"  Batch {batch_idx} error: {e}", flush=True)
                continue

    if all_errors:
        last_step_mean = np.mean([e["last_step"] for e in all_errors])
        all_avg_mean = np.mean([e["all_avg"] for e in all_errors])
        last_step_std = np.std([e["last_step"] for e in all_errors])
        all_avg_std = np.std([e["all_avg"] for e in all_errors])

        print(f"\n  {eval_dataset} — Last-step rel err: {last_step_mean:.6f} +/- {last_step_std:.6f}", flush=True)
        print(f"  {eval_dataset} — All-avg rel err: {all_avg_mean:.6f} +/- {all_avg_std:.6f}", flush=True)
        print(f"TEST_METRICS last_step_rel_err={last_step_mean:.6f} all_avg_rel_err={all_avg_mean:.6f}", flush=True)
    else:
        print(f"  WARNING: No valid evaluation batches for {eval_dataset}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="PDE Foundation ICL Training + Evaluation")
    parser.add_argument("--eval-dataset", type=str, required=True,
                        choices=["NS2D", "COMPRESSIBLE2D", "EULER2D"],
                        help="Dataset to evaluate on after training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="/tmp/pde_foundation_icl")
    parser.add_argument("--data-root", type=str, default="/data/icon-data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--dim-token", type=int, default=1024)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    cfg = make_config(args)

    # Load custom model
    from custom_model import CustomModel
    model = CustomModel(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Custom model parameters: {n_params:,}", flush=True)

    # Check for existing checkpoint (shared across parallel evaluations)
    ckpt_dir = os.path.join(args.output_dir, f"_shared_ckpt_s{args.seed}")
    ckpt_path = os.path.join(ckpt_dir, "model_final.pt")
    lock_path = os.path.join(ckpt_dir, "training.lock")
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = _train_or_wait_for_shared_checkpoint(cfg, model, ckpt_path, lock_path)

    # Evaluate on specified dataset
    evaluate_rollout(cfg, model, args.eval_dataset, trainer)


if __name__ == "__main__":
    main()

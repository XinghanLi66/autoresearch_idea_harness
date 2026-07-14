from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from .io import project_root, short_text


MLS_SUBTASKS: dict[str, list[str]] = {
    "dl_lr_schedule": ["resnet20-cifar10", "resnet56-cifar100", "mobilenetv2-fmnist"],
    "dl_activation_function": ["resnet20-cifar10", "vgg16bn-cifar100", "mobilenetv2-fmnist"],
    "cv_data_augmentation": ["resnet20-cifar10", "resnet56-cifar100", "mobilenetv2-fmnist"],
    "dl_weight_initialization": ["resnet56-cifar100", "vgg16bn-cifar100", "mobilenetv2-fmnist"],
    "cv_classification_loss": ["resnet56-cifar100", "vgg16bn-cifar100", "mobilenetv2-fmnist"],
    "cv_sample_weighting": ["resnet32-cifar10lt", "resnet32-cifar100lt", "vgg16bn-cifar100lt"],
    "cv_pooling_aggregation": ["resnet56-cifar100", "vgg16bn-cifar100", "mobilenetv2-fmnist"],
    "cv_multitask_loss": ["resnet20-cifar100mt", "resnet56-cifar100mt", "vgg16bn-cifar100mt"],
    "dl_regularization": ["resnet56-cifar100", "vgg16bn-cifar100", "mobilenetv2-fmnist"],
    "dl_residual_connection": ["resnet20-cifar10", "resnet56-cifar100", "resnet110-cifar100"],
}


class AbstractBenchmarkTask(ABC):
    name: str = ""
    task_type: str = "unknown"

    @property
    @abstractmethod
    def metric_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def lower_is_better(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def pass_threshold(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def baseline_metric(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def task_context(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def setup_workspace(self, workspace: Path) -> None:
        raise NotImplementedError

    def passed(self, metric: float) -> bool:
        target = self.pass_metric()
        return metric <= target if self.lower_is_better else metric >= target

    def pass_metric(self) -> float:
        baseline = self.baseline_metric()
        return baseline - self.pass_threshold if self.lower_is_better else baseline + self.pass_threshold

    def improvement(self, metric: float) -> float:
        if self.lower_is_better:
            return self.baseline_metric() - metric
        return metric - self.baseline_metric()

    def eval_papers(self) -> list[str]:
        return []

    def task_packet(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": self.name,
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "lower_is_better": self.lower_is_better,
            "baseline_metric": self.baseline_metric(),
            "pass_threshold": self.pass_threshold,
            "pass_metric": self.pass_metric(),
            "task_context": self.task_context(),
            "frontline_papers": load_frontline_papers(self.name, cfg),
            "worker_constraints": worker_constraints(self),
        }


_WORKER_CLAUDE_MD = """\
# Worker Instructions

CRITICAL: When running `bash run.sh result.json`, use the Bash tool directly with
`timeout=7200000` and wait synchronously for it to complete.

Do NOT use `run_in_background`, the Task tool, or set `run_in_background: true`.
The training takes up to 2 hours and the session must stay alive until result.json
is written. Backgrounding kills the training process when the session ends.
"""


def _write_worker_claude_md(workspace: Path) -> None:
    claude_dir = workspace / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "CLAUDE.md").write_text(_WORKER_CLAUDE_MD)


class MLSBenchTask(AbstractBenchmarkTask):
    task_type = "mls"
    mls_task_name: ClassVar[str] = ""
    pkg_name: ClassVar[str] = ""
    reference_baseline: ClassVar[str] = ""
    active_subtasks: ClassVar[list[str]] = []
    all_subtask_specs: ClassVar[dict[str, dict[str, Any]]] = {}
    _baseline_values: ClassVar[dict[str, float]] = {}
    edit_file: ClassVar[str] = ""
    edit_start: ClassVar[int] = 0
    edit_end: ClassVar[int] = 0
    edit_template_rel: ClassVar[str] = ""
    default_pass_threshold: ClassVar[float] = 0.0

    def __init__(self, cfg: dict[str, Any], subtask: str | None = None) -> None:
        self.cfg = cfg
        self._load_mls_metadata()
        if subtask:
            if subtask not in self.all_subtask_specs:
                raise ValueError(f"Unknown subtask {subtask!r} for {self.name}")
            self.active_subtasks = [subtask]

    def _load_mls_metadata(self) -> None:
        config_path = self.mls_root / "tasks" / self.mls_task_name / "config.json"
        if not config_path.exists():
            return
        try:
            raw = json.loads(config_path.read_text())
        except Exception:
            return
        files = raw.get("files") or []
        if files:
            file_row = files[0]
            edit = (file_row.get("edit") or [{}])[0]
            if not self.edit_file:
                self.edit_file = str(file_row.get("filename") or "")
            if not self.edit_start:
                self.edit_start = int(edit.get("start") or 0)
            if not self.edit_end:
                self.edit_end = int(edit.get("end") or 0)
        if not self.edit_template_rel:
            self.edit_template_rel = f"tasks/{self.mls_task_name}/edits/custom_template.py"
        if not self.all_subtask_specs:
            self.all_subtask_specs = {
                row["label"]: _subtask_spec_from_test_cmd(row)
                for row in raw.get("test_cmds") or []
                if row.get("label")
            }
        if not self.active_subtasks and self.all_subtask_specs:
            visible = [k for k, v in self.all_subtask_specs.items() if not v.get("hidden")]
            self.active_subtasks = [visible[0] if visible else next(iter(self.all_subtask_specs))]

    @property
    def mls_root(self) -> Path:
        return Path(self.cfg.get("mls_bench_root") or os.environ.get("MLS_BENCH_ROOT") or project_root() / "external" / "MLS-Bench")

    @property
    def python_bin(self) -> str:
        return str(self.cfg.get("benchmark_python") or os.environ.get("MLE_PYTHON") or sys.executable)

    @property
    def edit_template(self) -> Path:
        return self.mls_root / self.edit_template_rel

    @property
    def data_root(self) -> Path:
        return self.mls_root / "vendor" / "data"

    def baseline_metric(self) -> float:
        label = self.active_subtasks[0]
        override = self._threshold_override(label)
        if override and "baseline_metric" in override:
            return float(override["baseline_metric"])
        return float(self._baseline_values.get(label, 0.0))

    @property
    def pass_threshold(self) -> float:
        return self._configured_pass_threshold(self.default_pass_threshold)

    def _configured_pass_threshold(self, default: float) -> float:
        label = self.active_subtasks[0]
        override = self._threshold_override(label)
        if not override:
            return default
        if "pass_threshold" in override:
            return float(override["pass_threshold"])
        if "pass_metric" in override:
            if self.lower_is_better:
                return self.baseline_metric() - float(override["pass_metric"])
            return float(override["pass_metric"]) - self.baseline_metric()
        return default

    def _threshold_override(self, label: str) -> dict[str, Any] | None:
        v23 = self.cfg.get("v2_3", {})
        by_task = v23.get("task_thresholds", {}).get(self.name, {})
        if label in by_task:
            return by_task[label]
        if "default" in by_task:
            return by_task["default"]
        legacy = self.cfg.get("mls_task_thresholds", {}).get(self.name, {})
        if label in legacy:
            return legacy[label]
        return None

    def eval_papers(self) -> list[str]:
        return frontline_ids_from_asset("mls_tasks", self.name)

    def _read_editable_region(self) -> str:
        lines = self.edit_template.read_text().splitlines()
        return "\n".join(lines[self.edit_start - 1:self.edit_end]) + "\n"

    def setup_workspace(self, workspace: Path) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        edit_path = workspace / self.edit_file
        edit_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.edit_template, edit_path)
        (workspace / "editable_region.py").write_text(self._read_editable_region())
        shutil.copy2(project_root() / "scripts" / "run_mls_eval.py", workspace / "run_mls_eval.py")
        subtasks_json = json.dumps(self.active_subtasks)
        subtask_specs_json = json.dumps([
            {"label": label, **self.all_subtask_specs[label]}
            for label in self.active_subtasks
        ])
        run_sh = (
            "#!/bin/bash\nset -e\n"
            "RESULT=\"${1:-result.json}\"\n"
            f"{self.python_bin} run_mls_eval.py \\\n"
            "  --editable editable_region.py \\\n"
            f"  --template \"{self.edit_template}\" \\\n"
            f"  --edit-start {self.edit_start} \\\n"
            f"  --edit-end {self.edit_end} \\\n"
            f"  --subtasks '{subtasks_json}' \\\n"
            f"  --subtask-specs '{subtask_specs_json}' \\\n"
            f"  --script-name \"{Path(self.edit_file).name}\" \\\n"
            f"  --data-root \"{self.data_root}\" \\\n"
            f"  --python \"{self.python_bin}\" \\\n"
            "  --gpu \"${CUDA_VISIBLE_DEVICES:-0}\" \\\n"
            "  --out-json \"$RESULT\"\n"
        )
        (workspace / "run.sh").write_text(run_sh)
        os.chmod(workspace / "run.sh", 0o755)
        _write_worker_claude_md(workspace)

    def task_context(self) -> str:
        desc_path = self.mls_root / "tasks" / self.mls_task_name / "task_description.md"
        task_desc = desc_path.read_text() if desc_path.exists() else self.mls_task_name
        label = self.active_subtasks[0]
        baseline = self.baseline_metric()
        pass_metric = self.pass_metric()
        return (
            f"{task_desc}\n\n"
            f"**Reference baseline ({self.reference_baseline}, {label}):** "
            f"{self.metric_name} = {baseline:.2f}% "
            f"(pass if >= {pass_metric:.2f}%)\n\n"
            "## Editable Region\n\n"
            "Modify only `editable_region.py`; preserve the exact function signature.\n\n"
            f"Current content:\n```python\n{self._read_editable_region()}```\n\n"
            "Run evaluation with:\n```bash\nbash run.sh result.json\n```\n"
            "This writes signed `result.json` with `val_metric`."
        )


class DlLrScheduleTask(MLSBenchTask):
    name = "dl_lr_schedule"
    mls_task_name = "dl-lr-schedule"
    pkg_name = "pytorch-vision"
    reference_baseline = "warmup_cosine"
    active_subtasks = ["resnet20-cifar10"]
    all_subtask_specs = {
        "resnet20-cifar10": {"arch": "resnet20", "dataset": "cifar10", "epochs": 200},
        "resnet56-cifar100": {"arch": "resnet56", "dataset": "cifar100", "epochs": 200},
        "mobilenetv2-fmnist": {"arch": "mobilenetv2", "dataset": "fmnist", "epochs": 200},
    }
    _baseline_values = {
        "resnet20-cifar10": 92.71,
        "resnet56-cifar100": 72.43,
        "mobilenetv2-fmnist": 94.83,
    }
    edit_file = "pytorch-vision/custom_schedule.py"
    edit_start = 246
    edit_end = 269
    edit_template_rel = "tasks/dl-lr-schedule/edits/custom_template.py"
    default_pass_threshold = 0.48

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def pass_threshold(self) -> float:
        return self._configured_pass_threshold(self.default_pass_threshold)


class DlActivationFunctionTask(MLSBenchTask):
    name = "dl_activation_function"
    mls_task_name = "dl-activation-function"
    pkg_name = "pytorch-vision"
    reference_baseline = "gelu"
    active_subtasks = ["resnet20-cifar10"]
    all_subtask_specs = {
        "resnet20-cifar10": {"arch": "resnet20", "dataset": "cifar10", "epochs": 200},
        "vgg16bn-cifar100": {"arch": "vgg16bn", "dataset": "cifar100", "epochs": 200},
        "mobilenetv2-fmnist": {"arch": "mobilenetv2", "dataset": "fmnist", "epochs": 200},
    }
    _baseline_values = {
        "resnet20-cifar10": 92.97,
        "vgg16bn-cifar100": 71.38,
        "mobilenetv2-fmnist": 94.75,
    }
    edit_file = "pytorch-vision/custom_activation.py"
    edit_start = 32
    edit_end = 49
    edit_template_rel = "tasks/dl-activation-function/edits/custom_template.py"
    default_pass_threshold = 0.38

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def pass_threshold(self) -> float:
        return self._configured_pass_threshold(self.default_pass_threshold)


class CvDataAugmentationTask(MLSBenchTask):
    name = "cv_data_augmentation"
    mls_task_name = "cv-data-augmentation"
    pkg_name = "pytorch-vision"
    reference_baseline = "cutout"
    active_subtasks = ["resnet20-cifar10"]
    all_subtask_specs = {
        "resnet20-cifar10": {"arch": "resnet20", "dataset": "cifar10", "epochs": 200},
        "resnet56-cifar100": {"arch": "resnet56", "dataset": "cifar100", "epochs": 200},
        "mobilenetv2-fmnist": {"arch": "mobilenetv2", "dataset": "fmnist", "epochs": 200},
    }
    _baseline_values = {
        "resnet20-cifar10": 93.67,
        "resnet56-cifar100": 74.54,
        "mobilenetv2-fmnist": 94.72,
    }
    edit_file = "pytorch-vision/custom_augment.py"
    edit_start = 246
    edit_end = 275
    edit_template_rel = "tasks/cv-data-augmentation/edits/custom_template.py"
    default_pass_threshold = 0.09

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def pass_threshold(self) -> float:
        return self._configured_pass_threshold(self.default_pass_threshold)


class DlWeightInitializationTask(MLSBenchTask):
    name = "dl_weight_initialization"
    mls_task_name = "dl-weight-initialization"
    pkg_name = "pytorch-vision"
    reference_baseline = "worker_only_calibrated"
    active_subtasks = ["resnet56-cifar100"]

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False


class CvClassificationLossTask(MLSBenchTask):
    name = "cv_classification_loss"
    mls_task_name = "cv-classification-loss"
    pkg_name = "pytorch-vision"
    reference_baseline = "worker_only_calibrated"
    active_subtasks = ["resnet56-cifar100"]

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False


class CvSampleWeightingTask(MLSBenchTask):
    name = "cv_sample_weighting"
    mls_task_name = "cv-sample-weighting"
    pkg_name = "pytorch-vision"
    reference_baseline = "worker_only_calibrated"
    active_subtasks = ["resnet32-cifar10lt"]

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False


class CvPoolingAggregationTask(MLSBenchTask):
    name = "cv_pooling_aggregation"
    mls_task_name = "cv-pooling-aggregation"
    pkg_name = "pytorch-vision"
    reference_baseline = "worker_only_calibrated"
    active_subtasks = ["resnet56-cifar100"]

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False


class CvMultitaskLossTask(MLSBenchTask):
    name = "cv_multitask_loss"
    mls_task_name = "cv-multitask-loss"
    pkg_name = "pytorch-vision"
    reference_baseline = "worker_only_calibrated"
    active_subtasks = ["resnet20-cifar100mt"]

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False


class DlRegularizationTask(MLSBenchTask):
    name = "dl_regularization"
    mls_task_name = "dl-regularization"
    pkg_name = "pytorch-vision"
    reference_baseline = "worker_only_calibrated"
    active_subtasks = ["resnet56-cifar100"]

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False


class DlResidualConnectionTask(MLSBenchTask):
    name = "dl_residual_connection"
    mls_task_name = "dl-residual-connection"
    pkg_name = "pytorch-vision"
    reference_baseline = "worker_only_calibrated"
    active_subtasks = ["resnet20-cifar10"]

    @property
    def metric_name(self) -> str:
        return "test_acc"

    @property
    def lower_is_better(self) -> bool:
        return False


class MleBenchTask(AbstractBenchmarkTask):
    task_type = "mle"
    competition_id: ClassVar[str] = ""
    _baseline_score: ClassVar[float] = 0.0

    def __init__(self, cfg: dict[str, Any], subtask: str | None = None) -> None:
        self.cfg = cfg

    @property
    def metric_name(self) -> str:
        return "mle_norm_score"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def mle_data_dir(self) -> Path:
        return Path(self.cfg.get("mle_data_dir") or os.environ.get("MLE_DATA_DIR") or project_root() / "external" / "mlebench_data")

    @property
    def python_bin(self) -> str:
        return str(self.cfg.get("benchmark_python") or os.environ.get("MLE_PYTHON") or sys.executable)

    @property
    def prepared_public(self) -> Path:
        return self.mle_data_dir / self.competition_id / "prepared" / "public"

    @property
    def baseline_template(self) -> Path:
        return project_root() / "assets" / "mle_tasks" / self.name / "baseline.py"

    def baseline_metric(self) -> float:
        return self._baseline_score

    def eval_papers(self) -> list[str]:
        return frontline_ids_from_asset("mle_tasks", self.name)

    def setup_workspace(self, workspace: Path) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        if self.baseline_template.exists():
            shutil.copy2(self.baseline_template, workspace / "baseline.py")
        else:
            (workspace / "baseline.py").write_text(
                "from pathlib import Path\n\n"
                "data_dir = Path('data')\n"
                "raise NotImplementedError('implement proposal and write submission.csv')\n"
            )
        data_link = workspace / "data"
        if not data_link.exists():
            if self.prepared_public.exists():
                data_link.symlink_to(self.prepared_public)
            else:
                data_link.mkdir()
        shutil.copy2(project_root() / "scripts" / "mle_grade.py", workspace / "mle_grade.py")
        run_sh = (
            "#!/bin/bash\nset -e\n"
            "RESULT=\"${1:-result.json}\"\n"
            f"{self.python_bin} baseline.py\n"
            f"{self.python_bin} mle_grade.py \\\n"
            f"  --competition \"{self.competition_id}\" \\\n"
            "  --submission submission.csv \\\n"
            f"  --data-dir \"{self.mle_data_dir}\" \\\n"
            "  --out-json \"$RESULT\"\n"
        )
        (workspace / "run.sh").write_text(run_sh)
        (workspace / "run.sh").chmod(
            stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        )
        _write_worker_claude_md(workspace)

    def task_context(self) -> str:
        desc_path = self.prepared_public / "description.md"
        desc = desc_path.read_text() if desc_path.exists() else f"Competition: {self.competition_id}"
        sample_sub = self.prepared_public / "sample_submission.csv"
        sample_lines = ""
        if sample_sub.exists():
            lines = sample_sub.read_text().splitlines()[:6]
            sample_lines = "\n\n## Required submission.csv format\n```\n" + "\n".join(lines) + "\n```"
        baseline_code = self.baseline_template.read_text() if self.baseline_template.exists() else ""
        return (
            f"{desc}{sample_lines}\n\n"
            f"**Baseline normalized mlebench score: {self._baseline_score:.4f}** "
            f"(pass if >= {self._baseline_score + self.pass_threshold:.4f})\n\n"
            "All competition data is in `./data/`. Run `bash run.sh result.json`.\n\n"
            f"## Baseline starting point\n```python\n{baseline_code}\n```"
        )


class SpookyAuthorTask(MleBenchTask):
    name = "mle_spooky_author"
    competition_id = "spooky-author-identification"
    _baseline_score = 0.0

    @property
    def pass_threshold(self) -> float:
        return 0.05


class JigsawUnintendedBiasTask(MleBenchTask):
    name = "jigsaw-unintended-bias-in-toxicity-classification"
    competition_id = "jigsaw-unintended-bias-in-toxicity-classification"
    _baseline_score = 0.0

    @property
    def pass_threshold(self) -> float:
        return 0.05


class SmartphoneDecimeter2022Task(MleBenchTask):
    name = "smartphone-decimeter-2022"
    competition_id = "smartphone-decimeter-2022"
    _baseline_score = 0.0

    @property
    def pass_threshold(self) -> float:
        return 0.05


class PetfinderPawpularityTask(MleBenchTask):
    name = "petfinder-pawpularity-score"
    competition_id = "petfinder-pawpularity-score"
    _baseline_score = 0.0

    @property
    def pass_threshold(self) -> float:
        return 0.05


class CassavaLeafDiseaseTask(MleBenchTask):
    name = "cassava-leaf-disease-classification"
    competition_id = "cassava-leaf-disease-classification"
    _baseline_score = 0.0

    @property
    def pass_threshold(self) -> float:
        return 0.05


REGISTRY: dict[str, type[AbstractBenchmarkTask]] = {
    "dl_lr_schedule": DlLrScheduleTask,
    "dl_activation_function": DlActivationFunctionTask,
    "cv_data_augmentation": CvDataAugmentationTask,
    "dl_weight_initialization": DlWeightInitializationTask,
    "cv_classification_loss": CvClassificationLossTask,
    "cv_sample_weighting": CvSampleWeightingTask,
    "cv_pooling_aggregation": CvPoolingAggregationTask,
    "cv_multitask_loss": CvMultitaskLossTask,
    "dl_regularization": DlRegularizationTask,
    "dl_residual_connection": DlResidualConnectionTask,
    "mle_spooky_author": SpookyAuthorTask,
    "jigsaw-unintended-bias-in-toxicity-classification": JigsawUnintendedBiasTask,
    "smartphone-decimeter-2022": SmartphoneDecimeter2022Task,
    "petfinder-pawpularity-score": PetfinderPawpularityTask,
    "cassava-leaf-disease-classification": CassavaLeafDiseaseTask,
}


def get_task(cfg: dict[str, Any], name: str, subtask: str | None = None) -> AbstractBenchmarkTask:
    if name not in REGISTRY:
        raise ValueError(f"Unknown task {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, subtask=subtask)


def _subtask_spec_from_test_cmd(row: dict[str, Any]) -> dict[str, Any]:
    label = str(row.get("label") or "")
    cmd = str(row.get("cmd") or "")
    stem = Path(cmd).stem
    spec: dict[str, Any] = {
        "cmd": cmd,
        "group": row.get("group"),
        "compute": row.get("compute"),
        "time": row.get("time"),
        "package": row.get("package"),
        "hidden": bool(row.get("hidden", False)),
        "epochs": 200,
        "batch_size": 128,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }
    if stem:
        parts = stem.split("_")
        if len(parts) >= 2:
            spec["arch"] = parts[0]
            ds = "_".join(parts[1:])
            if ds.endswith("lt"):
                spec["dataset"] = ds[:-2]
                spec["imbalance_ratio"] = 50 if spec["arch"] == "vgg16bn" and spec["dataset"] == "cifar100" else 100
            elif ds.endswith("mt"):
                spec["dataset"] = "cifar100"
                spec["multitask"] = True
            else:
                spec["dataset"] = ds
    if "dataset" not in spec:
        m = re.search(r"-(cifar10|cifar100|fmnist)", label)
        if m:
            spec["dataset"] = m.group(1)
    dataset = spec.get("dataset")
    spec["data_subdir"] = "fmnist" if dataset == "fmnist" else "cifar"
    return spec


def worker_constraints(task: AbstractBenchmarkTask) -> dict[str, Any]:
    if isinstance(task, MLSBenchTask):
        return {
            "modifiable_files": ["editable_region.py"],
            "entrypoint": "bash run.sh result.json",
            "edit_file": task.edit_file,
            "edit_start": task.edit_start,
            "edit_end": task.edit_end,
        }
    return {
        "modifiable_files": ["baseline.py"],
        "entrypoint": "bash run.sh result.json",
        "required_output": "submission.csv",
    }


def asset_dir(group: str, task_name: str) -> Path:
    return project_root() / "assets" / group / task_name


def frontline_ids_from_asset(group: str, task_name: str) -> list[str]:
    path = asset_dir(group, task_name) / "frontline_papers.txt"
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def frontline_hints_from_asset(group: str, task_name: str) -> dict[str, dict[str, Any]]:
    path = asset_dir(group, task_name) / "frontline_papers.txt"
    if not path.exists():
        return {}
    hints: dict[str, dict[str, Any]] = {}
    pending_comment = ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            pending_comment = line.lstrip("#").strip()
            continue
        if not pending_comment:
            continue
        hint: dict[str, Any] = {}
        m = re.search(r"\b(19|20)\d{2}\b", pending_comment)
        if m:
            hint["year"] = int(m.group(0))
        if " - " in pending_comment:
            title = pending_comment.rsplit(" - ", 1)[-1].strip()
            if title:
                hint["title"] = title
        elif pending_comment:
            hint["title"] = pending_comment
        if hint:
            hints[line] = hint
    return hints


def load_frontline_papers(task_name: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    group = "mle_tasks" if task_name in {
        "mle_spooky_author",
        "jigsaw-unintended-bias-in-toxicity-classification",
        "smartphone-decimeter-2022",
        "petfinder-pawpularity-score",
        "cassava-leaf-disease-classification",
    } else "mls_tasks"
    meta_path = asset_dir(group, task_name) / "frontline_metadata.json"
    metadata: dict[str, dict[str, Any]] = {}
    if meta_path.exists():
        try:
            for row in json.loads(meta_path.read_text()):
                if row.get("arxiv_id"):
                    metadata[row["arxiv_id"]] = row
        except Exception:
            metadata = {}
    hints = frontline_hints_from_asset(group, task_name)
    out = []
    for arxiv_id in frontline_ids_from_asset(group, task_name):
        meta = metadata.get(arxiv_id) or find_arxiv_metadata(arxiv_id, cfg)
        hint = hints.get(arxiv_id) or {}
        out.append({
            "arxiv_id": arxiv_id,
            "title": meta.get("title") or hint.get("title") or f"[{arxiv_id}]",
            "abstract": short_text(meta.get("abstract", ""), 1200),
            "year": meta.get("year") or year_from_metadata(meta) or hint.get("year"),
        })
    return out


def find_arxiv_metadata(arxiv_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if not cfg.get("scan_arxiv_metadata", False):
        return {}
    root = Path(cfg.get("arxiv_root", project_root().parent / "data" / "arxiv" / "papers"))
    for path in root.rglob("metadata.json"):
        try:
            row = json.loads(path.read_text())
        except Exception:
            continue
        if row.get("arxiv_id") == arxiv_id or row.get("id") == arxiv_id:
            return row
    return {}


def year_from_metadata(meta: dict[str, Any]) -> int | None:
    for key in ("year", "created", "published"):
        value = meta.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


def verify_signed_result(result: dict[str, Any], secret: str | None = None) -> tuple[float | None, str | None]:
    val = result.get("val_metric")
    if val is None:
        return None, result.get("error") or "val_metric is null"
    if "_sig" not in result:
        return None, "result.json lacks HMAC signature"
    secret = secret or os.environ.get("BENCHMARK_HMAC_KEY", "benchmark-eval-secret")
    payload = f"{float(val):.6f}"
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(result["_sig"])):
        return None, "HMAC mismatch"
    return float(val), None

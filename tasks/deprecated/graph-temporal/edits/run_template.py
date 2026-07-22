"""Unified CLI runner for graph-temporal task.

Usage:  python run.py
Reads ENV and SEED from environment variables.
"""

import os
import sys

sys.path.insert(0, "/workspace")

from custom_graph_model import Custom, CustomConfig
from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.metrics import masked_mae
from basicts.runners.callback import EarlyStopping, GradientClipping

DS_INFO = {
    # METR-LA / PEMS-BAY store traffic *speed* and use 0.0 as the sentinel for
    # missing sensor readings, so null_val=0.0 is the standard BasicTS setting.
    # PEMS04 stores vehicle *flow counts* which are rarely exactly zero but
    # frequently very small at night; using 0.0 as the mask threshold leaves
    # near-zero denominators in the MAPE computation and inflates it 2-4x.
    # Following standard BasicTS practice we mask values below 1e-5 instead.
    "METR-LA":  {"num_features": 207, "adj": os.environ.get("DATA_ROOT", "/data") + "/datasets/METR-LA/adj_mx.pkl",  "null_val": 0.0},
    "PEMS-BAY": {"num_features": 325, "adj": os.environ.get("DATA_ROOT", "/data") + "/datasets/PEMS-BAY/adj_mx.pkl", "null_val": 0.0},
    "PEMS04":   {"num_features": 307, "adj": os.environ.get("DATA_ROOT", "/data") + "/datasets/PEMS04/adj_mx.pkl",   "null_val": 1e-5},
}

env = os.environ.get("ENV", "METR-LA")
seed = int(os.environ.get("SEED", 42))
ds = DS_INFO[env]
null_val = ds["null_val"]

model_config = CustomConfig(
    input_len=12,
    output_len=12,
    num_features=ds["num_features"],
    adj_path=ds["adj"],
)

cfg = BasicTSForecastingConfig(
    model=Custom,
    model_config=model_config,
    dataset_name=env,
    dataset_params={
        "input_len": 12, "output_len": 12,
        "use_timestamps": True,
        "data_file_path": os.environ.get("DATA_ROOT", "/data") + f"/datasets/{env}",
    },
    gpus="0", seed=seed, num_epochs=100, batch_size=64,
    metrics=["MAE", "RMSE", "MAPE"], target_metric="MAE",
    loss=masked_mae, null_val=null_val,
    norm_each_channel=False, rescale=True,
    optimizer_params={"lr": 2e-3, "weight_decay": 1e-4},
    callbacks=[EarlyStopping(patience=15), GradientClipping(5.0)],
    ckpt_save_dir=f"checkpoints/Custom/{env}_s{seed}",
)

BasicTSLauncher.launch_training(cfg)

"""Pre-edit operations for the VICON package.
Injects TRAIN_METRICS into train_utils.py board_loss and patches train.py
to support custom model selection via use_custom_model config flag.
"""

# Replace board_loss to also emit TRAIN_METRICS (lines 26-30 of train_utils.py)
_TRAIN_METRICS_PATCH = (
    'def board_loss(trainer, pairs, prefix, cfg):\n'
    '    loss = trainer.get_loss(pairs)\n'
    '    print(f"train step: {trainer.train_step}, {prefix}_loss: {loss}")\n'
    '    print(f"TRAIN_METRICS step={trainer.train_step} {prefix}_loss={loss:.6f}", flush=True)\n'
    '    if cfg.board:\n'
    '        wandb.log({"step": trainer.train_step, f"{prefix}_loss": loss})\n'
)

# Patch train.py to support custom model selection (lines 37-38)
_TRAIN_CUSTOM_MODEL = (
    '    # Creation of model instance\n'
    '    if getattr(cfg, "use_custom_model", False):\n'
    '        from custom_model import CustomModel\n'
    '        model = CustomModel(cfg.model)\n'
    '        print("Using Custom model")\n'
    '    else:\n'
    '        model = models.ICON_UNCROPPED(cfg.model)\n'
    '        print("Using VICON model")\n'
)

OPS = [
    # 1. Inject TRAIN_METRICS into train_utils.py board_loss function (lines 26-30)
    {
        "op": "replace",
        "file": "VICON/src/train_utils.py",
        "start_line": 26,
        "end_line": 30,
        "content": _TRAIN_METRICS_PATCH,
    },
    # 2. Patch train.py to support custom model (lines 37-38)
    {
        "op": "replace",
        "file": "VICON/src/train.py",
        "start_line": 37,
        "end_line": 38,
        "content": _TRAIN_CUSTOM_MODEL,
    },
]

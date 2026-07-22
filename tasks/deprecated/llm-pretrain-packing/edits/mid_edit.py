"""Mid-edit: creates custom_dataloader.py and patches base_train.py to use it."""
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

_NEW_IMPORT = "from nanochat.custom_dataloader import custom_tokenizing_distributed_data_loader_bos as tokenizing_distributed_data_loader_bos_bestfit, custom_tokenizing_distributed_data_loader_with_state_bos as tokenizing_distributed_data_loader_with_state_bos_bestfit"

OPS = [
    {
        "op": "create",
        "file": "nanochat/nanochat/custom_dataloader.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "replace",
        "file": "nanochat/scripts/base_train.py",
        "start_line": 29,
        "end_line": 29,
        "content": _NEW_IMPORT + "\n",
    },
]

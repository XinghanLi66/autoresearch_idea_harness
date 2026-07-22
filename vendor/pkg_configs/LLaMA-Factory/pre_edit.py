"""Pre-edit operations applied to the LLaMA-Factory workspace before the agent starts.

Each entry is a dict with:
  op      - "replace" | "insert" | "delete" | "create"
  file    - package-relative path (e.g. "LLaMA-Factory/src/...")
  line    - 1-indexed line number (for replace/delete/insert; not used for create)
  content - new line content (for replace/insert/create; trailing newline added if missing)
"""

import json

# Dataset entries are inserted into the workspace's data/dataset_info.json
# at workspace setup time. Their file_name fields point to the bind-mounted
# /data/llama-factory-data directory (see pkg config data_deps).
_DATASET_ENTRIES = {
    "math_step_dpo": {
        "file_name": "/data/llama-factory-data/math_step_dpo.json",
        "ranking": True,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    },
    "metamathqa": {
        "file_name": "/data/llama-factory-data/metamathqa.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
    },
}

# Render the entries as JSON object literal contents (no surrounding braces),
# ending with a comma so the next existing entry remains valid JSON.
_entries_json = json.dumps(_DATASET_ENTRIES, indent=2)
# Strip the outer braces and indent every line by 2 spaces, append trailing comma.
_inner = "\n".join("  " + line for line in _entries_json.splitlines()[1:-1]) + ","


OPS = [
    {
        "op": "replace",
        "file": "LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py",
        "start_line": 183, "end_line": 183,
        "content": '    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo", "custom"] = field(',
    },
    # Insert custom dataset entries right after the opening "{" of dataset_info.json.
    {
        "op": "insert",
        "file": "LLaMA-Factory/data/dataset_info.json",
        "after_line": 1,
        "content": _inner,
    },
]

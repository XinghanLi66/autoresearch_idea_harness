"""Parameter budget check for rl-offline-pomdp (standalone).

Run by tools.py before training: python /workspace/_task/budget_check.py
Imports each baseline, instantiates models, counts params, and
asserts the agent's model doesn't exceed 1.05x the largest baseline.

The POMDP task uses a FIXED ModelBackbone (encoders + LSTM) and EDITABLE
output heads. The budget counts only head parameters (total - backbone).
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch

TASK_DIR = "/workspace/_task"
WORKSPACE_FILE = "/workspace/katakomba/algorithms/small_scale/custom_nethack_pomdp.py"

# -- Env dimensions (no gym needed) --
# NetHack has 121 discrete actions for all characters.
NUM_ACTIONS = 121
# Backbone config (must match TrainConfig defaults)
RNN_HIDDEN_DIM = 2048
RNN_LAYERS = 2
RNN_DROPOUT = 0.0
USE_PREV_ACTION = True


def load_module(path, name=None):
    name = name or f"_mod_{id(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_ops(lines, ops, filename):
    result = list(lines)
    sorted_ops = sorted(
        [o for o in ops if o.get("file") == filename],
        key=lambda o: -o.get("start_line", o.get("after_line", 0)),
    )
    for op in sorted_ops:
        if op["op"] == "replace":
            s, e = op["start_line"] - 1, op["end_line"]
            result[s:e] = op["content"].splitlines()
        elif op["op"] == "insert":
            after = op["after_line"]
            result[after:after] = op["content"].splitlines()
        elif op["op"] == "delete":
            s, e = op["start_line"] - 1, op["end_line"]
            del result[s:e]
    return result


def count_head_params(module_path):
    """Import module, instantiate OfflineAlgorithm with ModelBackbone, return head param count.

    Head params = total model params - backbone params.
    """
    mod = load_module(module_path, f"_check_{id(module_path)}")
    # Create the FIXED backbone (defined in the template module itself)
    backbone = mod.ModelBackbone(
        action_dim=NUM_ACTIONS,
        rnn_hidden_dim=RNN_HIDDEN_DIM,
        rnn_layers=RNN_LAYERS,
        rnn_dropout=RNN_DROPOUT,
        use_prev_action=USE_PREV_ACTION,
    )
    backbone_params = sum(p.numel() for p in backbone.parameters())

    # Create a minimal TrainConfig (defined in the template)
    config = mod.TrainConfig()
    algo = mod.OfflineAlgorithm(
        backbone=backbone,
        config=config,
        device="cpu",
    )
    total_params = sum(p.numel() for p in algo.model.parameters())
    head_params = total_params - backbone_params
    return head_params, backbone_params, total_params


# -- Get template content --
mid_edit = load_module(os.path.join(TASK_DIR, "edits", "mid_edit.py"), "_mid_edit")
config = json.loads(open(os.path.join(TASK_DIR, "config.json")).read())
editable_file = None
for f in config.get("files", []):
    if f.get("edit"):
        editable_file = f["filename"]
        break

template_content = None
for op in mid_edit.OPS:
    if op.get("op") == "create" and op.get("file") == editable_file:
        template_content = op["content"]
        break

assert template_content, f"No template found for {editable_file}"
template_lines = template_content.splitlines()

# -- Count params for each baseline --
baseline_head_params = {}
for bl_name, bl_cfg in config.get("baselines", {}).items():
    edit_path = os.path.join(TASK_DIR, bl_cfg["edit_ops"])
    if not os.path.exists(edit_path):
        continue
    bl_mod = load_module(edit_path, f"_bl_{bl_name}")
    ops = getattr(bl_mod, "OPS", [])
    modified_lines = apply_ops(template_lines, ops, editable_file)
    modified_code = "\n".join(modified_lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(modified_code)
        tmp_path = f.name
    try:
        head_p, backbone_p, total_p = count_head_params(tmp_path)
        baseline_head_params[bl_name] = head_p
        print(f"  baseline {bl_name}: {head_p} head params (total: {total_p}, backbone: {backbone_p})")
    except Exception as e:
        print(f"  baseline {bl_name}: ERROR ({e})")
    finally:
        os.unlink(tmp_path)

if not baseline_head_params:
    print("WARNING: no baselines could be evaluated, skipping budget check")
    sys.exit(0)

max_baseline = max(baseline_head_params.values())
max_name = max(baseline_head_params, key=baseline_head_params.get)
budget = int(max_baseline * 1.05)

# -- Count params for agent's version --
agent_head, agent_backbone, agent_total = count_head_params(WORKSPACE_FILE)
print(f"\n  agent model: {agent_head} head params (total: {agent_total}, backbone: {agent_backbone})")
print(f"  budget: {budget} (1.05 x {max_name}={max_baseline})")
print(f"  num_actions={NUM_ACTIONS}")

if agent_head > budget:
    print(f"\nFAILED: {agent_head} > {budget}", file=sys.stderr)
    sys.exit(1)

print("\nPASSED")

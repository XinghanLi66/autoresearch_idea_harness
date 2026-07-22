"""Set up an isolated nanoGPT workspace for a single baseline edit.

Usage:
    python verify124m_setup.py <baseline_name> <workspace_root>

This:
  1. Copies vendor/external_packages/nanoGPT -> <workspace_root>/nanoGPT
  2. Renders tasks/ttt-memory/edits/custom_template.py -> <workspace_root>/nanoGPT/custom_pretrain.py
  3. Applies tasks/ttt-memory/edits/<baseline>.edit.py (replace ops on the rendered file)

It does NOT run training — that's done by a separate apptainer call in the SLURM script.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path("/scratch/gpfs/CHIJ/bohan/MLS-Bench")
SRC_PKG = ROOT / "vendor/external_packages/nanoGPT"
TEMPLATE = ROOT / "tasks/ttt-memory/edits/custom_template.py"
EDITS_DIR = ROOT / "tasks/ttt-memory/edits"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def apply_replace_op(target_file: Path, start_line: int, end_line: int, content: str):
    """1-indexed inclusive replace: lines [start_line, end_line] -> content."""
    text = target_file.read_text()
    lines = text.splitlines(keepends=True)
    if start_line < 1 or end_line > len(lines):
        raise ValueError(f"replace range {start_line}-{end_line} out of bounds (file has {len(lines)} lines)")
    if not content.endswith("\n"):
        content = content + "\n"
    new_lines = lines[: start_line - 1] + [content] + lines[end_line:]
    target_file.write_text("".join(new_lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("workspace_root")
    args = p.parse_args()

    ws = Path(args.workspace_root)
    pkg_dst = ws / "nanoGPT"
    if pkg_dst.exists():
        shutil.rmtree(pkg_dst)
    ws.mkdir(parents=True, exist_ok=True)
    print(f"copying nanoGPT -> {pkg_dst}", flush=True)
    shutil.copytree(SRC_PKG, pkg_dst)

    custom_pretrain = pkg_dst / "custom_pretrain.py"
    print(f"writing template -> {custom_pretrain}", flush=True)
    shutil.copy(TEMPLATE, custom_pretrain)

    edit_path = EDITS_DIR / f"{args.baseline}.edit.py"
    if not edit_path.exists():
        raise FileNotFoundError(f"baseline edit not found: {edit_path}")
    print(f"applying edit: {edit_path}", flush=True)
    mod = _load_module(edit_path)

    target = pkg_dst / "custom_pretrain.py"  # all baseline edits target nanoGPT/custom_pretrain.py
    for op in mod.OPS:
        assert op["file"].endswith("nanoGPT/custom_pretrain.py"), op["file"]
        if op["op"] == "replace":
            apply_replace_op(target, op["start_line"], op["end_line"], op["content"])
            print(f"  replace lines {op['start_line']}-{op['end_line']} ({len(op['content'].splitlines())} new lines)", flush=True)
        else:
            raise NotImplementedError(op["op"])

    # Sanity: file must still be syntactically valid Python.
    import ast
    ast.parse(target.read_text())
    print(f"OK — {target} ({target.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MLS-Bench unified CLI.

Subcommands:
    agent     Run an LLM agent against a task
    baseline  Run baseline method(s) and record results
    build     Build package runtime (Apptainer, Docker, or local/conda)
    run       Run a script inside a pre-built container image
    fetch     Clone/update external packages from packages.yaml
    data      List/manage large datasets declared in pkg_config data_deps

Notes:
    - `build` automatically prepares data dependencies after building the image.
    - `agent`/`baseline` auto-build images (and data) if missing.
    - `data` is mainly useful for listing status (`--list`) or manual preparation.

Examples:
    mlsbench fetch                                        # clone all packages
    mlsbench build pytorch-examples                       # build image/local env + prepare data
    mlsbench data --list                                  # list data deps and status
    mlsbench baseline demo-task-1 --name default          # run a baseline
    mlsbench agent demo-task-1 --model claude-sonnet-4-6  # run an agent
    mlsbench run CORL --task rl-offline-continuous --run-cmd scripts/halfcheetah.sh
"""

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from mlsbench import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGES_DIR = PROJECT_ROOT / "vendor" / "images"
PKG_CONFIGS_DIR = PROJECT_ROOT / "vendor" / "pkg_configs"
EXT_PKG_DIR = PROJECT_ROOT / "vendor" / "external_packages"
PACKAGES_YAML = PROJECT_ROOT / "configs" / "packages.yaml"
PASSTHROUGH_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "HF_ENDPOINT",
    "OPENROUTER_API_KEY_NEW",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], *, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, **kwargs)


def normalize(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "")


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def docker_image_tag(name: str) -> str:
    """Return a Docker-compatible image tag for a package name."""
    return f"mlsbench/{name.lower()}:latest"


def docker_run_instruction_lines(cmd: str) -> list[str]:
    """Render a Dockerfile RUN instruction, preserving multiline shell blocks."""
    if "\n" not in cmd:
        return [f"RUN {cmd}"]
    marker = "MLSBENCH_RUN"
    return [f"RUN <<'{marker}'", *cmd.splitlines(), marker]


def resolve_docker_extra_files(pkg_config: dict, data_root: str | Path = "") -> list[dict]:
    """Resolve pkg_config extra_files into Docker named build contexts."""
    entries: list[dict] = []
    for idx, ef in enumerate(pkg_config.get("extra_files", [])):
        src_str = expand_path_template(ef["src"], data_root)
        src = Path(src_str).expanduser().resolve()
        if not src.exists():
            logger.warning("extra_files src not found, skipping: %s", src)
            continue
        context_name = f"mls_extra_{idx}"
        if src.is_dir():
            context_path = src
            copy_src = "."
        else:
            context_path = src.parent
            copy_src = src.name
        entries.append({
            "context_name": context_name,
            "context_path": str(context_path),
            "copy_src": copy_src,
            "dst": ef["dst"],
        })
    return entries


def iter_passthrough_env_vars(names: tuple[str, ...] = PASSTHROUGH_ENV_VARS):
    """Yield host env vars that should be forwarded into Docker build/run steps."""
    for name in names:
        value = os.environ.get(name)
        if value:
            yield name, value


def get_apptainer_build_cmd() -> list[str]:
    """Return the appropriate apptainer build command for the current uid."""
    cmd = ["apptainer", "build"]
    if os.geteuid() != 0:
        cmd.extend(["--fakeroot", "--ignore-fakeroot-command"])
    return cmd


def expand_path_template(s: str, data_root: str | Path = "") -> str:
    """Expand template variables in path strings.

    Supported variables:
        {data_root}    — from config.yaml (default: vendor/data)
        {project_root} — MLS-Bench project root
    """
    result = s.replace("{project_root}", str(PROJECT_ROOT))
    result = result.replace("{data_root}", str(data_root) if data_root else str(PROJECT_ROOT / "vendor" / "data"))
    return result


def conda_env_for_pkg(pkg_name: str) -> str:
    """Return the per-package conda environment name."""
    return f"mlsbench-{pkg_name}"


def find_conda_exe() -> str | None:
    """Locate the conda executable on PATH or common local installs."""
    conda_exe = shutil.which("conda")
    if conda_exe:
        return conda_exe

    candidates = [
        os.environ.get("CONDA_EXE", ""),
        str(Path.home() / "miniconda3" / "condabin" / "conda"),
        str(Path.home() / "anaconda3" / "condabin" / "conda"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _conda_env_exists(conda_exe: str, env_name: str) -> bool:
    """Check whether a named conda environment already exists."""
    try:
        result = subprocess.run(
            [conda_exe, "env", "list", "--json"],
            capture_output=True, text=True, check=True,
        )
        envs = json.loads(result.stdout).get("envs", [])
        return any(Path(e).name == env_name for e in envs)
    except Exception:
        return False


def _parse_base_image(base_image: str) -> dict:
    """Parse a Docker base_image string into structured info.

    Returns a dict with keys: python, torch, cuda (any may be empty).

    Examples::

        pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
          -> {'python': '3.10', 'torch': '2.1.2', 'cuda': 'cu121'}
        python:3.11-slim
          -> {'python': '3.11', 'torch': '', 'cuda': ''}
    """
    info: dict[str, str] = {"python": "3.10", "torch": "", "cuda": ""}

    # Python version (explicit, e.g. python:3.11-slim)
    m = re.search(r"python[:/](\d+\.\d+)", base_image)
    if m:
        info["python"] = m.group(1)

    # PyTorch version  (e.g. pytorch:2.1.2-cuda...)
    m = re.search(r"pytorch[:/](\d+\.\d+\.\d+)", base_image)
    if m:
        info["torch"] = m.group(1)
        # The official pytorch/pytorch images switched to Python 3.11 starting
        # with the 2.4.x line (and cudnn9 tag), so derive the env's Python from
        # the torch version when no explicit `python:X.Y` is in the base_image.
        if not re.search(r"python[:/]\d+\.\d+", base_image):
            try:
                major, minor, _ = info["torch"].split(".", 2)
                if (int(major), int(minor)) >= (2, 4):
                    info["python"] = "3.11"
            except Exception:
                pass

    # CUDA version  (e.g. cuda12.1 -> cu121)
    m = re.search(r"cuda(\d+)\.(\d+)", base_image)
    if m:
        info["cuda"] = f"cu{m.group(1)}{m.group(2)}"

    return info


def _conda_base_install_cmds(pkg_config: dict) -> list[str]:
    """Derive pip install commands for the conda env base layer.

    Mirrors what the Docker ``base_image`` provides (e.g. PyTorch + CUDA).
    Returns a list of shell commands.
    """
    base_image = pkg_config.get("base_image", "")
    if not base_image:
        return []
    info = _parse_base_image(base_image)
    cmds: list[str] = []
    if info["torch"]:
        torch_spec = f"torch=={info['torch']}"
        tv_map = {
            "2.1": "0.16.2",
            "2.2": "0.17.2",
            "2.3": "0.18.1",
            "2.4": "0.19.1",
            "2.5": "0.20.1",
        }
        torch_minor = ".".join(info["torch"].split(".")[:2])
        tv_version = tv_map.get(torch_minor, "")
        extras = f" torchvision=={tv_version}" if tv_version else ""
        if info["cuda"]:
            cmds.append(
                f"python -m pip install {torch_spec}{extras} --index-url "
                f"https://download.pytorch.org/whl/{info['cuda']}"
            )
        else:
            cmds.append(f"python -m pip install {torch_spec}{extras}")
    return cmds


def _prefer_env_python_for_pip(cmd: str, *, use_conda: bool) -> str:
    """Rewrite bare ``pip ...`` to ``python -m pip ...`` for conda-backed local runs.

    Some shared-machine environments put ``~/.local/bin`` ahead of the conda env's
    bin directory, so invoking ``pip`` directly can escape the target environment.
    """
    if not use_conda:
        return cmd
    stripped = cmd.lstrip()
    if not stripped.startswith("pip "):
        return cmd
    leading = cmd[: len(cmd) - len(stripped)]
    return f"{leading}python -m {stripped}"


def _remove_conda_env(conda_exe: str, env_name: str) -> None:
    """Delete a named conda env if it exists."""
    logger.info("Removing conda env '%s'", env_name)
    subprocess.run(
        [conda_exe, "remove", "-n", env_name, "--all", "-y"],
        check=True,
    )


def ensure_conda_env(
    pkg_name: str,
    pkg_config: dict,
    *,
    force: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    """Create the per-package conda env if it doesn't exist.

    Mirrors Docker ``base_image``: creates a conda env with the right
    Python version and installs PyTorch + CUDA matching the base image.
    Returns the conda env name.
    """
    env_name = conda_env_for_pkg(pkg_name)
    conda_exe = find_conda_exe()
    if not conda_exe:
        raise RuntimeError(
            "Local runtime requires conda but 'conda' is not on PATH. "
            "Install miniconda or set conda_prefix in config."
        )
    if _conda_env_exists(conda_exe, env_name):
        if force:
            _remove_conda_env(conda_exe, env_name)
        else:
            logger.info("Conda env '%s' already exists", env_name)
            return env_name

    if _conda_env_exists(conda_exe, env_name):
        logger.info("Conda env '%s' already exists", env_name)
        return env_name

    base_image = pkg_config.get("base_image", "")
    info = _parse_base_image(base_image)
    py_ver = info["python"]
    logger.info("Creating conda env '%s' (python=%s) for package '%s'", env_name, py_ver, pkg_name)
    subprocess.run(
        [conda_exe, "create", "-n", env_name, f"python={py_ver}", "-y"],
        check=True,
    )

    # Install base-image packages (e.g. PyTorch + CUDA) into the new env
    base_cmds = _conda_base_install_cmds(pkg_config)
    for cmd_str in base_cmds:
        logger.info("[conda-base] %s", cmd_str)
        subprocess.run(
            [conda_exe, "run", "--no-capture-output", "-n", env_name,
             "bash", "-c", cmd_str],
            check=True,
            env=env,
        )

    return env_name


def wrap_with_conda(cmd: list[str], global_config: dict | None, *, pkg_name: str | None = None) -> list[str]:
    """Optionally wrap a command in ``conda run`` based on config.

    When *pkg_name* is given and no explicit conda_env/conda_prefix is
    configured, the per-package env ``mlsbench-<pkg_name>`` is used
    automatically when conda is available.
    """
    global_config = global_config or {}
    conda_prefix = str(global_config.get("conda_prefix", "") or "").strip()
    conda_env = str(global_config.get("conda_env", "") or "").strip()
    conda_exe: str | None = None

    # Auto-resolve per-package env when no explicit config
    if not conda_prefix and not conda_env and pkg_name:
        conda_exe = find_conda_exe()
        if not conda_exe:
            return cmd
        conda_env = conda_env_for_pkg(pkg_name)

    if not conda_prefix and not conda_env:
        return cmd

    if not conda_exe:
        conda_exe = find_conda_exe()
    if not conda_exe:
        raise RuntimeError(
            "Conda-backed local runtime requested, but 'conda' is not on PATH."
        )

    wrapped = [conda_exe, "run", "--no-capture-output"]
    if conda_prefix:
        wrapped.extend(["--prefix", conda_prefix])
    else:
        wrapped.extend(["--name", conda_env])
    wrapped.extend(cmd)
    return wrapped


_LOCAL_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def local_thread_limit(global_config: dict | None = None) -> int:
    """Return the default CPU thread budget for local/conda executions."""
    global_config = global_config or {}
    raw = (
        os.environ.get("MLSBENCH_LOCAL_THREADS")
        or global_config.get("local_thread_limit")
        or ""
    )
    try:
        limit = int(str(raw).strip()) if str(raw).strip() else 16
    except ValueError:
        limit = 16
    return max(1, limit)


def apply_local_thread_limits(
    env: dict[str, str],
    global_config: dict | None = None,
) -> dict[str, str]:
    """Populate default thread limits for local/conda execution.

    The defaults are intentionally moderate so that multiple local jobs can
    coexist without each process grabbing all host CPU cores. Explicit values
    already present in *env* are preserved.
    """
    limit = str(local_thread_limit(global_config))
    for key in _LOCAL_THREAD_ENV_KEYS:
        env.setdefault(key, limit)
    return env


def local_thread_limit_exports(global_config: dict | None = None) -> list[str]:
    """Return shell export lines for the default local thread budget."""
    limit = local_thread_limit(global_config)
    return [f"export {key}=${{{key}:-{limit}}}" for key in _LOCAL_THREAD_ENV_KEYS]


def _expand_env_vars(value: str, base_env: dict[str, str]) -> str:
    pattern = re.compile(r"\$(\w+)|\$\{([^}]+)\}")

    def repl(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2) or ""
        return base_env.get(name, "")

    return pattern.sub(repl, value)


def _translate_local_string(
    value: str,
    path_map: dict[str, str],
    base_env: dict[str, str],
) -> str:
    translated = _expand_env_vars(value, base_env)
    for container_path, host_path in sorted(path_map.items(), key=lambda item: len(item[0]), reverse=True):
        # Only rewrite standalone path segments, not incidental substrings
        # inside URLs like ".../datasets/...".
        pattern = re.compile(
            rf"(?<![A-Za-z0-9._-]){re.escape(container_path)}(?=$|\s|['\"=:/])"
        )
        translated = pattern.sub(host_path, translated)
    return translated


def _resolve_local_path_map(
    pkg_config: dict,
    pkg_dir: Path,
    data_root: str | Path,
) -> dict[str, str]:
    workdir = pkg_config.get("workdir", "/app").rstrip("/")
    path_map = {f"{workdir}/{pkg_dir.name}": str(pkg_dir.resolve())}
    resolved_data_root = str(Path(data_root).expanduser().resolve())
    path_map.setdefault("/data", resolved_data_root)
    # Also map {workdir}/data -> data_root for containers that store data
    # under the workdir (e.g. /workspace/data in many PyTorch images).
    path_map.setdefault(f"{workdir}/data", resolved_data_root)
    # Map workdir root -> pkg_dir's parent so install_cmds with bare
    # `cd /workspace` (then `python pkg/script.py`) work in local mode.
    path_map.setdefault(workdir, str(pkg_dir.parent.resolve()))
    # Map container HOME (e.g. /root) to real HOME so install_cmds with
    # hard-coded paths like /root/.qlib/ translate correctly in local mode.
    container_home = str(pkg_config.get("env", {}).get("HOME", "")).strip()
    real_home = os.environ.get("HOME", "")
    if container_home and real_home and container_home != real_home:
        path_map[container_home] = real_home
    for bind in resolve_data_binds(pkg_config, data_root):
        host_path, container_path = bind.split(":", 1)
        path_map[container_path] = str(Path(host_path).expanduser().resolve())
    return path_map


def _build_local_env(pkg_config: dict, pkg_dir: Path, data_root: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    path_map = _resolve_local_path_map(pkg_config, pkg_dir, data_root)
    merged_env = dict(pkg_config.get("env", {}))
    merged_env.update(pkg_config.get("local_env", {}))
    for key, value in merged_env.items():
        expanded = expand_path_template(str(value), data_root)
        env[key] = _translate_local_string(expanded, path_map, env)
    return env


def _should_skip_local_install_cmd(cmd: str) -> bool:
    stripped = cmd.strip()
    if not stripped:
        return True
    privileged_prefixes = (
        "apt-get ",
        "apt ",
        "yum ",
        "dnf ",
        "apk ",
        "pacman ",
        "sudo ",
    )
    if any(stripped.startswith(prefix) for prefix in privileged_prefixes):
        return True
    if " apt-get " in stripped or " DEBIAN_FRONTEND=" in stripped:
        return True
    return False


def _local_build_stamp_path(pkg_name: str) -> Path:
    stamp_dir = PROJECT_ROOT / "vendor" / "images" / "local"
    stamp_dir.mkdir(parents=True, exist_ok=True)
    return stamp_dir / f"{pkg_name}.json"


def local_python_target_dir(pkg_name: str, project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return root / "vendor" / "local_site_packages" / pkg_name


def _git_head_commit(pkg_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(pkg_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return ""


def _local_build_fingerprint(pkg_name: str, pkg_dir: Path, pkg_config: dict, global_config: dict) -> str:
    payload = {
        "pkg": pkg_name,
        "pkg_dir": str(pkg_dir.resolve()),
        "head": _git_head_commit(pkg_dir),
        "install_cmds": pkg_config.get("local_install_cmds", pkg_config.get("install_cmds", [])),
        "local_env": pkg_config.get("local_env", {}),
        "conda_env": conda_env_for_pkg(pkg_name),
        "data_root": global_config.get("data_root", str(PROJECT_ROOT / "vendor" / "data")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _has_conda_support(global_config: dict) -> bool:
    """Return True if local mode should use per-package conda envs.

    True when: (a) an explicit conda_env/conda_prefix is configured, OR
    (b) conda is available on PATH (auto-detect).
    """
    if (str(global_config.get("conda_env", "") or "").strip()
            or str(global_config.get("conda_prefix", "") or "").strip()):
        return True
    return find_conda_exe() is not None


def build_local_package(
    pkg_name: str,
    pkg_config: dict,
    pkg_dir: Path,
    global_config: dict,
    *,
    force: bool = False,
) -> None:
    """Prepare a package for local runtime by executing install_cmds in conda.

    Automatically creates a per-package conda env ``mlsbench-<pkg>`` when
    conda is available — analogous to building a Docker image per package.
    """
    stamp_path = _local_build_stamp_path(pkg_name)
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _local_build_fingerprint(pkg_name, pkg_dir, pkg_config, global_config)
    if stamp_path.exists() and not force:
        try:
            previous = json.loads(stamp_path.read_text())
        except json.JSONDecodeError:
            previous = {}
        if previous.get("fingerprint") == fingerprint:
            logger.info("Local package already prepared: %s", pkg_name)
            return

    data_root = global_config.get("data_root", str(PROJECT_ROOT / "vendor" / "data"))
    build_env = _build_local_env(pkg_config, pkg_dir, data_root)
    build_env.setdefault("PYTHONNOUSERSITE", "1")
    build_env.setdefault("PIP_NO_USER_CONFIG", "1")

    use_conda = _has_conda_support(global_config)
    if use_conda:
        # Create per-package conda env (like building a Docker image)
        ensure_conda_env(pkg_name, pkg_config, force=force, env=build_env)
    else:
        local_site = local_python_target_dir(pkg_name)
        if force and local_site.exists():
            shutil.rmtree(local_site)
        local_site.mkdir(parents=True, exist_ok=True)
        build_env["PIP_TARGET"] = str(local_site.resolve())
        current_pythonpath = build_env.get("PYTHONPATH", "")
        build_env["PYTHONPATH"] = (
            f"{local_site.resolve()}:{current_pythonpath}" if current_pythonpath else str(local_site.resolve())
        )
    install_cmds = pkg_config.get("local_install_cmds", pkg_config.get("install_cmds", []))
    cwd = str(pkg_dir.resolve())

    for raw_cmd in install_cmds:
        expanded = expand_path_template(str(raw_cmd), data_root)
        translated = _translate_local_string(expanded, _resolve_local_path_map(pkg_config, pkg_dir, data_root), build_env)
        translated = _prefer_env_python_for_pip(translated, use_conda=use_conda)
        if _should_skip_local_install_cmd(translated):
            logger.info("[local-build] Skipping privileged/container-only install command: %s", raw_cmd)
            continue
        # Use -c (not -lc) when conda is active: login shells load user
        # profiles that may shadow the conda env's binaries (e.g. pip).
        shell_flag = "-c" if use_conda else "-lc"
        wrapped_cmd = wrap_with_conda(["bash", shell_flag, translated], global_config, pkg_name=pkg_name)
        # Pipe "y\n" to stdin so interactive prompts (e.g. qlib data
        # download confirmation) are answered automatically.
        run_cmd(wrapped_cmd, cwd=cwd, env=build_env, input="y\n", text=True)

    stamp_path.write_text(
        json.dumps(
            {
                "pkg": pkg_name,
                "pkg_dir": str(pkg_dir.resolve()),
                "fingerprint": fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def resolve_data_binds(pkg_config: dict, data_root: str | Path = "") -> list[str]:
    """Resolve data_bind entries from pkg_config, expanding template variables.

    Also auto-generates binds from data_deps declarations.
    Returns a list of "HOST:CONTAINER" strings.
    """
    binds: list[str] = []

    # Explicit data_bind entries
    data_bind = pkg_config.get("data_bind")
    if data_bind:
        entries = data_bind if isinstance(data_bind, list) else [data_bind]
        for entry in entries:
            binds.append(expand_path_template(entry, data_root))

    # Auto-generate binds from data_deps (skip if already covered by data_bind).
    # IMPORTANT: if host_path doesn't exist on disk, we must NOT create the
    # bind. Some packages (e.g. Time-Series-Library) bake data into /data
    # inside the SIF at build time; if we bind an empty host dir over it,
    # we mask the in-image data and every test fails with file-not-found
    # (or, worse, the runtime falls back to network downloads). Skip+warn
    # so the SIF data wins; the warning prompts the user to run
    # `mlsbench data` for genuinely host-resident deps.
    for dep in pkg_config.get("data_deps", []):
        host_path = dep.get("host_path", "")
        container_path = dep.get("container_path", "")
        if host_path and container_path:
            resolved = expand_path_template(host_path, data_root)
            if not Path(resolved).exists():
                dep_name = dep.get("name", resolved)
                logger.warning(
                    "Data dependency '%s' not found at %s — skipping bind. "
                    "If the package's container has data baked in, this is "
                    "fine; otherwise run `mlsbench data %s` to prepare.",
                    dep_name, resolved, pkg_config.get("__pkg_name__", ""),
                )
                continue
            entry = f"{resolved}:{container_path}"
            if entry not in binds:
                binds.append(entry)

    return binds


def find_pkg_config_file(pkg_name: str) -> Path:
    """Find pkg config JSON by package name.

    Convention: <dir>/config.json where dir name = package name.
    """
    norm = normalize(pkg_name)
    for d in PKG_CONFIGS_DIR.iterdir():
        if not d.is_dir():
            continue
        cfg = d / "config.json"
        if cfg.is_file() and normalize(d.name) == norm:
            return cfg
    available = [d.name for d in PKG_CONFIGS_DIR.iterdir()
                 if d.is_dir() and (d / "config.json").is_file()]
    raise FileNotFoundError(
        f"No config found for '{pkg_name}' in {PKG_CONFIGS_DIR}\n"
        f"Available: {available}"
    )


def load_pkg_config(pkg_name: str) -> tuple[dict, str]:
    """Load pkg config. Returns (config_dict, canonical_name)."""
    cfg_file = find_pkg_config_file(pkg_name)
    with open(cfg_file) as f:
        return json.load(f), cfg_file.parent.name


def _is_local_registry_package(info: dict) -> bool:
    """Return True for scaffold-only packages backed by a local stub dir."""
    return str(info.get("url", "")).strip().lower() == "local"


def _ensure_local_package_stub(pkg_name: str) -> Path:
    """Create the minimal external_packages/<pkg> directory for local packages."""
    pkg_dir = EXT_PKG_DIR / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    init_py = pkg_dir / "__init__.py"
    if not init_py.exists():
        init_py.touch()
    return pkg_dir


def _fetch_single_package(pkg_name: str) -> Path | None:
    """Auto-fetch a single package from packages.yaml. Returns pkg_dir or None."""
    import yaml as _yaml

    if not PACKAGES_YAML.exists():
        return None
    with open(PACKAGES_YAML) as f:
        registry = _yaml.safe_load(f) or {}
    packages = registry.get("packages", {})
    if pkg_name not in packages:
        # Try fuzzy match
        norm = normalize(pkg_name)
        for k in packages:
            if normalize(k) == norm:
                pkg_name = k
                break
        else:
            return None
    info = packages[pkg_name]
    EXT_PKG_DIR.mkdir(parents=True, exist_ok=True)
    if _is_local_registry_package(info):
        logger.info("[auto-fetch] Creating local package stub for %s", pkg_name)
        return _ensure_local_package_stub(pkg_name)

    url, commit = info["url"], info["commit"]
    pkg_dir = EXT_PKG_DIR / pkg_name
    logger.info("[auto-fetch] Cloning %s from %s", pkg_name, url)
    subprocess.run(["git", "clone", url, str(pkg_dir)], check=True)
    logger.info("[auto-fetch] Checking out %s @ %s", pkg_name, commit[:12])
    subprocess.run(["git", "-C", str(pkg_dir), "checkout", commit], check=True)
    return pkg_dir


def find_ext_pkg_dir(pkg_name: str) -> Path:
    """Find external_packages/<pkg> source directory by fuzzy name match.

    If the package is not found locally, auto-fetches it from packages.yaml.
    """
    EXT_PKG_DIR.mkdir(parents=True, exist_ok=True)
    norm = normalize(pkg_name)
    for d in EXT_PKG_DIR.iterdir():
        if d.is_dir() and normalize(d.name) == norm:
            return d
    # Not found locally — try auto-fetch
    fetched = _fetch_single_package(pkg_name)
    if fetched and fetched.is_dir():
        return fetched
    available = [d.name for d in EXT_PKG_DIR.iterdir() if d.is_dir()]
    raise FileNotFoundError(
        f"External package directory not found for '{pkg_name}'\n"
        f"Available: {available}"
    )


def load_global_config(config_path: str | None) -> dict:
    """Load global config from YAML file. Returns empty dict if path is None."""
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        logger.warning("Config file not found: %s", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_task_config(task_name: str) -> dict:
    """Load task config.json for the given task."""
    config_path = PROJECT_ROOT / "tasks" / task_name / "config.json"
    if not config_path.exists():
        logger.error("Task config not found: %s", config_path)
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


def apply_baseline(test_cmds: list[dict], baseline_config: dict) -> list[dict]:
    """Return modified test_cmds with baseline cmd substitution.

    If baseline_config has "labels", only matching test_cmds get replaced.
    If "labels" is absent, ALL test_cmds get their cmd replaced.
    """
    result = copy.deepcopy(test_cmds)
    bl_cmd = baseline_config["cmd"]
    labels = baseline_config.get("labels")  # None = replace all
    for entry in result:
        if labels is None or entry.get("label") in labels:
            entry["cmd"] = bl_cmd
    return result


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------

def generate_def_file(pkg_config: dict, pkg_dir: Path, data_root: str | Path = "") -> str:
    """Generate an Apptainer definition file from pkg config."""
    base_image = pkg_config["base_image"]
    workdir = pkg_config.get("workdir", "/app")
    install_cmds = pkg_config.get("install_cmds", [])
    env = pkg_config.get("env", {})

    pkg_workdir = f"{workdir}/{pkg_dir.name}"
    files_section = f"    {pkg_dir.resolve()} {workdir}/{pkg_dir.name}\n"
    for ef in pkg_config.get("extra_files", []):
        src_str = expand_path_template(ef["src"], data_root)
        src = Path(src_str).expanduser().resolve()
        if src.exists():
            files_section += f"    {src} {ef['dst']}\n"
        else:
            logger.warning("extra_files src not found, skipping: %s", src)
    post_section = "\n".join(f"    {line}" for line in install_cmds)
    env_section = "\n".join(f"    export {k}={v}" for k, v in env.items())

    return f"""\
Bootstrap: docker
From: {base_image}

%files
{files_section}
%post
    cd {pkg_workdir}
{post_section}

%environment
{env_section}

%runscript
    exec bash "$@"
"""


def generate_dockerfile(
    pkg_config: dict,
    pkg_dir: Path,
    docker_extra_files: list[dict] | None = None,
) -> str:
    """Generate a Dockerfile from pkg config."""
    base_image = pkg_config["base_image"]
    workdir = pkg_config.get("workdir", "/app")
    pkg_workdir = f"{workdir.rstrip('/')}/{pkg_dir.name}"
    install_cmds = pkg_config.get("install_cmds", [])
    env = pkg_config.get("env", {})
    docker_extra_files = docker_extra_files or []

    lines = ["# syntax=docker/dockerfile:1.4", f"FROM {base_image}"]
    lines.append(f"COPY {pkg_dir.name} {pkg_workdir}")
    for ef in docker_extra_files:
        lines.append(f"COPY --from={ef['context_name']} {ef['copy_src']} {ef['dst']}")
    lines.append(f"WORKDIR {pkg_workdir}")
    for name in PASSTHROUGH_ENV_VARS:
        lines.append(f"ARG {name}")
    for cmd in install_cmds:
        lines.extend(docker_run_instruction_lines(cmd))
    for k, v in env.items():
        lines.append(f"ENV {k}={v}")
    lines.append('ENTRYPOINT ["bash"]')
    return "\n".join(lines) + "\n"


def _build_apptainer_sandbox(
    pkg_config: dict,
    pkg_dir: Path,
    output_sif: Path,
    build_env: dict,
    data_root: str | Path = "",
    force: bool = False,
) -> None:
    """Build an Apptainer SIF via sandbox mode (works in unprivileged containers).

    Steps:
      1. Pull the base docker image into a writable sandbox directory
      2. Copy package files and extra_files into the sandbox
      3. Run install_cmds inside the sandbox with ``apptainer exec --writable``
      4. Convert the sandbox to a SIF file
    """
    import shutil as _shutil
    import tempfile as _tempfile

    base_image = pkg_config["base_image"]
    workdir = pkg_config.get("workdir", "/app")
    install_cmds = pkg_config.get("install_cmds", [])

    sandbox_dir = Path(_tempfile.mkdtemp(prefix="mlsbench-sandbox-", dir=build_env.get("TMPDIR", "/tmp")))
    logger.info("[sandbox] Building in %s", sandbox_dir)

    try:
        # 1. Pull base image as sandbox
        logger.info("[sandbox] Pulling base image: %s", base_image)
        run_cmd(["apptainer", "build", "--sandbox", str(sandbox_dir / "root"),
                 f"docker://{base_image}"], env=build_env)

        sandbox_root = sandbox_dir / "root"

        # 2. Copy package files into sandbox
        dest = sandbox_root / workdir.lstrip("/")
        dest.mkdir(parents=True, exist_ok=True)
        pkg_dest = dest / pkg_dir.name
        if pkg_dest.exists():
            _shutil.rmtree(pkg_dest)
        _shutil.copytree(pkg_dir, pkg_dest)
        logger.info("[sandbox] Copied %s -> %s", pkg_dir, pkg_dest)

        # Copy extra_files
        for ef in pkg_config.get("extra_files", []):
            src_str = expand_path_template(ef["src"], data_root)
            src = Path(src_str).expanduser().resolve()
            if src.exists():
                ef_dest = sandbox_root / ef["dst"].lstrip("/")
                ef_dest.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    if ef_dest.exists():
                        _shutil.rmtree(ef_dest)
                    _shutil.copytree(src, ef_dest)
                else:
                    _shutil.copy2(src, ef_dest)
                logger.info("[sandbox] Copied extra_file %s -> %s", src, ef_dest)

        # 3. Fix apt cache permissions from host side before exec.
        #    The _apt user owns /var/cache/apt/archives/partial with mode 700;
        #    even with --fakeroot the UID mapping may not cover _apt.
        apt_archives = sandbox_root / "var" / "cache" / "apt" / "archives"
        if apt_archives.exists():
            subprocess.run(["chmod", "-R", "777", str(apt_archives)])

        # 4. Run install commands inside sandbox using --fakeroot.
        #    Unlike ``apptainer build --fakeroot`` (which mounts /proc and
        #    requires SYS_ADMIN), ``apptainer exec --writable --fakeroot``
        #    only uses user-namespace UID remapping via /etc/subuid and works
        #    in unprivileged K8s pods.  This makes groupadd etc. work.
        env_vars = pkg_config.get("env", {})
        env_flags = []
        for k, v in env_vars.items():
            env_flags.extend(["--env", f"{k}={v}"])
        for cmd in install_cmds:
            logger.info("[sandbox] Running: %s", cmd)
            run_cmd(["apptainer", "exec", "--writable", "--fakeroot",
                     "--no-home", "--no-mount", "home",
                     *env_flags,
                     "--pwd", f"{workdir}/{pkg_dir.name}",
                     str(sandbox_root), "bash", "-c", cmd], env=build_env)

        # 5. Convert sandbox to SIF
        if output_sif.exists() and force:
            output_sif.unlink()
        logger.info("[sandbox] Converting sandbox to SIF: %s", output_sif)
        run_cmd(["apptainer", "build", str(output_sif), str(sandbox_root)], env=build_env)

    finally:
        # Clean up sandbox
        _shutil.rmtree(sandbox_dir, ignore_errors=True)


def cmd_build(args):
    pkg_name = args.pkg_name
    config, stem = load_pkg_config(pkg_name)
    pkg_dir = find_ext_pkg_dir(pkg_name)

    global_config = load_global_config(getattr(args, "config", None))
    container_runtime = global_config.get("container_runtime", "apptainer")
    platform = global_config.get("platform", "")
    data_root = global_config.get("data_root", str(PROJECT_ROOT / "vendor" / "data"))

    if container_runtime != "local" and "base_image" not in config:
        logger.error("pkg config '%s.json' must have 'base_image'.", stem)
        sys.exit(1)

    # If extra_files reference {data_root}, the corresponding data_deps must be
    # prepared BEFORE the image build (the build step copies them into the
    # image).  Skip preps whose 'prepare' script needs the image to already
    # exist; those still run post-build below.
    if config.get("extra_files") and config.get("data_deps") and not args.dry_run:
        extra_srcs = " ".join(ef.get("src", "") for ef in config.get("extra_files", []))
        if "{data_root}" in extra_srcs:
            pre_build_deps = []
            for dep in config.get("data_deps", []):
                host = expand_path_template(dep.get("host_path", ""), data_root)
                if host and any(host in expand_path_template(ef.get("src", ""), data_root)
                                for ef in config.get("extra_files", [])):
                    pre_build_deps.append(dep)
            if pre_build_deps:
                logger.info("Preparing pre-build data dependencies for '%s'...", stem)
                pre_cfg = {**config, "data_deps": pre_build_deps}
                prepare_data_for_package(stem, pre_cfg, data_root, global_config=global_config)

    if container_runtime == "local":
        if args.dry_run:
            logger.info("Dry run: local build would prepare %s in-place at %s", stem, pkg_dir)
        else:
            build_local_package(stem, config, pkg_dir, global_config, force=args.force)
            logger.info("Local package prepared: %s", pkg_dir)
    elif container_runtime == "docker":
        # Docker build
        docker_extra_files = resolve_docker_extra_files(config, data_root=data_root)
        dockerfile_content = generate_dockerfile(
            config,
            pkg_dir,
            docker_extra_files=docker_extra_files,
        )

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        dockerfile_path = IMAGES_DIR / f"{stem}.Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        logger.info("Dockerfile written to: %s", dockerfile_path)
        logger.info("--- Dockerfile ---\n%s", dockerfile_content)

        if args.dry_run:
            logger.info("Dry run: skipping build.")
            return

        tag = docker_image_tag(stem)
        context_dir = pkg_dir.parent
        build_cmd = ["docker", "build"]
        if platform:
            build_cmd.extend(["--platform", platform])
        for name, value in iter_passthrough_env_vars():
            build_cmd.extend(["--build-arg", f"{name}={value}"])
        for ef in docker_extra_files:
            build_cmd.extend(["--build-context", f"{ef['context_name']}={ef['context_path']}"])
        build_cmd.extend([
            "-t", tag,
            "-f", str(dockerfile_path),
            str(context_dir),
        ])

        run_cmd(build_cmd)
        logger.info("Docker image built: %s", tag)
    else:
        # Apptainer build
        def_content = generate_def_file(config, pkg_dir, data_root=data_root)

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        output_sif = IMAGES_DIR / f"{stem}.sif"
        def_path = IMAGES_DIR / f"{stem}.def"
        def_path.write_text(def_content)
        logger.info("Apptainer definition written to: %s", def_path)
        logger.info("--- Definition file ---\n%s", def_content)

        if args.dry_run:
            logger.info("Dry run: skipping build.")
            return

        build_cmd = get_apptainer_build_cmd()
        if args.force:
            build_cmd.append("--force")
        build_cmd.extend([str(output_sif), str(def_path)])

        # Use scratch-local tmpdir to avoid filling up /tmp on large images
        build_env = os.environ.copy()
        tmpdir = IMAGES_DIR / "tmp"
        cachedir = IMAGES_DIR / "cache"
        tmpdir.mkdir(parents=True, exist_ok=True)
        cachedir.mkdir(parents=True, exist_ok=True)
        build_env["TMPDIR"] = str(tmpdir)
        build_env["APPTAINER_TMPDIR"] = str(tmpdir)
        build_env["APPTAINER_CACHEDIR"] = str(cachedir)
        build_env["SINGULARITY_CACHEDIR"] = str(cachedir)
        logger.info("Using TMPDIR: %s", tmpdir)
        logger.info("Using APPTAINER_CACHEDIR: %s", cachedir)

        result = subprocess.run(build_cmd, env=build_env)
        if result.returncode != 0:
            logger.warning("Fakeroot build failed (exit %d). Trying sandbox fallback...", result.returncode)
            _build_apptainer_sandbox(config, pkg_dir, output_sif, build_env, data_root,
                                     force=getattr(args, "force", False))
        logger.info("Image built: %s", output_sif)

    # Prepare data dependencies after successful image build
    if config.get("data_deps") and not args.dry_run:
        logger.info("Preparing data dependencies for '%s'...", stem)
        prepare_data_for_package(stem, config, data_root, global_config=global_config)


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

def cmd_run(args):
    pkg_name = args.pkg_name
    config, stem = load_pkg_config(pkg_name)
    pkg_dir = find_ext_pkg_dir(pkg_name)
    workdir = args.workdir or config.get("workdir", "/app")

    global_config = load_global_config(getattr(args, "config", None))
    container_runtime = global_config.get("container_runtime", "apptainer")

    # use_cuda: global config overrides per-package config
    if "use_cuda" in global_config:
        use_cuda = bool(global_config["use_cuda"])
    else:
        use_cuda = config.get("use_cuda", False)
    platform = global_config.get("platform", "")

    task_mount = workdir.rstrip("/") + "/_task"

    if container_runtime == "docker":
        tag = docker_image_tag(stem)
        # Check if Docker image exists
        check = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            logger.error(
                "Docker image not found: %s\n"
                "Run 'mlsbench build %s' first.",
                tag, pkg_name,
            )
            sys.exit(1)

        container_cmd = ["docker", "run", "--rm", "--entrypoint", ""]
        if platform:
            container_cmd.extend(["--platform", platform])
        if use_cuda:
            container_cmd.extend(["--gpus", "all"])

        # Inject env vars from pkg config
        for k, v in config.get("env", {}).items():
            container_cmd.extend(["-e", f"{k}={v}"])
        for name, value in iter_passthrough_env_vars():
            container_cmd.extend(["-e", f"{name}={value}"])

        # Bind mounts
        pkg_workdir = f"{workdir}/{pkg_dir.name}"
        container_cmd.extend(["-v", f"{pkg_dir.resolve()}:{pkg_workdir}"])
        logger.info("Binding %s -> %s", pkg_dir.name, pkg_workdir)

        if args.task:
            task_dir = PROJECT_ROOT / "tasks" / args.task
            if not task_dir.exists():
                logger.error("Task directory not found: %s", task_dir)
                sys.exit(1)
            container_cmd.extend(["-v", f"{task_dir.resolve()}:{task_mount}"])
            logger.info("Binding task '%s' -> %s", args.task, task_mount)

        # Config-level data bind (with template expansion)
        data_root = global_config.get(
            "data_root", str(PROJECT_ROOT / "vendor" / "data"))
        for db in resolve_data_binds(config, data_root):
            container_cmd.extend(["-v", db])
            logger.info("Data bind: %s", db)

        for extra in args.bind:
            container_cmd.extend(["-v", extra])

        container_cmd.extend(["-w", pkg_workdir])
        container_cmd.append(tag)
    else:
        image = IMAGES_DIR / f"{stem}.sif"
        if not image.exists():
            logger.error(
                "Image not found: %s\n"
                "Run 'mlsbench build %s' first (on a node with internet).",
                image, pkg_name,
            )
            sys.exit(1)

        container_cmd = ["apptainer", "exec"]
        if use_cuda:
            container_cmd.append("--nv")
        if config.get("writable_tmpfs", False):
            container_cmd.append("--writable-tmpfs")
        if config.get("no_home", False):
            container_cmd.append("--no-home")

        # Inject env vars from pkg config (e.g. MUJOCO_GL, PYOPENGL_PLATFORM)
        env_pairs = [f"{k}={v}" for k, v in config.get("env", {}).items()]
        if env_pairs:
            container_cmd.extend(["--env", ",".join(env_pairs)])

        binds: list[str] = []

        # Bind external package source -> workdir/pkg_name (live code overlay)
        pkg_workdir = f"{workdir}/{pkg_dir.name}"
        binds.append(f"{pkg_dir.resolve()}:{pkg_workdir}")
        logger.info("Binding %s -> %s", pkg_dir.name, pkg_workdir)

        # Optional: bind task directory at workdir/_task
        if args.task:
            task_dir = PROJECT_ROOT / "tasks" / args.task
            if not task_dir.exists():
                logger.error("Task directory not found: %s", task_dir)
                sys.exit(1)
            binds.append(f"{task_dir.resolve()}:{task_mount}")
            logger.info("Binding task '%s' -> %s", args.task, task_mount)

        # Config-level data bind (with template expansion)
        data_root = global_config.get(
            "data_root", str(PROJECT_ROOT / "vendor" / "data"))
        for db in resolve_data_binds(config, data_root):
            binds.append(db)
            logger.info("Data bind: %s", db)

        for extra in args.bind:
            binds.append(extra)

        container_cmd.extend(["--bind", ",".join(binds)])
        container_cmd.extend(["--pwd", pkg_workdir])
        container_cmd.append(str(image))

    # Resolve the script to run
    if args.task:
        task_dir = PROJECT_ROOT / "tasks" / args.task
        script = task_dir / args.run_cmd
        if not script.exists():
            logger.error("Script not found: %s", script)
            sys.exit(1)
        container_cmd.extend(["bash", f"{task_mount}/{args.run_cmd}"])
    else:
        script = Path(args.run_cmd)
        if not script.is_absolute():
            script = Path.cwd() / script
        if not script.exists():
            logger.error("Script not found: %s", script)
            sys.exit(1)
        container_cmd.extend(["bash", str(script)])

    if args.dry_run:
        print(" \\\n    ".join(container_cmd))
        return

    logger.info("Running '%s' in %s container", args.run_cmd, stem)
    result = run_cmd(container_cmd, check=False)
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------

def _find_completed_groups_from_logs(
    task_name: str,
    baseline_name: str,
    seeds: list[int],
    logs_dir: Path,
) -> set[int]:
    """Scan log directories to find groups where all (label, seed) pairs succeeded.

    Looks at ``logs_dir/<task>/<baseline>/<timestamp>/group_<N>/exit_codes.txt``
    for exit code 0 on every required ``<label>_s<seed>``.  Returns the set of
    group numbers that are fully complete (across all seeds) in at least one
    timestamped run.
    """
    baseline_logs = logs_dir / task_name / baseline_name
    if not baseline_logs.exists():
        return set()

    # Collect the best exit-code info across all timestamped runs (latest wins)
    # group_key -> label -> seed -> success?
    from collections import defaultdict
    group_success: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    for ts_dir in sorted(baseline_logs.iterdir()):
        if not ts_dir.is_dir():
            continue
        for group_dir in ts_dir.iterdir():
            if not group_dir.is_dir():
                continue
            m = re.match(r"group_(\d+)", group_dir.name)
            if not m:
                continue
            group_key = int(m.group(1))

            ec_file = group_dir / "exit_codes.txt"
            if not ec_file.exists():
                continue

            for line in ec_file.read_text().splitlines():
                line = line.strip()
                if ":" not in line:
                    continue
                label_seed, code_str = line.rsplit(":", 1)
                try:
                    code = int(code_str)
                except ValueError:
                    continue
                if code != 0:
                    continue
                # Parse label and seed from "label_s42"
                m2 = re.match(r"(.+)_s(\d+)$", label_seed)
                if not m2:
                    continue
                label = m2.group(1)
                seed = int(m2.group(2))
                if seed in seeds:
                    group_success[group_key][label].add(seed)

    # A group is complete if every label in that group has all seeds succeeded
    seeds_set = set(seeds)
    completed = set()
    for group_key, label_map in group_success.items():
        # All labels in this group must have all seeds
        if label_map and all(s == seeds_set for s in label_map.values()):
            completed.add(group_key)

    return completed


def _filter_test_cmds_for_resume(
    test_cmds: list[dict],
    baseline_name: str,
    task_name: str,
    seeds: list[int],
    logs_dir: Path,
    leaderboard,
) -> list[dict]:
    """Remove test_cmd groups that already completed (via logs or leaderboard)."""
    from collections import defaultdict
    from mlsbench.agent.leaderboard import Leaderboard

    # --- Source 1: check logs for exit code 0 ---
    completed_groups = _find_completed_groups_from_logs(
        task_name, baseline_name, seeds, logs_dir,
    )

    # --- Source 2: check leaderboard for existing metrics ---
    # Note: tools.test() always writes is_final=false (final flag is only set
    # via agent submit()). Baselines never call submit, so we accept any
    # row matching this baseline regardless of the is_final flag.
    records = leaderboard.all_records()
    model_name = f"baseline:{baseline_name}"
    baseline_records = [
        r for r in records
        if r.get("model") == model_name
        and r.get("seed") != "mean"
    ]

    def label_has_metrics_for_seed(label: str, seed: int) -> bool:
        for rec in baseline_records:
            rec_seed = rec.get("seed")
            try:
                if int(float(rec_seed)) != seed:
                    continue
            except (ValueError, TypeError):
                continue
            for key, val in rec.items():
                if key in Leaderboard.META_COLS or key.startswith("elapsed_"):
                    continue
                if key.endswith(f"_{label}") and val not in ("", None):
                    return True
        return False

    def label_complete_in_leaderboard(label: str) -> bool:
        return all(label_has_metrics_for_seed(label, s) for s in seeds)

    # --- Group test_cmds ---
    groups: dict[int, list[dict]] = defaultdict(list)
    auto_group = 10000
    for entry in test_cmds:
        g = entry.get("group")
        if g is None:
            groups[auto_group].append(entry)
            auto_group += 1
        else:
            groups[g].append(entry)

    # --- Filter: skip groups whose metrics actually landed in the leaderboard
    #     (the authoritative signal). Logs with exit=0 alone are NOT enough,
    #     because a SLURM job can succeed (exit=0) yet still fail to populate
    #     the leaderboard (parser miss, manual SLURM submission outside
    #     mlsbench, mid-run leaderboard wipe, etc.). When that happens, the
    #     user should be able to recover by re-running, which only works if
    #     --resume insists on leaderboard evidence.
    kept = []
    for group_key in sorted(groups.keys()):
        entries = groups[group_key]
        labels = [e.get("label", "test") for e in entries]

        if all(label_complete_in_leaderboard(l) for l in labels):
            via_logs = " (and exit=0 in logs)" if group_key in completed_groups else ""
            logger.info(
                "[resume] Skipping group %d (labels %s) — metrics in leaderboard%s",
                group_key, labels, via_logs,
            )
            continue

        incomplete = [l for l in labels if not label_complete_in_leaderboard(l)]
        if group_key in completed_groups:
            logger.warning(
                "[resume] Re-running group %d (labels %s) — exit=0 in old logs "
                "but leaderboard is missing metrics for: %s",
                group_key, labels, incomplete,
            )
        else:
            logger.info(
                "[resume] Running group %d — incomplete labels: %s",
                group_key, incomplete,
            )
        kept.extend(entries)

    return kept


def _run_single_baseline(
    baseline_name: str,
    baseline_config: dict,
    task_name: str,
    task_config: dict,
    run_config: dict,
    seeds: list[int],
    parser,
    leaderboard,
    rigorous: bool,
    workspace_root: Path | None = None,
    resume: bool = False,
) -> tuple[str, str]:
    """Run a single baseline in its own isolated workspace. Returns (name, result)."""
    if workspace_root is None:
        workspace_root = PROJECT_ROOT / "vendor" / "workspace"

    logger.info("=" * 60)
    logger.info("Running baseline: %s (task: %s)", baseline_name, task_name)
    logger.info("=" * 60)

    modified_task_config = copy.deepcopy(task_config)
    # Apply cmd if the baseline provides a different training script
    if baseline_config.get("cmd"):
        test_cmds = task_config.get("test_cmds", [])
        modified_cmds = apply_baseline(test_cmds, baseline_config)
        modified_task_config["test_cmds"] = modified_cmds

    # Resume: skip groups that already have complete results
    if resume:
        slurm_cfg = run_config.get("slurm", {})
        logs_dir = PROJECT_ROOT / slurm_cfg.get("logs_dir", "results")
        original_cmds = modified_task_config.get("test_cmds", [])
        filtered = _filter_test_cmds_for_resume(
            original_cmds, baseline_name, task_name, seeds,
            logs_dir, leaderboard,
        )
        if not filtered:
            logger.info("[resume] Baseline '%s' fully complete, skipping.", baseline_name)
            return baseline_name, "All groups already complete (resumed)."
        modified_task_config["test_cmds"] = filtered

    from mlsbench.agent.tools import WorkspaceTools
    # Forward use_cuda / platform overrides from global config
    use_cuda_override = run_config.get("use_cuda")          # None if absent
    if use_cuda_override is not None:
        use_cuda_override = bool(use_cuda_override)
    tools = WorkspaceTools(
        task_name=task_name,
        config_task=modified_task_config,
        config_edit=task_config.get("files", []),
        workspace_root=workspace_root,
        project_root=PROJECT_ROOT,
        max_tests=1,
        model_name=f"baseline:{baseline_name}",
        parser=parser,
        leaderboard=leaderboard,
        save_path=run_config.get("save_path", ""),
        seeds=seeds,
        slurm_config=run_config.get("slurm"),
        exp_name=baseline_name,
        container_runtime=run_config.get("container_runtime", "apptainer"),
        use_cuda=use_cuda_override,
        platform=run_config.get("platform", ""),
        gpu_devices=run_config.get("gpu_devices", ""),
        compute_scale=run_config.get("compute_scale", 1.0),
        global_config=run_config,
        allow_web_search=run_config.get("allow_web_search", False),
        tavily_api_key=(run_config.get("providers", {}).get("tavily", {}) or {}).get("api_key", ""),
        max_web_credits=run_config.get("max_web_credits", 20),
        extra_env=baseline_config.get("env"),
    )

    _setup_workspace_for_baseline(task_name, task_config, tools, PROJECT_ROOT)

    if baseline_config.get("edit_ops"):
        _apply_rigorous_edit(task_name, task_config, baseline_config, tools)

    logger.info("Running test (is_final=True) for baseline '%s'", baseline_name)
    result = tools.test(is_final=True)
    if tools._last_test_had_failures:
        raise RuntimeError(f"Baseline '{baseline_name}' had failing test commands.\n{result}")
    return baseline_name, result


def cmd_baseline(args):
    """Run baseline(s) for a task through the standard test pipeline."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    task_name = args.task
    global_config = load_global_config(args.config)
    task_config = load_task_config(task_name)

    # Filter test_cmds by --group and/or --label
    if args.group is not None or args.label is not None:
        original_cmds = task_config.get("test_cmds", [])
        filtered_cmds = original_cmds
        if args.group is not None:
            allowed_groups = set(args.group)
            filtered_cmds = [e for e in filtered_cmds if e.get("group") in allowed_groups]
        if args.label is not None:
            allowed_labels = set(args.label)
            filtered_cmds = [e for e in filtered_cmds if e.get("label") in allowed_labels]
        if not filtered_cmds:
            avail_groups = sorted({e.get("group") for e in original_cmds if "group" in e})
            avail_labels = sorted({e.get("label") for e in original_cmds if "label" in e})
            logger.error(
                "No test_cmds match filters (group=%s, label=%s). "
                "Available groups: %s, labels: %s",
                args.group, args.label, avail_groups, avail_labels,
            )
            sys.exit(1)
        logger.info(
            "Filtering test_cmds: %d/%d (group=%s, label=%s)",
            len(filtered_cmds), len(original_cmds), args.group, args.label,
        )
        task_config["test_cmds"] = filtered_cmds

    baselines = task_config.get("baselines", {})
    if not baselines:
        logger.error("No baselines defined in tasks/%s/config.json", task_name)
        sys.exit(1)

    # Determine which baselines to run (--name is action="append", so a list)
    if args.name:
        missing = [n for n in args.name if n not in baselines]
        if missing:
            logger.error(
                "Baseline(s) not found: %s. Available: %s",
                missing, list(baselines.keys()),
            )
            sys.exit(1)
        baselines_to_run = {n: baselines[n] for n in args.name}
    else:
        baselines_to_run = baselines

    # Override seeds: CLI --seed > task config > global config
    seeds = task_config.get("seeds") or global_config.get("seeds", [42])
    if args.seed is not None:
        seeds = [args.seed]

    # Merge seed override into config for WorkspaceTools
    run_config = copy.deepcopy(global_config)
    run_config["seeds"] = seeds

    # Setup shared infrastructure
    from mlsbench.agent.leaderboard import Leaderboard
    leaderboard = Leaderboard(PROJECT_ROOT / "tasks" / task_name / "leaderboard.csv")

    from mlsbench.agent.parsers import load_parser
    parser = load_parser(task_name, PROJECT_ROOT)

    rigorous = task_config.get("rigorous_codebase", False)

    # Timestamped workspace root so concurrent/subsequent runs don't collide
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    workspace_root = PROJECT_ROOT / "vendor" / "workspace" / f"{timestamp}_{os.getpid()}"

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(baselines_to_run)) as executor:
        futures = {
            executor.submit(
                _run_single_baseline,
                name, config, task_name, task_config,
                run_config, seeds, parser, leaderboard, rigorous,
                workspace_root,
                resume=getattr(args, "resume", False),
            ): name
            for name, config in baselines_to_run.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                baseline_name, result = future.result()
                logger.info("Baseline '%s' result:\n%s", baseline_name, result)
            except Exception:
                failures.append(name)
                logger.exception("Baseline '%s' failed", name)

    logger.info("All baselines complete.")
    if failures:
        sys.exit(1)


def _setup_workspace_for_baseline(
    task_name: str,
    task_config: dict,
    tools,
    project_root: Path,
) -> None:
    """Setup workspace for baseline run: copy packages and apply pre-edits."""
    import importlib.util
    import shutil

    workspace_task_dir = tools.workspace_task_dir
    ext_dir = project_root / "vendor" / "external_packages"
    pkg_configs_dir = project_root / "vendor" / "pkg_configs"

    # Collect all packages from FULL task config (not filtered by --label/--group)
    # so that mid_edit ops referencing other packages still work
    full_task_config = load_task_config(task_name)
    seen_norm: set[str] = set()
    all_packages: list[str] = []
    for entry in full_task_config.get("test_cmds", []):
        pkg = entry.get("package")
        if pkg:
            norm = normalize(pkg)
            if norm not in seen_norm:
                seen_norm.add(norm)
                all_packages.append(pkg)

    # Always start fresh for baseline workspaces
    if workspace_task_dir.exists():
        logger.info("[workspace] Removing stale workspace: %s", workspace_task_dir)
        shutil.rmtree(workspace_task_dir)
    workspace_task_dir.mkdir(parents=True, exist_ok=True)

    for pkg in all_packages:
        dst = workspace_task_dir / pkg

        src = None
        norm = normalize(pkg)
        for d in ext_dir.iterdir():
            if d.is_dir() and normalize(d.name) == norm:
                src = d
                break
        if src is None:
            raise FileNotFoundError(f"External package '{pkg}' not found in {ext_dir}")

        logger.info("[workspace] Copying %s -> %s", src, dst)
        shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns('.git'))

    # Apply pre_edit ops (package-level patches)
    # Use full config so pre_edits for ALL packages are applied even when
    # test_cmds are filtered by --label/--group
    pre_edit_ops = _load_pre_edit_ops(full_task_config, pkg_configs_dir)
    if pre_edit_ops:
        logger.info("[workspace] Applying %d pre_edit operation(s)", len(pre_edit_ops))
        tools.apply_pre_edit(pre_edit_ops)

    # Apply mid_edit ops (task-specific workspace setup)
    from mlsbench.agent.tools import load_mid_edit_ops
    mid_edit_ops = load_mid_edit_ops(task_name, project_root / "tasks")
    if mid_edit_ops:
        logger.info("[workspace] Applying %d mid_edit operation(s)", len(mid_edit_ops))
        tools.apply_pre_edit(mid_edit_ops)


def _apply_rigorous_edit(
    task_name: str,
    task_config: dict,
    baseline_config: dict,
    tools,
) -> None:
    """Load and apply edit operations from a baseline's edit file.

    The edit file is a Python module that exports an OPS list (same convention
    as pre_edit.py). Supported ops:

        replace  – replace start_line..end_line (inclusive, 1-indexed) with content
        insert   – insert content after after_line (1-indexed)
        delete   – delete start_line..end_line (inclusive, 1-indexed)

    Ops should be ordered bottom-to-top (highest line numbers first) so that
    earlier ops do not shift line numbers for later ops.
    """
    import importlib.util

    edit_ops_rel = baseline_config.get("edit_ops")
    if not edit_ops_rel:
        logger.info("[rigorous] Baseline has no 'edit_ops' — skipping code edit")
        return

    task_dir = PROJECT_ROOT / "tasks" / task_name
    edit_file = task_dir / edit_ops_rel
    if not edit_file.exists():
        raise FileNotFoundError(f"Baseline edit file not found: {edit_file}")

    spec = importlib.util.spec_from_file_location("rigorous_edit", edit_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ops = getattr(module, "OPS", [])

    if not ops:
        raise ValueError(f"No OPS found in {edit_file}")

    for op in ops:
        filename = op["file"]
        ws_path = tools._resolve_workspace_path(filename)
        op_type = op["op"]

        if not ws_path.exists():
            raise FileNotFoundError(f"Workspace file not found: {ws_path}")

        lines = ws_path.read_text().splitlines(keepends=True)

        if op_type == "replace":
            start = op["start_line"]
            end = op["end_line"]
            content = op["content"]
            if not content.endswith("\n"):
                content += "\n"
            content_lines = content.splitlines(keepends=True)
            lines[start - 1 : end] = content_lines
            logger.info(
                "[rigorous] Replaced %s lines %d–%d (%d → %d lines)",
                filename, start, end, end - start + 1, len(content_lines),
            )

        elif op_type == "insert":
            after_line = op["after_line"]
            content = op["content"]
            if not content.endswith("\n"):
                content += "\n"
            content_lines = content.splitlines(keepends=True)
            for i, cl in enumerate(content_lines):
                lines.insert(after_line + i, cl)
            logger.info(
                "[rigorous] Inserted %d lines after line %d in %s",
                len(content_lines), after_line, filename,
            )

        elif op_type == "delete":
            start = op["start_line"]
            end = op.get("end_line", start)
            del lines[start - 1 : end]
            logger.info("[rigorous] Deleted lines %d–%d in %s", start, end, filename)

        else:
            raise ValueError(f"Unknown rigorous edit op: {op_type}")

        ws_path.write_text("".join(lines))


def _load_pre_edit_ops(task_config: dict, pkg_configs_dir: Path) -> list[dict]:
    """Load pre_edit ops from pkg_configs/<Pkg>.pre_edit.py for each package."""
    from mlsbench.agent.tools import load_pre_edit_ops
    return load_pre_edit_ops(task_config, pkg_configs_dir)


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def cmd_fetch(args):
    """Clone or update external packages from packages.yaml."""
    if not PACKAGES_YAML.exists():
        logger.error("packages.yaml not found at %s", PACKAGES_YAML)
        sys.exit(1)

    with open(PACKAGES_YAML) as f:
        pkg_registry = yaml.safe_load(f) or {}

    packages = pkg_registry.get("packages", {})
    if not packages:
        logger.error("No packages defined in packages.yaml")
        sys.exit(1)

    # Filter to single package if --name is given
    if args.name:
        if args.name not in packages:
            logger.error(
                "Package '%s' not found in packages.yaml. Available: %s",
                args.name, list(packages.keys()),
            )
            sys.exit(1)
        packages = {args.name: packages[args.name]}

    EXT_PKG_DIR.mkdir(parents=True, exist_ok=True)

    for pkg_name, info in packages.items():
        pkg_dir = EXT_PKG_DIR / pkg_name

        if _is_local_registry_package(info):
            if pkg_dir.exists():
                logger.info("[fetch] %s is a local scaffold package, keeping %s", pkg_name, pkg_dir)
            else:
                logger.info("[fetch] Creating local package stub for %s", pkg_name)
            _ensure_local_package_stub(pkg_name)
            continue

        url = info["url"]
        commit = info["commit"]

        if pkg_dir.exists():
            logger.info("[fetch] %s already exists, updating...", pkg_name)
            run_cmd(["git", "-C", str(pkg_dir), "fetch", "origin"], check=False)
        else:
            logger.info("[fetch] Cloning %s from %s", pkg_name, url)
            run_cmd(["git", "clone", url, str(pkg_dir)])

        logger.info("[fetch] Checking out %s @ %s", pkg_name, commit[:12])
        run_cmd(["git", "-C", str(pkg_dir), "checkout", commit])

    logger.info("All packages fetched.")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def prepare_data_for_package(
    pkg_name: str,
    pkg_config: dict,
    data_root: str,
    *,
    list_only: bool = False,
    global_config: dict | None = None,
) -> None:
    """Prepare data dependencies for a single package.

    Checks each data_dep: skips if already present, runs prepare script if
    missing.  When *list_only* is True, prints status without downloading.
    """
    deps = pkg_config.get("data_deps", [])
    if not deps:
        return

    print(f"\n{'=' * 50}")
    print(f"  Package: {pkg_name} ({len(deps)} data dep(s))")
    print(f"{'=' * 50}")

    for dep in deps:
        name = dep.get("name", "?")
        host_path = expand_path_template(dep.get("host_path", ""), data_root)
        container_path = dep.get("container_path", "")
        prepare = dep.get("prepare", "")
        desc = dep.get("description", "")

        exists = Path(host_path).exists() and any(Path(host_path).iterdir()) if Path(host_path).exists() else False
        status = "READY" if exists else "MISSING"

        print(f"\n  [{status}] {name}")
        if desc:
            print(f"    Description: {desc}")
        print(f"    Host path:      {host_path}")
        print(f"    Container path: {container_path}")
        if prepare:
            print(f"    Prepare script: {prepare}")

        if list_only:
            continue

        if exists and not prepare:
            logger.info("Data '%s' already exists at %s, skipping", name, host_path)
            continue

        if exists and prepare:
            logger.info("Data '%s' exists at %s, re-running prepare script for completeness", name, host_path)

        if not exists and not prepare:
            logger.warning(
                "Data '%s' is MISSING at %s and no prepare script is defined. "
                "Please download/prepare it manually.",
                name, host_path,
            )
            continue

        # Run the prepare script
        script_path = PROJECT_ROOT / prepare
        if not script_path.exists():
            logger.error("Prepare script not found: %s", script_path)
            continue

        logger.info("Running prepare script for '%s': %s", name, prepare)
        runtime = (global_config or {}).get("container_runtime", "")
        if runtime in ("apptainer", "docker"):
            cmd = [sys.executable, str(script_path), "--data-root", str(data_root)]
        else:
            # Use "python" so conda run resolves it via the env's PATH,
            # not sys.executable which may be a different Python version.
            cmd = wrap_with_conda(
                ["python", str(script_path), "--data-root", str(data_root)],
                global_config,
                pkg_name=pkg_name,
            )
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("Prepare script failed for '%s' (exit code %d)", name, result.returncode)
        else:
            logger.info("Data '%s' prepared successfully", name)


def cmd_data(args):
    """Download/prepare large datasets declared in pkg_config data_deps."""
    global_config = load_global_config(args.config)
    data_root = global_config.get("data_root", str(PROJECT_ROOT / "vendor" / "data"))

    # Collect all packages with data_deps
    packages_with_deps: list[tuple[str, dict]] = []
    for d in PKG_CONFIGS_DIR.iterdir():
        if not d.is_dir():
            continue
        cfg_file = d / "config.json"
        if not cfg_file.is_file():
            continue
        with open(cfg_file) as f:
            config = json.load(f)
        deps = config.get("data_deps", [])
        if deps:
            packages_with_deps.append((d.name, config))

    if args.pkg_name:
        norm = normalize(args.pkg_name)
        packages_with_deps = [
            (name, cfg) for name, cfg in packages_with_deps
            if normalize(name) == norm
        ]
        if not packages_with_deps:
            logger.error(
                "No data_deps found for package '%s'. "
                "Packages with data_deps: %s",
                args.pkg_name,
                [name for name, _ in packages_with_deps] if packages_with_deps
                else "(none found — check pkg_configs)",
            )
            sys.exit(1)

    if not packages_with_deps:
        logger.info("No packages with data_deps found.")
        return

    for pkg_name, config in packages_with_deps:
        prepare_data_for_package(
            pkg_name,
            config,
            data_root,
            list_only=args.list_deps,
            global_config=global_config,
        )

    print()
    logger.info("Data command complete.")


# ---------------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------------

def cmd_agent(args):
    """Run an LLM agent against a task."""
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: config.yaml not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        global_config = yaml.safe_load(f)

    # Inject runtime options
    global_config["model"] = args.model
    global_config["mode"] = args.mode
    global_config["verbose"] = getattr(args, "verbose", False)
    global_config["allow_web_search"] = getattr(args, "allow_web_search", False)
    global_config["max_web_credits"] = getattr(args, "max_web_credits", 20)
    global_config["extra_context"] = getattr(args, "extra_context", None)
    global_config["disable_budget_check"] = getattr(args, "disable_budget_check", False)

    # Collect OpenEvolve-specific runtime knobs if the user provided them
    oe_knobs = dict(global_config.get("openevolve") or {})
    if getattr(args, "openevolve_iterations", None) is not None:
        oe_knobs["iterations"] = args.openevolve_iterations
    if getattr(args, "openevolve_config", None):
        oe_knobs["config_path"] = args.openevolve_config
    if oe_knobs:
        global_config["openevolve"] = oe_knobs

    # Collect Discover-specific runtime knobs
    disc_knobs = dict(global_config.get("discover") or {})
    if getattr(args, "discover_iterations", None) is not None:
        disc_knobs["iterations"] = args.discover_iterations
    if getattr(args, "discover_config", None):
        disc_knobs["config_path"] = args.discover_config
    if getattr(args, "discover_tasks", None):
        disc_knobs["extra_tasks"] = [
            t.strip() for t in args.discover_tasks.split(",") if t.strip()
        ]
    if getattr(args, "discover_val_tasks", None):
        disc_knobs["val_tasks"] = list(args.discover_val_tasks)
    if disc_knobs:
        global_config["discover"] = disc_knobs
    if getattr(args, "tinker_api_key", None):
        global_config.setdefault("providers", {}).setdefault("tinker", {})["api_key"] = args.tinker_api_key

    agent_type = getattr(args, "agent_type", "interactive")
    if agent_type == "openevolve":
        from mlsbench.agent.openevolve_agent import OpenEvolveAgent
        agent_cls = OpenEvolveAgent
    elif agent_type == "discover":
        from mlsbench.agent.discover_agent import DiscoverAgent
        agent_cls = DiscoverAgent
    else:
        from mlsbench.agent.interactive import InteractiveAgent
        agent_cls = InteractiveAgent

    agent_kwargs = {}
    if agent_type == "discover":
        agent_kwargs["val_tasks"] = getattr(args, "discover_val_tasks", [])

    agent = agent_cls(
        task_name=args.task,
        global_config=global_config,
        workspace_root=args.workspace,
        **agent_kwargs,
    )

    summary = agent.run(resume=getattr(args, "resume", False))
    print(f"\n[done] Summary: {summary}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MLS-Bench: unified CLI for agent, baseline, build, run, and fetch",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- agent --
    p_agent = subparsers.add_parser("agent", help="Run an LLM agent against a task")
    p_agent.add_argument("task", help="Task name (directory under tasks/)")
    p_agent.add_argument(
        "--model", required=True,
        help="Model to use, e.g. claude-sonnet-4-6, deepseek-chat, gpt-4o",
    )
    p_agent.add_argument(
        "--config", default=None,
        help="Path to config.yaml (default: configs/config.yaml in project root)",
    )
    p_agent.add_argument(
        "--workspace", default=None,
        help="Override workspace root directory",
    )
    p_agent.add_argument(
        "--mode", choices=["sci", "eng"], default="sci",
        help="Agent mode: 'sci' (default) for novel algorithmic contributions, "
             "'eng' for engineering-only optimizations (control baseline)",
    )
    p_agent.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full untruncated logs (thinking, prompts, diffs, results)",
    )
    p_agent.add_argument(
        "--resume", action="store_true",
        help="Resume a previously interrupted agent run from its log",
    )
    p_agent.add_argument(
        "--agent-type", choices=["interactive", "openevolve", "discover"], default="interactive",
        help="Agent implementation: 'interactive' (chat-style tool-use, default), "
             "'openevolve' (evolutionary code search via vendored OpenEvolve), or "
             "'discover' (test-time RL LoRA via ttt-discover + Tinker)",
    )
    p_agent.add_argument(
        "--openevolve-iterations", type=int, default=None,
        help="Number of OpenEvolve iterations (agent-type openevolve only)",
    )
    p_agent.add_argument(
        "--openevolve-config", default=None,
        help="Path to an OpenEvolve YAML config (agent-type openevolve only)",
    )
    p_agent.add_argument(
        "--discover-iterations", type=int, default=None,
        help="Number of ttt-discover training epochs (agent-type discover only)",
    )
    p_agent.add_argument(
        "--discover-config", default=None,
        help="Path to a discover YAML config (agent-type discover only)",
    )
    p_agent.add_argument(
        "--discover-tasks", default=None,
        help="Comma-separated additional task ids for multi-task training "
             "(v1.1 — not yet implemented)",
    )
    p_agent.add_argument(
        "--discover-val-tasks", type=parse_csv_list, default=[],
        help="Comma-separated task ids used as a Discover validation pool "
             "(no LoRA optimizer step on validation rollouts)",
    )
    p_agent.add_argument(
        "--tinker-api-key", default=None,
        help="Tinker API key for ttt-discover (falls back to TINKER_API_KEY env)",
    )
    p_agent.add_argument(
        "--allow-web-search", action="store_true",
        help="Expose a client-side web_search(query, max_results) tool to the agent. "
             "Backed by Tavily. Requires providers.tavily.api_key in the config "
             "(or TAVILY_API_KEY env var). Each call counts against the step budget.",
    )
    p_agent.add_argument(
        "--max-web-credits", type=int, default=20,
        help="Cap on total Tavily credits per run, shared between web_search and "
             "web_extract (default 20). Pricing: basic search=1, advanced search=2, "
             "basic extract=1/URL, advanced extract=2/URL. Set to 0 for unlimited. "
             "Has no effect unless --allow-web-search is also passed.",
    )
    p_agent.add_argument(
        "--extra-context", default=None,
        help="Inject extra context at the top of the initial prompt. Either "
             "'baseline'/'theory' (looks up context_packs/science_priors_10tasks_v3/"
             "contexts/<task>__{baseline_derivation,deep_theory}_context.md and "
             "errors out if missing), or a path to any context file (e.g. a "
             "research proposal the agent should implement).",
    )
    p_agent.add_argument(
        "--disable-budget-check", action="store_true",
        help="Skip budget_check.py before each test, and remove the parameter-scale "
             "reminder from the system prompt (both sci and eng modes). Tagged in "
             "the leaderboard model name as ':nobudget'.",
    )
    p_agent.set_defaults(func=cmd_agent)

    # -- baseline --
    p_baseline = subparsers.add_parser(
        "baseline",
        help="Run baseline method(s) for a task and record results to leaderboard",
    )
    p_baseline.add_argument("task", help="Task name (directory under tasks/)")
    p_baseline.add_argument(
        "--name", default=None, action="append",
        help="Baseline name to run (from config.json baselines). Can be "
             "repeated: --name foo --name bar. If omitted, runs ALL baselines.",
    )
    p_baseline.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to global config.yaml (default: configs/config.yaml)",
    )
    p_baseline.add_argument(
        "--seed", type=int, default=None,
        help="Override seed (default: use seeds from config.yaml)",
    )
    p_baseline.add_argument(
        "--group", type=int, action="append", default=None,
        help="Only run test_cmds in the specified group(s). "
             "Can be repeated: --group 0 --group 2. "
             "If omitted, runs all groups.",
    )
    p_baseline.add_argument(
        "--label", action="append", default=None,
        help="Only run test_cmds with the specified label(s). "
             "Can be repeated: --label NS --label Burgers. "
             "If omitted, runs all labels.",
    )
    p_baseline.add_argument(
        "--resume", action="store_true", default=False,
        help="Skip test_cmd groups whose metrics already exist in the leaderboard. "
             "Useful for resuming after partial failures (e.g. training done, eval missing).",
    )
    p_baseline.set_defaults(func=cmd_baseline)

    # -- fetch --
    p_fetch = subparsers.add_parser(
        "fetch",
        help="Clone/update external packages from packages.yaml",
    )
    p_fetch.add_argument(
        "--name", default=None,
        help="Fetch a single package by name (default: fetch all)",
    )
    p_fetch.set_defaults(func=cmd_fetch)

    # -- build --
    p_build = subparsers.add_parser(
        "build",
        help="Prepare a package for the configured runtime (Apptainer, Docker, or local/conda)",
    )
    p_build.add_argument("pkg_name", help="Package name (matched against pkg_configs/<pkg>.json)")
    p_build.add_argument("--force", action="store_true", help="Overwrite existing image (Apptainer only)")
    p_build.add_argument("--dry-run", action="store_true", help="Generate definition file only, skip build")
    p_build.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to global config.yaml (default: configs/config.yaml)",
    )
    p_build.set_defaults(func=cmd_build)

    # -- run --
    p_run = subparsers.add_parser(
        "run",
        help="Run a script in a pre-built container image (offline OK)",
    )
    p_run.add_argument("pkg_name", help="Package name (matched against pkg_configs/<pkg>.json)")
    p_run.add_argument(
        "--run-cmd", required=True,
        help="Script to run. Resolved relative to the task dir if --task is given, "
             "otherwise must be an absolute or CWD-relative path.",
    )
    p_run.add_argument(
        "--task", default=None,
        help="Task name (directory under tasks/); bind-mounts task dir at "
             "workdir/_task and resolves --run-cmd relative to it.",
    )
    p_run.add_argument(
        "--bind", action="append", default=[],
        help="Extra bind mounts (HOST:CONTAINER)",
    )
    p_run.add_argument(
        "--workdir", default=None,
        help="Override working directory inside container (default: from pkg config)",
    )
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="Print command without executing",
    )
    p_run.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to global config.yaml (default: configs/config.yaml)",
    )
    p_run.set_defaults(func=cmd_run)

    # -- data --
    p_data = subparsers.add_parser(
        "data",
        help="Download/prepare large datasets declared in pkg_config data_deps",
    )
    p_data.add_argument(
        "pkg_name", nargs="?", default=None,
        help="Package name to prepare data for (default: all packages with data_deps)",
    )
    p_data.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to global config.yaml (default: configs/config.yaml)",
    )
    p_data.add_argument(
        "--list", action="store_true", dest="list_deps",
        help="List data dependencies without downloading",
    )
    p_data.set_defaults(func=cmd_data)

    # -- score --
    from mlsbench.scoring import register_score_subcommand
    register_score_subcommand(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

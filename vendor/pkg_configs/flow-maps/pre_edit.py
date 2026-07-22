"""Patches applied to the flow-maps workspace copy for MLS-Bench.

- Adds configs.cifar10_bench (short run; FID once at the final training step). The
  template supports ``FLOWMAPS_UNET_SIZE`` (``small`` / ``medium`` / ``large``) for
  EDM2 width/depth presets (see ``cifar10_bench.py.template``).
- Disables W&B network I/O when WANDB_DISABLED is set.
- Extends logging.log_metrics to print TEST_METRICS (same convention as
  cv-meanflow-training: fid=, best_fid=) after the final-step FID.
- learn.py: after training, always call ``compute_fid_on_the_fly`` + ``TEST_METRICS``
  once (bypasses pmap/last-step misses in ``log_metrics``).
- fid_utils.fid_from_stats: pure NumPy trace/dot (avoids JAX/numpy mixing issues).
- fid_utils.download: resolve default data/ relative to package (flow-maps/data),
  not cwd; optional FID_INCEPTION_WEIGHTS / INCEPTION_V3_FID_PICKLE for offline.
- Ensure flow-maps/data/ exists (upstream has no data/). Inception weights are baked
  at image build under ``/opt/mlsbench/inception_v3_weights_fid.pickle`` (not under
  ``/workspace/flow-maps``, so the workspace bind mount does not hide them). CIFAR-10
  TFDS lives under ``/data/tfds``. ``FID_INCEPTION_WEIGHTS`` matches that path.

Ops run in list order. **learn.py:** apply the WANDB block (higher line numbers,
179–186) **before** the train-loop FINAL block (95–96) so line numbers in
``train_loop`` are unchanged when the second replace runs.

**logging.py:** replace the step-gate block (300–317) before the FID/TEST_METRICS
block (329–341); the first edit adds one line, and the second range matches the
shifted file (comment + ``for`` … ``metrics`` through line 341 of the *updated*
file).

**fid_utils.py:** ``fid_from_stats`` (72–78) and ``download`` (87–116) keep the
same line counts as upstream, so ranges stay valid when applied in that order.
"""

from pathlib import Path

_PKG = Path(__file__).resolve().parent
_CIFAR10_BENCH = (_PKG / "cifar10_bench.py.template").read_text()

_WANDB_PATCH = """    # Set up weights and biases tracking
    _wandb_kw = dict(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        name=cfg.logging.wandb_name,
        config=cfg.to_dict(),
    )
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        _wandb_kw["mode"] = "disabled"
    print("Setting up wandb.")
    wandb.init(**_wandb_kw)"""

_FINAL_PATCH = """    # MLS-Bench: guaranteed final FID + TEST_METRICS (multi-GPU can miss last step in log_metrics)
    if hasattr(cfg.logging, "fid_freq") and cfg.logging.fid_freq > 0:
        try:
            _sc = getattr(cfg.logging, "fid_n_steps_flow", 8)
            _nsl = list(_sc) if isinstance(_sc, (list, tuple)) else [_sc]
            _metrics = {}
            for _ns in _nsl:
                _fs, prng_key = logging.compute_fid_on_the_fly(
                    cfg,
                    statics,
                    train_state,
                    prng_key,
                    n_samples=getattr(cfg.logging, "fid_n_samples", 10000),
                    batch_size=getattr(cfg.logging, "fid_batch_size", 256),
                    n_steps_flow=_ns,
                )
                _metrics[f"fid_{int(_ns)}_steps"] = _fs
            _fv = float("nan")
            _pn_out = 1
            for _ns in _nsl:
                _k = f"fid_{int(_ns)}_steps"
                _pn_out = int(_ns)
                if _k in _metrics:
                    _val = _metrics[_k]
                    _fv = float(_val.item() if hasattr(_val, "item") else _val)
                    break
            _psd = cfg.training.psd_type
            _psd_s = "none" if _psd is None else str(_psd)
            _fid_s = f"{_fv:.2f}" if _fv == _fv else "nan"
            print(
                f"TEST_METRICS: fid={_fid_s}, best_fid={_fid_s}, "
                f"loss_type={cfg.training.loss_type}, psd_type={_psd_s}, "
                f"n_flow={_pn_out}, seed={cfg.training.seed}",
                flush=True,
            )
        except Exception as _e:
            print(f"Warning: MLS-Bench final FID failed: {_e}", flush=True)
    # dump one final time
    logging.save_state(train_state, cfg)"""

_FID_FROM_STATS_PATCH = """def fid_from_stats(mu1, sigma1, mu2, sigma2):
    diff = np.asarray(mu1 - mu2, dtype=np.float64)
    offset = np.eye(sigma1.shape[0]) * 1e-6
    covmean, _ = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)
    covmean = np.real(covmean)
    mat = np.asarray(sigma1 + sigma2 - 2 * covmean, dtype=np.float64)
    return float(np.dot(diff, diff) + np.trace(mat))"""

# Upstream logging.py lines 300–317: add Python-int step for FID gating (JAX
# DeviceArray breaks ``(step % fid_freq) == 0`` in ``if``).
_LOG_METRICS_STEP_GATE = """    step = dist_utils.safe_index(cfg, train_state.step)
    _step_i = int(jax.device_get(step))
    learning_rate = statics.schedule(step)

    # Standard metrics
    metrics = {
        f"loss": loss_value,
        f"grad": compute_grad_norm(grads),
        f"learning_rate": learning_rate,
        f"step_time": step_time,
    }

    # Compute FID on-the-fly if enabled and at the right frequency
    if (
        hasattr(cfg.logging, "fid_freq")
        and cfg.logging.fid_freq > 0
        and (_step_i % int(cfg.logging.fid_freq)) == 0
        and _step_i > 0
    ):"""

_LOGGING_FID_PATCH = """            # Compute FID for each step count
            for n_steps in n_steps_list:
                fid_score, prng_key = compute_fid_on_the_fly(
                    cfg,
                    statics,
                    train_state,
                    prng_key,
                    n_samples=getattr(cfg.logging, "fid_n_samples", 10000),
                    batch_size=getattr(cfg.logging, "fid_batch_size", 256),
                    n_steps_flow=n_steps,
                )
                # Log with step-specific key (must match n_steps, not env alone)
                metrics[f"fid_{int(n_steps)}_steps"] = fid_score
            _fv = float("nan")
            _pn_out = 1
            for n_steps in n_steps_list:
                k = f"fid_{int(n_steps)}_steps"
                _pn_out = int(n_steps)
                if k in metrics:
                    val = metrics[k]
                    try:
                        _fv = float(val.item() if hasattr(val, "item") else val)
                    except (TypeError, ValueError):
                        _fv = float("nan")
                    break
            _psd = cfg.training.psd_type
            _psd_s = "none" if _psd is None else str(_psd)
            _best = _fv
            _fid_s = f"{_fv:.2f}" if _fv == _fv else "nan"
            _best_s = f"{_best:.2f}" if _best == _best else "nan"
            print(
                f"TEST_METRICS: fid={_fid_s}, best_fid={_best_s}, "
                f"loss_type={cfg.training.loss_type}, psd_type={_psd_s}, "
                f"n_flow={_pn_out}, seed={cfg.training.seed}",
                flush=True,
            )"""

_FID_DOWNLOAD_PATCH = '''def download(url, ckpt_dir="data"):
    name = url[url.rfind("/") + 1 : url.rfind("?")]
    _env = os.environ.get("FID_INCEPTION_WEIGHTS") or os.environ.get(
        "INCEPTION_V3_FID_PICKLE"
    )
    if _env and os.path.isfile(_env):
        return os.path.abspath(_env)
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    elif ckpt_dir == "data":
        _here = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.normpath(os.path.join(_here, "..", "..", "data"))
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: "{url[:url.rfind("?")]}" to {ckpt_file}')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + ".temp")
        with open(ckpt_file_temp, "wb") as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("An error occured while downloading, please try again.")
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file'''

OPS = [
    {
        "op": "create",
        "file": "flow-maps/data/.mlsbench_keep",
        "content": "# MLS-Bench: ensures data/ exists for inception weights bind mount\n",
    },
    {
        "op": "create",
        "file": "flow-maps/py/configs/cifar10_bench.py",
        "content": _CIFAR10_BENCH,
    },
    {
        "op": "replace",
        "file": "flow-maps/py/launchers/learn.py",
        "start_line": 179,
        "end_line": 186,
        "content": _WANDB_PATCH,
    },
    {
        "op": "replace",
        "file": "flow-maps/py/launchers/learn.py",
        "start_line": 95,
        "end_line": 96,
        "content": _FINAL_PATCH,
    },
    {
        "op": "replace",
        "file": "flow-maps/py/common/logging.py",
        "start_line": 300,
        "end_line": 317,
        "content": _LOG_METRICS_STEP_GATE,
    },
    {
        "op": "replace",
        "file": "flow-maps/py/common/logging.py",
        "start_line": 329,
        "end_line": 341,
        "content": _LOGGING_FID_PATCH,
    },
    {
        "op": "replace",
        "file": "flow-maps/py/common/fid_utils.py",
        "start_line": 72,
        "end_line": 78,
        "content": _FID_FROM_STATS_PATCH,
    },
    {
        "op": "replace",
        "file": "flow-maps/py/common/fid_utils.py",
        "start_line": 87,
        "end_line": 116,
        "content": _FID_DOWNLOAD_PATCH,
    },
]

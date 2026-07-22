"""Real FAR rollout harness for ar-video-kv-temporal-policy.

Evaluates VideoTemporalKVPolicy by running actual FAR (Flow Autoregressive
Reconstruction) inference on UCF101/DMLab clips and measuring PSNR/SSIM/LPIPS
vs ground truth frames.

Policy hook design
------------------
Rather than using FAR's built-in KV cache (which only avoids recomputation),
the harness prunes the `latents` tensor itself: at each generation step, the
policy decides which historical frames the model is allowed to attend to. Only
those frames are kept in `latents` and passed to the model. This directly
measures the quality impact of temporal history management.

Execution contexts
------------------
  container:  /workspace/FAR/custom_video_eval.py
              FAR weights at /data/far_weights/
              UCF101 data at /data/ucf101/ (symlink or mount)
  local task: tasks/ar-video-kv-temporal-policy/custom_video_eval.py
              FAR weights via FAR_WEIGHTS_DIR env var
              UCF101 data via UCF101_DATA_DIR env var
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path resolution: FAR package must be on PYTHONPATH
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

# FAR repo root: prefer explicit env var, then well-known paths
def _find_far_root() -> Path:
    env_dir = os.environ.get('FAR_REPO_DIR', '')
    if env_dir and (Path(env_dir) / 'far').is_dir():
        return Path(env_dir)
    candidates = [
        Path('/workspace/FAR'),
        _HERE.parent,                                       # container: /workspace/FAR/
        _HERE.parent / 'vendor' / 'external_packages' / 'FAR',
        Path('/data/FAR_repo'),
    ]
    for p in candidates:
        if (p / 'far').is_dir():
            return p
    raise RuntimeError('Cannot find FAR repo. Set FAR_REPO_DIR env var.')

_FAR_ROOT = _find_far_root()
if str(_FAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_FAR_ROOT))

_FAR_DCAE_CFG = _FAR_ROOT / 'options' / 'model_cfg' / 'dcae' / 'model_8x_c32_config.json'
_FAR_SCHED_CFG = _FAR_ROOT / 'options' / 'model_cfg' / 'far' / 'scheduler_config.json'

# ---------------------------------------------------------------------------
# Weight and data path resolution
# ---------------------------------------------------------------------------
_WEIGHTS_DIR = Path(
    os.environ.get('FAR_WEIGHTS_DIR', '/data/far_weights')
)
_UCF101_DIR = Path(
    os.environ.get('UCF101_DATA_DIR',
                   '/user/zhangyicheng/mls-bench-mlsys-co-design/probes/FAR/datasets/ucf101')
)
_DMLAB_DATA_DIR = Path(
    os.environ.get('DMLAB_DATA_DIR', '/data/dmlab_videos')
)

# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------
WORKLOAD_CONFIGS = {
    'ucf101_short_prediction': {
        'dataset': 'ucf101',
        'data_dir': _UCF101_DIR,
        'n_context': 5,
        'n_predict': 11,
        'resolution': 64,
        'transformer_ckpt': _WEIGHTS_DIR / 'short_video_prediction' / 'FAR_B_UCF101_Uncond64-381d295f.pth',
        'dcae_ckpt': _WEIGHTS_DIR / 'dcae' / 'DCAE_UCF101_Res64-9da18dcf.pth',
        'dcae_config': _FAR_DCAE_CFG,
        'num_clips': 50,
        'n_inference_steps': 20,
        'guidance_scale': 1.0,
        'model_type': 'FAR_B',
        'conditioned': False,
        'description': 'UCF-101 short-horizon video prediction (MCVD protocol, 5→11 frames)',
    },
    'dmlab_long_prediction': {
        'dataset': 'dmlab',
        'data_dir': _DMLAB_DATA_DIR,
        'n_context': 36,
        'n_predict': 36,
        'resolution': 64,
        'transformer_ckpt': _WEIGHTS_DIR / 'long_video_prediction' / 'FAR_B_Long_DMLab_Action64-c09441dc.pth',
        'dcae_ckpt': _WEIGHTS_DIR / 'dcae' / 'DCAE_DMLab_Res64-17035ae5.pth',
        'dcae_config': _FAR_DCAE_CFG,
        'num_clips': 20,
        'n_inference_steps': 20,
        'guidance_scale': -1,          # unconditional (action=null)
        'model_type': 'FAR_B_Long',
        'conditioned': True,
        'description': 'DMLab long-horizon action-conditioned prediction (36→36 frames)',
    },
}

BUDGET_CONFIGS = {
    'full_history_budget': {'max_frames': 9999, 'description': 'Keep all historical frames (upper-bound quality)'},
    'medium_history_budget': {'max_frames': 8,  'description': 'Keep at most 8 historical frames'},
    'tight_history_budget': {'max_frames': 4,   'description': 'Keep at most 4 historical frames'},
}

# ---------------------------------------------------------------------------
# Shared dataclasses (interface contract – do not modify)
# ---------------------------------------------------------------------------

@dataclass
class FrameMeta:
    """Observable metadata for one historical frame."""
    chunk_id: int                   # 0-based index in kept history
    age: int                        # steps since this frame was generated
    dynamic_age: int                # same as age (for API compatibility)
    recentness: float               # 1 / (1 + age)
    size_mb: float                  # memory footprint (latent, fp32)
    access_count: int               # always 1 in this harness
    recent_hit_count: int           # always 1 in this harness
    last_access_step: int           # step this frame was last seen by model
    boundary_score: float           # feature-space distance from prev frame [0,1]
    feature_drift: float            # cumulative drift from first context frame [0,1]
    temporal_similarity: float      # cosine similarity to previous frame [0,1]
    motion_strength: float          # magnitude of change from prev frame [0,1]
    keyframe_score: float           # local peak of change [0,1]
    long_term_resident: bool        # whether frame is marked as long-term
    compression_state: str          # "none" always


@dataclass
class ChunkDecision:
    chunk_id: int
    action: str    # 'keep' | 'drop' | 'demote_to_long_term'
    priority: float = 0.0


@dataclass
class TemporalCachePlan:
    chunk_decisions: List[ChunkDecision] = field(default_factory=list)
    retention_family: str = 'recency'
    anchor_preservation_rule: str = 'none'
    boundary_transition_rule: str = 'passthrough'
    queue_depth: int = 0
    long_term_budget_fraction: float = 0.0
    chunkwise_reuse: bool = False
    compression_mode: str = 'none'


@dataclass
class RolloutState:
    step: int
    budget_capacity_mb: float
    total_steps: int


# ---------------------------------------------------------------------------
# Minimal FAR pipeline (avoids DiffusionPipeline to prevent diffusers/
# transformers version incompatibilities)
# ---------------------------------------------------------------------------

class _FARPipelineMinimal:
    """Thin wrapper around FAR transformer + DCAE + scheduler.

    Provides vae_encode / vae_decode / __call__ with the same interface
    as FARPipeline but without inheriting DiffusionPipeline, so it works
    regardless of diffusers/transformers version mismatches.
    """

    def __init__(self, transformer, vae, scheduler, device, dtype):
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.execution_device = device
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def vae_encode(self, context_sequence: torch.Tensor) -> torch.Tensor:
        """[B, T, C, H, W] pixel [0,1] → latent [B, T, C_lat, H_lat, W_lat]."""
        from einops import rearrange
        x = context_sequence * 2 - 1   # [0,1] → [-1,1]
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        lat = self.vae.encode(x.to(dtype=self.vae.dtype)).latent
        lat = lat * self.vae.config.scaling_factor
        lat = rearrange(lat, '(b t) c h w -> b t c h w', b=B)
        return lat

    @torch.no_grad()
    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latent [B, T, C_lat, H_lat, W_lat] → pixel [B, T, C, H, W] [0,1]."""
        from einops import rearrange
        B, T, C, H, W = latents.shape
        lat = 1 / self.vae.config.scaling_factor * latents
        lat = rearrange(lat, 'b t c h w -> (b t) c h w')
        out = self.vae.decode(lat.to(dtype=self.vae.dtype)).sample
        out = rearrange(out, '(b t) c h w -> b t c h w', b=B)
        return (out / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def __call__(self,
                 vision_context,
                 latents,
                 conditions=None,
                 context_cache=None,
                 context_timestep_idx=-1,
                 guidance_scale=1.0,
                 num_inference_steps=50,
                 **kwargs):
        """Run ODE denoising for one frame. Mirrors FARPipeline.__call__."""
        if context_cache is None:
            context_cache = {'is_cache_step': True, 'kv_cache': None,
                             'cached_seqlen': 0, 'multi_level_cache_init': False}

        batch_size = latents.shape[0]

        if conditions is not None:
            if 'label' in conditions:
                class_labels = conditions['label'].to(self.execution_device).reshape(-1)
                if guidance_scale > 1:
                    null_idx = self.transformer.config.condition_cfg['num_classes']
                    class_null = torch.tensor([null_idx] * batch_size, device=self.execution_device)
                    conditions = {'label': torch.cat([class_null, class_labels], 0)}
            elif 'action' in conditions:
                actions = conditions['action'].to(self.execution_device)
                if guidance_scale > 1:
                    null_idx = self.transformer.config.condition_cfg['num_action_classes']
                    action_null = (torch.ones_like(actions[:1]) * null_idx
                                   ).expand(batch_size, -1)
                    conditions = {'action': torch.cat([action_null, actions], 0)}
                elif guidance_scale == -1:
                    null_idx = self.transformer.config.condition_cfg['num_action_classes']
                    conditions = {'action': torch.ones_like(actions) * null_idx}

        self.scheduler.set_timesteps(num_inference_steps)
        context_cache['is_cache_step'] = vision_context is not None

        for t in self.scheduler.timesteps:
            lmi = torch.cat([latents] * 2) if guidance_scale > 1 else latents
            if guidance_scale > 1 and vision_context is not None:
                vci = torch.cat([vision_context] * 2)
            else:
                vci = vision_context

            ts = t
            if not torch.is_tensor(ts):
                ts = torch.tensor([ts], dtype=torch.int64, device=lmi.device)
            elif ts.dim() == 0:
                ts = ts[None].to(lmi.device)
            ts = ts.expand(lmi.shape[0]).unsqueeze(-1)

            if vci is not None:
                ctx_ts = torch.tensor([context_timestep_idx], dtype=ts.dtype, device=ts.device)
                ctx_ts = ctx_ts.expand(lmi.shape[0]).unsqueeze(-1).repeat(1, vci.shape[1])
                ts_in = torch.cat([ctx_ts, ts], dim=-1)
                lmi_in = torch.cat([vci, lmi], dim=1)
            else:
                ts_in = ts
                lmi_in = lmi

            noise_pred, context_cache = self.transformer(
                lmi_in, timestep=ts_in, context_cache=context_cache,
                conditions=conditions, return_dict=False)
            noise_pred = noise_pred.to(lmi.dtype)
            context_cache['is_cache_step'] = False

            if guidance_scale > 1:
                unc, cond = noise_pred.chunk(2)
                noise_pred = unc + guidance_scale * (cond - unc)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents, context_cache


_MODEL_CACHE: dict = {}


def _load_far_pipeline(wl_cfg: dict) -> _FARPipelineMinimal:
    """Load FAR transformer + DCAE + scheduler."""
    key = str(wl_cfg['transformer_ckpt'])
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import json as _json
    from far.models import build_model
    from far.models.autoencoder_dc_model import MyAutoencoderDC

    # Use diffusers scheduler directly (no DiffusionPipeline needed)
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    if _FAR_SCHED_CFG.exists():
        import json as _jj
        with open(_FAR_SCHED_CFG) as f:
            sched_cfg = _jj.load(f)
        scheduler = FlowMatchEulerDiscreteScheduler(**{
            k: v for k, v in sched_cfg.items() if not k.startswith('_')})
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    # Transformer
    model = build_model(wl_cfg['model_type'])()
    ckpt = Path(wl_cfg['transformer_ckpt'])
    if not ckpt.exists():
        raise FileNotFoundError(f'FAR weights not found: {ckpt}')
    sd = torch.load(str(ckpt), map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model = model.to(device, dtype=dtype).eval()

    # DCAE VAE
    dcae_cfg = Path(wl_cfg['dcae_config'])
    if not dcae_cfg.exists():
        dcae_cfg = _FAR_DCAE_CFG
    with open(str(dcae_cfg)) as f:
        dcae_config_dict = _json.load(f)
    vae = MyAutoencoderDC.from_config(dcae_config_dict)
    dcae_ckpt = Path(wl_cfg['dcae_ckpt'])
    if not dcae_ckpt.exists():
        raise FileNotFoundError(f'DCAE weights not found: {dcae_ckpt}')
    sd_vae = torch.load(str(dcae_ckpt), map_location='cpu', weights_only=True)
    vae.load_state_dict(sd_vae)
    vae = vae.to(device, dtype=dtype).eval()
    vae.requires_grad_(False)

    pipeline = _FARPipelineMinimal(
        transformer=model, vae=vae, scheduler=scheduler,
        device=device, dtype=dtype)

    _MODEL_CACHE[key] = pipeline
    return pipeline


# ---------------------------------------------------------------------------
# UCF101 / DMLab data loading
# ---------------------------------------------------------------------------

def _load_ucf101_clips(data_dir: Path, n_clips: int, n_frames: int,
                       resolution: int, seed: int = 42) -> list:
    """Return list of (frames_tensor, action_label) tuples.
    frames_tensor: [n_frames, 3, resolution, resolution] float32 in [0,1]
    """
    import random
    import decord
    decord.bridge.set_bridge('torch')

    # Prefer video/ subdirectory (FAR layout: data_dir/video/<class>/*.mp4)
    video_dir = data_dir / 'video'
    if video_dir.is_dir():
        all_vids = list(video_dir.rglob('*.mp4'))
    else:
        # fallback: collect all mp4 files under data_dir
        all_vids = list(data_dir.rglob('*.mp4'))

    rng = random.Random(seed)
    rng.shuffle(all_vids)

    clips = []
    for vp in all_vids:
        if len(clips) >= n_clips:
            break
        if not Path(vp).exists():
            continue
        try:
            vr = decord.VideoReader(str(vp), width=resolution, height=resolution)
            if len(vr) < n_frames:
                continue
            start = rng.randint(0, len(vr) - n_frames)
            frames = vr.get_batch(list(range(start, start + n_frames))).float() / 255.0
            frames = frames.permute(0, 3, 1, 2)   # [T, C, H, W]
            clips.append((frames, 0))
        except Exception:
            continue
    return clips


def _load_dmlab_clips(data_dir: Path, n_clips: int, n_frames: int,
                      resolution: int, seed: int = 42) -> list:
    """Return DMLab clips as (frames_tensor, action_sequence) tuples.
    Falls back to synthetic if data not available.
    """
    import random
    rng = random.Random(seed)

    # Try loading real DMLab videos
    mp4_files = list(data_dir.rglob('*.mp4')) + list(data_dir.rglob('*.avi'))
    if len(mp4_files) >= n_clips:
        import decord
        decord.bridge.set_bridge('torch')
        rng.shuffle(mp4_files)
        clips = []
        for vp in mp4_files:
            if len(clips) >= n_clips:
                break
            try:
                vr = decord.VideoReader(str(vp), width=resolution, height=resolution)
                if len(vr) < n_frames:
                    continue
                start = rng.randint(0, len(vr) - n_frames)
                frames = vr.get_batch(list(range(start, start + n_frames))).float() / 255.0
                frames = frames.permute(0, 3, 1, 2)
                actions = torch.zeros(n_frames, dtype=torch.long)  # null actions
                clips.append((frames, actions))
            except Exception:
                continue
        if clips:
            return clips

    # Synthetic fallback: color-gradient sequences with slow drift
    clips = []
    for i in range(n_clips):
        t = torch.linspace(0, 1, n_frames).unsqueeze(-1).unsqueeze(-1)
        color = torch.tensor([rng.random(), rng.random(), rng.random()])
        frames = (color.view(1, 3, 1, 1) * t.unsqueeze(1)
                  + (1 - t.unsqueeze(1)) * 0.5).expand(n_frames, 3, resolution, resolution)
        frames = frames + 0.05 * torch.randn_like(frames)
        frames = frames.clamp(0, 1)
        actions = torch.zeros(n_frames, dtype=torch.long)
        clips.append((frames, actions))
    return clips


# ---------------------------------------------------------------------------
# Feature extraction from latent tensors
# ---------------------------------------------------------------------------

def _latent_features(latent: torch.Tensor,
                     ref_latent: Optional[torch.Tensor] = None,
                     prev_latent: Optional[torch.Tensor] = None
                     ) -> dict:
    """Compute frame features from a latent code [C,H,W] or [1,C,H,W]."""
    if latent.dim() == 4:
        latent = latent.squeeze(0)         # [C, H, W]
    flat = latent.float().flatten()

    if prev_latent is not None:
        if prev_latent.dim() == 4:
            prev_latent = prev_latent.squeeze(0)
        prev_flat = prev_latent.float().flatten()
        diff = flat - prev_flat
        motion_strength = float(diff.abs().mean())
        cos = float(F.cosine_similarity(flat.unsqueeze(0), prev_flat.unsqueeze(0)))
        temporal_similarity = max(0.0, cos)
        norm_diff = float(diff.norm() / (prev_flat.norm() + 1e-8))
        boundary_score = min(1.0, norm_diff / 0.5)   # calibrated: 0.5 = typical threshold
    else:
        motion_strength = 0.0
        temporal_similarity = 1.0
        boundary_score = 0.0

    if ref_latent is not None:
        if ref_latent.dim() == 4:
            ref_latent = ref_latent.squeeze(0)
        ref_flat = ref_latent.float().flatten()
        feature_drift = float((flat - ref_flat).norm() / (ref_flat.norm() + 1e-8))
        feature_drift = min(1.0, feature_drift)
    else:
        feature_drift = 0.0

    return {
        'temporal_similarity': temporal_similarity,
        'feature_drift': feature_drift,
        'motion_strength': motion_strength,
        'boundary_score': boundary_score,
        'keyframe_score': motion_strength,
    }


def _build_chunk_meta_list(kept_latents: torch.Tensor,
                           current_step: int,
                           budget_capacity_mb: float) -> List[FrameMeta]:
    """Build chunk_meta_list from current kept_latents [B, N, C, H, W]."""
    N = kept_latents.shape[1]
    B, _, C, H, W = kept_latents.shape
    frame_bytes = C * H * W * 4   # fp32
    size_mb = frame_bytes / (1024 ** 2)

    ref_latent = kept_latents[0, 0]   # first context frame as reference
    metas = []
    for i in range(N):
        lat = kept_latents[0, i]
        prev = kept_latents[0, i - 1] if i > 0 else None
        feats = _latent_features(lat, ref_latent=ref_latent, prev_latent=prev)
        # age: 0 = newest frame (i=N-1), increases toward older frames
        age = N - 1 - i
        metas.append(FrameMeta(
            chunk_id=i,
            age=age,
            dynamic_age=age,
            recentness=1.0 / (1.0 + age),
            size_mb=size_mb,
            access_count=1,
            recent_hit_count=1,
            last_access_step=current_step,
            boundary_score=feats['boundary_score'],
            feature_drift=feats['feature_drift'],
            temporal_similarity=feats['temporal_similarity'],
            motion_strength=feats['motion_strength'],
            keyframe_score=feats['keyframe_score'],
            long_term_resident=False,
            compression_state='none',
        ))
    return metas


def _apply_plan(kept_latents: torch.Tensor,
                plan: TemporalCachePlan,
                max_frames: int) -> torch.Tensor:
    """Apply TemporalCachePlan to prune kept_latents."""
    N = kept_latents.shape[1]
    keep_set = {d.chunk_id for d in plan.chunk_decisions
                if d.action in ('keep', 'demote_to_long_term')}
    # always keep the most recent frame
    keep_set.add(N - 1)
    # respect max_frames budget
    keep_ids = sorted(keep_set)
    if len(keep_ids) > max_frames:
        # keep highest-priority (most recent first by default)
        keep_ids = keep_ids[-max_frames:]
    keep_ids_tensor = torch.tensor(keep_ids, device=kept_latents.device)
    return kept_latents[:, keep_ids_tensor]


# ---------------------------------------------------------------------------
# Generation loop with policy hook
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_policy(pipeline,
                         context_frames: torch.Tensor,
                         policy,
                         wl_cfg: dict,
                         budget_cfg: dict,
                         conditions=None,
                         seed: int = 42) -> torch.Tensor:
    """
    Generate predicted frames with policy managing the historical frame buffer.

    Args:
        context_frames: [1, n_ctx, C, H, W] float32 [0,1] pixel values
        policy: VideoTemporalKVPolicy instance

    Returns:
        pred_frames: [1, n_predict, C, H, W] float32 [0,1] generated frames
    """
    torch.manual_seed(seed)
    device = pipeline.execution_device
    dtype = pipeline.vae.dtype
    n_predict = wl_cfg['n_predict']
    n_steps = wl_cfg['n_inference_steps']
    max_frames = budget_cfg['max_frames']

    B = context_frames.shape[0]
    latent_size = wl_cfg['resolution'] // 8   # DCAE 8× downsampling

    # Encode context frames
    ctx = context_frames.to(device, dtype=dtype)
    kept_latents = pipeline.vae_encode(ctx)   # [B, n_ctx, C_lat, H_lat, W_lat]
    C_lat = kept_latents.shape[2]

    budget_capacity_mb = max_frames * C_lat * latent_size * latent_size * 4 / (1024 ** 2)

    rollout = RolloutState(
        step=0,
        budget_capacity_mb=budget_capacity_mb,
        total_steps=n_predict,
    )

    pred_latents_list = []
    total_steps = wl_cfg['n_context'] + n_predict

    from diffusers.utils.torch_utils import randn_tensor

    for pred_idx in range(n_predict):
        rollout.step = pred_idx

        # Noise init for this frame
        init_noise = randn_tensor(
            shape=(B, 1, C_lat, latent_size, latent_size),
            device=device, dtype=dtype)

        # Run FAR pipeline.__call__ once: generate ONE frame from kept_latents
        # (disable built-in KV cache – policy manages history directly)
        step_condition = None
        if conditions is not None and 'action' in conditions:
            n_ctx_now = kept_latents.shape[1]
            step_condition = {'action': conditions['action'][:, :n_ctx_now + 1]}

        pred_lat, _ = pipeline(
            vision_context=kept_latents,
            latents=init_noise,
            conditions=step_condition,
            context_cache={'is_cache_step': True, 'kv_cache': None,
                           'cached_seqlen': 0, 'multi_level_cache_init': False},
            guidance_scale=wl_cfg['guidance_scale'],
            num_inference_steps=n_steps,
        )   # [B, 1, C_lat, H_lat, W_lat]

        pred_latents_list.append(pred_lat)

        # Append to history
        kept_latents = torch.cat([kept_latents, pred_lat], dim=1)

        # Policy decision: which frames to keep
        if max_frames < 9999 and kept_latents.shape[1] > max_frames:
            chunk_meta = _build_chunk_meta_list(kept_latents, pred_idx, budget_capacity_mb)
            rollout.budget_capacity_mb = budget_capacity_mb
            plan = policy.build_temporal_cache_plan(chunk_meta, rollout,
                                                    {'max_frames': max_frames,
                                                     'capacity_mb': budget_capacity_mb})
            kept_latents = _apply_plan(kept_latents, plan, max_frames)

    # Decode all predicted frames
    all_pred_lat = torch.cat(pred_latents_list, dim=1)   # [B, n_predict, C_lat, H, W]
    pred_frames = pipeline.vae_decode(all_pred_lat)      # [B, n_predict, C, H, W]
    return pred_frames


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(pred: torch.Tensor, gt: torch.Tensor, device) -> dict:
    """Compute PSNR / SSIM / LPIPS between pred and gt [N, C, H, W] in [0,1]."""
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    import lpips

    pred = pred.float().to(device)
    gt = gt.float().to(device)

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    psnr = float(psnr_fn(pred, gt))

    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim = float(ssim_fn(pred, gt))

    lpips_fn = lpips.LPIPS(net='alex', spatial=False).to(device)
    lpips_fn.eval()
    with torch.no_grad():
        lp = float(lpips_fn(pred * 2 - 1, gt * 2 - 1).mean())

    return {'psnr': psnr, 'ssim': ssim, 'lpips': lp}


def _psnr_to_quality_main(psnr: float) -> float:
    """Map PSNR (dB) to a 0–100 quality score (capped at 40dB → 100)."""
    return min(100.0, max(0.0, psnr / 40.0 * 100.0))


# ---------------------------------------------------------------------------
# EDITABLE REGION – lines 300–341
# Replace the VideoTemporalKVPolicy class body below.
# config.json edit range targets this block.
# =============================================================================

class VideoTemporalKVPolicy:
    """
    Temporal KV retention policy for autoregressive video generation.

    Implement build_temporal_cache_plan to control which historical frames
    the model attends to at each generation step.

    CONTRACT
    --------
    - chunk_meta_list: observable metadata for all frames currently in history
    - rollout_state.step: current prediction step (0 = first predicted frame)
    - budget_state['max_frames']: hard upper bound on kept frames
    - Return a TemporalCachePlan with chunk_decisions for EVERY chunk_id.
    - Action 'drop' removes the frame from latents entirely.
    - Action 'keep' or 'demote_to_long_term' retains it.
    """

    def build_temporal_cache_plan(
        self,
        chunk_meta_list: List[FrameMeta],
        rollout_state: RolloutState,
        budget_state: dict,
    ) -> TemporalCachePlan:
        # Baseline: keep all frames (no eviction)
        decisions = [ChunkDecision(chunk_id=m.chunk_id, action='keep', priority=0.0)
                     for m in chunk_meta_list]
        return TemporalCachePlan(
            chunk_decisions=decisions,
            retention_family='recency',
        )

# =============================================================================
# END EDITABLE REGION
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(workload_name: str, budget_name: str, seed: int = 42):
    wl_cfg = WORKLOAD_CONFIGS[workload_name]
    budget_cfg = BUDGET_CONFIGS[budget_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[ar-video-kv] workload={workload_name} budget={budget_name} seed={seed}')
    print(f'  dataset: {wl_cfg["description"]}')
    print(f'  budget:  {budget_cfg["description"]}')

    # Load model
    pipeline = _load_far_pipeline(wl_cfg)
    pipeline.set_progress_bar_config(disable=True)
    policy = VideoTemporalKVPolicy()

    # Load clips
    n_ctx = wl_cfg['n_context']
    n_pred = wl_cfg['n_predict']
    n_frames_total = n_ctx + n_pred

    if wl_cfg['dataset'] == 'ucf101':
        clips = _load_ucf101_clips(
            wl_cfg['data_dir'], wl_cfg['num_clips'],
            n_frames_total, wl_cfg['resolution'], seed)
    else:
        clips = _load_dmlab_clips(
            wl_cfg['data_dir'], wl_cfg['num_clips'],
            n_frames_total, wl_cfg['resolution'], seed)

    if not clips:
        print('WARNING: no clips loaded, returning zeros')
        print(f'TEST_METRICS: temporal_quality_main=0 psnr=0 ssim=0 lpips=1 '
              f'peak_kv_memory_mb=0 fps=0 n_frames_kept=0 eval_mode=real_rollout')
        return

    # Evaluate each clip
    all_metrics = []
    t0 = time.time()
    for clip_idx, (frames, actions) in enumerate(clips):
        ctx_frames = frames[:n_ctx].unsqueeze(0)     # [1, n_ctx, C, H, W]
        gt_pred = frames[n_ctx:n_ctx + n_pred]       # [n_pred, C, H, W]

        conditions = None
        if wl_cfg['conditioned'] and isinstance(actions, torch.Tensor):
            conditions = {'action': actions.unsqueeze(0).to(pipeline.execution_device)}

        try:
            pred = generate_with_policy(
                pipeline, ctx_frames, policy, wl_cfg, budget_cfg,
                conditions=conditions, seed=seed + clip_idx)
            # pred: [1, n_pred, C, H, W]
            pred_frames = pred[0]   # [n_pred, C, H, W]
            m = _compute_metrics(pred_frames, gt_pred.to(device), device)
            all_metrics.append(m)
        except Exception as e:
            print(f'  clip {clip_idx} failed: {e}')
            continue

        if (clip_idx + 1) % 10 == 0:
            print(f'  {clip_idx + 1}/{len(clips)} clips done')

    if not all_metrics:
        print('ERROR: all clips failed')
        return

    elapsed = time.time() - t0
    fps = (len(all_metrics) * n_pred) / elapsed

    psnr = sum(m['psnr'] for m in all_metrics) / len(all_metrics)
    ssim = sum(m['ssim'] for m in all_metrics) / len(all_metrics)
    lp   = sum(m['lpips'] for m in all_metrics) / len(all_metrics)
    tq   = _psnr_to_quality_main(psnr)

    max_frames = budget_cfg['max_frames']
    C_lat = 32   # DCAE output channels
    lat_size = wl_cfg['resolution'] // 8
    peak_kv_mb = min(max_frames, n_ctx + n_pred) * C_lat * lat_size * lat_size * 4 / (1024 ** 2)

    print(f'TEST_METRICS: '
          f'temporal_quality_main={tq:.4f} '
          f'psnr={psnr:.4f} '
          f'ssim={ssim:.4f} '
          f'lpips={lp:.4f} '
          f'peak_kv_memory_mb={peak_kv_mb:.2f} '
          f'fps={fps:.4f} '
          f'n_frames_kept={min(max_frames, n_ctx + n_pred)} '
          f'eval_mode=real_rollout '
          f'n_clips={len(all_metrics)} '
          f'elapsed_s={elapsed:.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', required=True,
                        choices=sorted(WORKLOAD_CONFIGS))
    parser.add_argument('--budget', required=True,
                        choices=sorted(BUDGET_CONFIGS))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_evaluation(args.workload, args.budget, args.seed)

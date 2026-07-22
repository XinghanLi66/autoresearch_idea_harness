"""
Extract per-chunk temporal feature statistics from real video frames.

For each workload, reads a directory of sequential JPEG frames and computes:
  - temporal_similarity:  cosine similarity of mean-pooled DCT features between chunk and previous
  - feature_drift:        L2 distance from reference (first) chunk features, normalized
  - keyframe_score:       optical flow magnitude proxy (Farneback or pixel-diff fallback)
  - boundary_score:       histogram chi-squared distance (scene change detector)
  - access_count:         always 1 at trace time (runtime-updated by harness)
  - chunk_size_tokens:    estimated token count from spatial resolution + chunk_frames

The output is a trace JSON file analogous to DLM's llada_step_trace_*.json, which the
ar-video-kv-temporal-policy harness can load instead of DEFAULT_BENCHMARK_TRACE.

Trace schema:
{
  "trace_version": "video-chunk-trace-v1",
  "workload":       "ucf101_short_prediction",
  "source_dataset": "UCF101",
  "source_family":  "ucf101",
  "frame_count":    48,
  "chunk_frames":   4,
  "n_chunks":       12,
  "chunks": [
    {
      "chunk_id":           0,
      "temporal_similarity": 1.0,
      "feature_drift":       0.0,
      "keyframe_score":      0.12,
      "boundary_score":      0.05,
      "chunk_size_tokens":   256
    },
    ...
  ]
}
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _load_frame_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _load_frame_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load frame: {path}")
    return img


def _resize(frame: np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _dct_features(gray: np.ndarray, n: int = 8) -> np.ndarray:
    """Compute DCT-based compact feature vector from a grayscale frame."""
    resized = _resize(gray, (64, 64)).astype(np.float32)
    dct = cv2.dct(resized)
    # Take top-left n×n coefficients (low-frequency content)
    block = dct[:n, :n].flatten()
    norm = np.linalg.norm(block)
    return block / (norm + 1e-8)


def _hist_features(bgr: np.ndarray) -> np.ndarray:
    """Compute normalized color histogram for scene change detection."""
    hists = []
    for ch in range(3):
        h = cv2.calcHist([bgr], [ch], None, [32], [0, 256]).flatten()
        hists.append(h)
    feat = np.concatenate(hists)
    feat /= feat.sum() + 1e-8
    return feat


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    return max(-1.0, min(1.0, dot))


def _chi_squared_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = a + b + 1e-8
    return float(np.sum((a - b) ** 2 / denom))


def _optical_flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Farneback optical flow magnitude — proxy for motion / keyframe saliency."""
    prev_r = _resize(prev_gray, (32, 32))
    curr_r = _resize(curr_gray, (32, 32))
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_r, curr_r, None,
            pyr_scale=0.5, levels=3, winsize=5,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(mag))
    except Exception:
        # Fallback: simple pixel difference
        diff = np.abs(curr_r.astype(np.float32) - prev_r.astype(np.float32))
        return float(np.mean(diff)) / 255.0


# ---------------------------------------------------------------------------
# Chunk-level aggregation
# ---------------------------------------------------------------------------

def _aggregate_chunk_features(
    frames_gray: list[np.ndarray],
    frames_bgr: list[np.ndarray],
) -> dict:
    """Aggregate per-frame features into chunk-level statistics."""
    dct_feats = [_dct_features(f) for f in frames_gray]
    hist_feats = [_hist_features(f) for f in frames_bgr]
    # Representative features: mean across frames in the chunk
    chunk_dct = np.mean(dct_feats, axis=0)
    chunk_dct /= np.linalg.norm(chunk_dct) + 1e-8
    chunk_hist = np.mean(hist_feats, axis=0)
    # Optical flow magnitude within the chunk (max frame-pair)
    max_flow = 0.0
    for i in range(len(frames_gray) - 1):
        mag = _optical_flow_magnitude(frames_gray[i], frames_gray[i + 1])
        max_flow = max(max_flow, mag)
    return {"dct": chunk_dct, "hist": chunk_hist, "max_flow": max_flow}


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

WORKLOAD_META = {
    "ucf101_short_prediction": {
        "source_dataset": "UCF101",
        "source_family": "ucf101",
        "chunk_frames": 4,
        "target_chunks": 12,  # ~48 frames
        "description": "Short action prediction — single scene, smooth motion",
    },
    "dmlab_long_navigation": {
        "source_dataset": "DMLab-30",
        "source_family": "dmlab",
        "chunk_frames": 4,
        "target_chunks": 16,  # ~64 frames
        "description": "Long first-person navigation — slow drift, rare boundaries",
    },
    "minecraft_scene_cut": {
        "source_dataset": "Minecraft (OpenAI VPT / IGLU)",
        "source_family": "minecraft",
        "chunk_frames": 4,
        "target_chunks": 14,  # ~56 frames, with scene cut
        "description": "Scene-cut sequence — abrupt boundary ~40% through",
    },
}

FRAME_DIR_MAP = {
    "ucf101_short_prediction": "ucf101",
    "dmlab_long_navigation": "dmlab",
    "minecraft_scene_cut": "minecraft",
}


def _load_frames_from_dir(frame_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load all frames from a directory, sorted by filename."""
    paths = sorted(frame_dir.glob("frame_*.jpg")) + sorted(frame_dir.glob("*.jpg"))
    # Deduplicate
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    paths = unique_paths
    if not paths:
        raise ValueError(f"No frames found in {frame_dir}")
    grays, bgrs = [], []
    for p in paths:
        bgr = _load_frame_bgr(p)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        bgrs.append(bgr)
    return grays, bgrs


def _load_frames_from_video(video_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    grays, bgrs = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        bgrs.append(frame)
    cap.release()
    return grays, bgrs


def extract_trace(
    workload: str,
    data_root: Path,
    output_path: Optional[Path] = None,
) -> dict:
    meta = WORKLOAD_META[workload]
    frame_subdir = FRAME_DIR_MAP[workload]
    frame_dir = data_root / frame_subdir
    chunk_frames = meta["chunk_frames"]
    target_chunks = meta["target_chunks"]

    # Load frames — prefer directory of JPEGs, fall back to video files
    grays, bgrs = [], []
    if frame_dir.exists() and list(frame_dir.glob("frame_*.jpg")):
        print(f"  Loading frames from directory: {frame_dir}", flush=True)
        grays, bgrs = _load_frames_from_dir(frame_dir)
    else:
        # Look for video files
        video_files = list(frame_dir.glob("*.mp4")) + list(frame_dir.glob("*.avi"))
        if video_files:
            print(f"  Loading frames from video: {video_files[0]}", flush=True)
            grays, bgrs = _load_frames_from_video(video_files[0])
        else:
            raise ValueError(
                f"No frames or video found in {frame_dir}. "
                "Run download_trace_data.py first."
            )

    total_frames = len(grays)
    n_chunks = min(target_chunks, total_frames // chunk_frames)
    if n_chunks == 0:
        raise ValueError(f"Not enough frames ({total_frames}) for chunk_frames={chunk_frames}")

    print(f"  {total_frames} frames → {n_chunks} chunks of {chunk_frames} frames each", flush=True)

    # Build per-chunk feature vectors
    chunk_feats = []
    for c in range(n_chunks):
        start = c * chunk_frames
        end = start + chunk_frames
        cg = grays[start:end]
        cb = bgrs[start:end]
        feat = _aggregate_chunk_features(cg, cb)
        chunk_feats.append(feat)

    # Build statistics
    ref_dct = chunk_feats[0]["dct"]
    ref_hist = chunk_feats[0]["hist"]

    # Normalise feature_drift range per-workload (cumulative drift from reference)
    raw_drifts = [
        float(np.linalg.norm(cf["dct"] - ref_dct))
        for cf in chunk_feats
    ]
    max_drift = max(raw_drifts[1:]) if len(raw_drifts) > 1 else 1.0

    # boundary_score: use ABSOLUTE calibration (not per-workload normalization)
    # Calibrated so that:
    #   chi-sq ~0.0-0.02  → smooth same-scene motion  → boundary_score ~0.0-0.13
    #   chi-sq ~0.05-0.10 → moderate scene activity    → boundary_score ~0.33-0.67
    #   chi-sq >=0.15     → hard scene cut             → boundary_score ~1.0
    BOUNDARY_CALIB_MAX = 0.15
    raw_boundaries = [0.0] + [
        _chi_squared_distance(chunk_feats[i - 1]["hist"], chunk_feats[i]["hist"])
        for i in range(1, n_chunks)
    ]
    # No per-workload max normalization — use absolute calibration
    # (DMLab smooth nav stays near 0, Minecraft cut reaches 1.0)

    # Normalise keyframe_score (max optical flow within chunk) per-workload
    raw_kf = [cf["max_flow"] for cf in chunk_feats]
    max_kf = max(raw_kf) if max(raw_kf) > 0 else 1.0

    chunks_out = []
    for c in range(n_chunks):
        cf = chunk_feats[c]
        # temporal_similarity: cosine similarity vs previous chunk
        if c == 0:
            tsim = 1.0
        else:
            tsim = float(_cosine_similarity(cf["dct"], chunk_feats[c - 1]["dct"]))
            # Map from [-1,1] to [0,1] and clamp to [0.3, 1.0] for realistic range
            tsim = max(0.30, min(1.0, 0.5 + 0.5 * tsim))

        # feature_drift: normalised L2 from reference, range [0,1]
        drift = min(1.0, raw_drifts[c] / (max_drift + 1e-8))

        # keyframe_score: normalised optical flow, range [0,1]
        kf = min(1.0, raw_kf[c] / (max_kf + 1e-8))

        # boundary_score: absolute-calibrated chi-squared distance, range [0,1]
        # Same scale across all workloads so policies can use a fixed threshold
        bd = min(1.0, raw_boundaries[c] / BOUNDARY_CALIB_MAX)

        # chunk_size_tokens: proportional to resolution (default 64×64 chunks at 4 frames)
        h, w = grays[c * chunk_frames].shape
        spatial_tokens = (h // 8) * (w // 8)  # rough ViT-style patch count
        chunk_size_tokens = spatial_tokens * chunk_frames

        chunks_out.append({
            "chunk_id": c,
            "temporal_similarity": round(tsim, 4),
            "feature_drift": round(drift, 4),
            "keyframe_score": round(kf, 4),
            "boundary_score": round(bd, 4),
            "access_count": 1,
            "chunk_size_tokens": chunk_size_tokens,
        })

    trace = {
        "trace_version": "video-chunk-trace-v1",
        "workload": workload,
        "source_dataset": meta["source_dataset"],
        "source_family": meta["source_family"],
        "description": meta["description"],
        "frame_count": total_frames,
        "chunk_frames": chunk_frames,
        "n_chunks": n_chunks,
        "chunks": chunks_out,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(trace, indent=2))
        print(f"  Trace written to: {output_path}", flush=True)

    return trace


def main():
    parser = argparse.ArgumentParser(description="Extract video-chunk traces for benchmark")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory containing ucf101/, dmlab/, minecraft/ subdirs of frames",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for trace JSON files",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=list(WORKLOAD_META.keys()),
        choices=list(WORKLOAD_META.keys()),
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    for workload in args.workloads:
        print(f"\n=== Extracting trace: {workload} ===", flush=True)
        out_path = out_dir / f"{workload}.trace.json"
        try:
            trace = extract_trace(workload, data_root, output_path=out_path)
            n = len(trace["chunks"])
            tsim_mean = sum(c["temporal_similarity"] for c in trace["chunks"]) / n
            bd_max = max(c["boundary_score"] for c in trace["chunks"])
            print(
                f"  OK: {n} chunks, "
                f"mean_tsim={tsim_mean:.3f}, "
                f"max_boundary={bd_max:.3f}",
                flush=True,
            )
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            raise

    print("\nAll traces extracted.", flush=True)


if __name__ == "__main__":
    main()

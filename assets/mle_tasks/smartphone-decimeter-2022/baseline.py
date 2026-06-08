"""
Baseline solution: use host WLS trajectory estimates when available.

Produces submission.csv with columns:
phone, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees.
"""
from pathlib import Path

import numpy as np
import pandas as pd


data_dir = Path("data")


def ecef_to_lat_lon(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 ECEF meters to latitude/longitude degrees."""
    a = 6378137.0
    e2 = 6.69437999014e-3
    b = a * np.sqrt(1.0 - e2)
    ep2 = (a * a - b * b) / (b * b)
    p = np.sqrt(x * x + y * y)
    theta = np.arctan2(z * a, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3,
        p - e2 * a * np.cos(theta) ** 3,
    )
    return np.degrees(lat), np.degrees(lon)


sample = pd.read_csv(data_dir / "sample_submission.csv")
pred_parts: list[pd.DataFrame] = []

for gnss_path in sorted((data_dir / "test").glob("*/*/device_gnss.csv")):
    phone_name = gnss_path.parent.name
    drive_id = gnss_path.parent.parent.name
    phone = f"{drive_id}_{phone_name}"
    cols = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    try:
        df = pd.read_csv(gnss_path, usecols=lambda c: c in cols)
    except Exception:
        continue
    if not set(cols).issubset(df.columns):
        continue
    df = df.dropna(subset=cols).groupby("utcTimeMillis", as_index=False).first()
    if df.empty:
        continue
    lat, lon = ecef_to_lat_lon(
        df["WlsPositionXEcefMeters"].to_numpy(dtype=float),
        df["WlsPositionYEcefMeters"].to_numpy(dtype=float),
        df["WlsPositionZEcefMeters"].to_numpy(dtype=float),
    )
    pred_parts.append(pd.DataFrame({
        "phone": phone,
        "UnixTimeMillis": df["utcTimeMillis"].astype("int64"),
        "LatitudeDegrees": lat,
        "LongitudeDegrees": lon,
    }))

if pred_parts:
    pred = pd.concat(pred_parts, ignore_index=True).sort_values(["phone", "UnixTimeMillis"])
    out_parts: list[pd.DataFrame] = []
    for phone, target in sample.groupby("phone", sort=False):
        src = pred[pred["phone"] == phone]
        if src.empty:
            out_parts.append(target)
            continue
        merged = pd.merge_asof(
            target.sort_values("UnixTimeMillis"),
            src.sort_values("UnixTimeMillis"),
            on="UnixTimeMillis",
            by="phone",
            direction="nearest",
            suffixes=("", "_wls"),
        )
        merged["LatitudeDegrees"] = merged["LatitudeDegrees_wls"].fillna(merged["LatitudeDegrees"])
        merged["LongitudeDegrees"] = merged["LongitudeDegrees_wls"].fillna(merged["LongitudeDegrees"])
        out_parts.append(merged[sample.columns])
    submission = pd.concat(out_parts, ignore_index=True)
else:
    submission = sample

submission.to_csv("submission.csv", index=False)
print(f"submission.csv written ({len(submission)} rows)")

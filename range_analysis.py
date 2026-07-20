"""
range_analysis.py
-------------------
Computes camera-to-drone distance per frame from GPS logs and correlates
it with YOLO detection results (success/confidence), to characterize the
practical maximum detection range of a camera + model combination.

Expected input: a CSV log with one row per frame, containing:
  frame, timestamp, camera_lat, camera_lon, drone_lat, drone_lon,
  drone_alt_m, detected (0/1), confidence

If your telemetry log has different column names, adjust the
COLUMN NAMES section below to match.

Usage:
  python range_analysis.py --log flight_log.csv --out results

Outputs:
  results/range_vs_detection.png   - detection rate/confidence vs distance
  results/range_summary.csv        - per-frame distance + detection joined
  Prints: max reliable detection range to console
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---- COLUMN NAMES: adjust these if your CSV uses different headers ----
COL_CAMERA_LAT = "camera_lat"
COL_CAMERA_LON = "camera_lon"
COL_DRONE_LAT = "drone_lat"
COL_DRONE_LON = "drone_lon"
COL_DETECTED = "detected"
COL_CONFIDENCE = "confidence"
# -------------------------------------------------------------------------


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two GPS points, in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    df["distance_m"] = df.apply(
        lambda row: haversine_distance_m(
            row[COL_CAMERA_LAT], row[COL_CAMERA_LON],
            row[COL_DRONE_LAT], row[COL_DRONE_LON],
        ),
        axis=1,
    )
    return df


def summarize_by_distance_bins(df: pd.DataFrame, bin_size_m: float = 10.0) -> pd.DataFrame:
    max_dist = df["distance_m"].max()
    bins = list(range(0, int(max_dist) + int(bin_size_m), int(bin_size_m)))
    df["distance_bin"] = pd.cut(df["distance_m"], bins=bins)

    summary = df.groupby("distance_bin", observed=True).agg(
        detection_rate=(COL_DETECTED, "mean"),
        avg_confidence=(COL_CONFIDENCE, "mean"),
        n_frames=(COL_DETECTED, "count"),
    ).reset_index()
    summary["distance_bin_start"] = summary["distance_bin"].apply(lambda b: b.left)
    return summary


def find_reliable_range(summary: pd.DataFrame, min_detection_rate: float = 0.8) -> float:
    """Finds the largest distance bin where detection rate stays reliable."""
    reliable = summary[summary["detection_rate"] >= min_detection_rate]
    if reliable.empty:
        return 0.0
    return float(reliable["distance_bin_start"].max())


def plot_results(summary: pd.DataFrame, out_path: Path):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.bar(summary["distance_bin_start"], summary["detection_rate"],
            width=8, alpha=0.6, color="steelblue", label="Detection rate")
    ax1.set_xlabel("Camera-to-drone distance (m)")
    ax1.set_ylabel("Detection rate", color="steelblue")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(summary["distance_bin_start"], summary["avg_confidence"],
              color="darkorange", marker="o", label="Avg confidence")
    ax2.set_ylabel("Average confidence", color="darkorange")
    ax2.set_ylim(0, 1.05)

    plt.title("Detection Rate & Confidence vs. Camera-to-Drone Distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="CSV with per-frame GPS + detection data")
    ap.add_argument("--out", default="results", help="Output directory for plot/summary")
    ap.add_argument("--bin-size", type=float, default=10.0, help="Distance bin size in meters")
    ap.add_argument("--min-detection-rate", type=float, default=0.8,
                     help="Detection rate threshold to consider a distance 'reliable'")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.log)
    df = compute_distances(df)
    df.to_csv(out_dir / "range_summary_raw.csv", index=False)

    summary = summarize_by_distance_bins(df, bin_size_m=args.bin_size)
    summary.to_csv(out_dir / "range_summary.csv", index=False)

    plot_results(summary, out_dir / "range_vs_detection.png")

    reliable_range = find_reliable_range(summary, args.min_detection_rate)
    print("\n" + "=" * 50)
    print(f"Max distance tested       : {df['distance_m'].max():.1f} m")
    print(f"Reliable detection range  : ~{reliable_range:.0f} m "
          f"(detection rate >= {args.min_detection_rate:.0%})")
    print("=" * 50)


if __name__ == "__main__":
    main()

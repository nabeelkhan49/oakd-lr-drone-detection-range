"""
finetune_drone_vs_bird.py
--------------------------
Downloads the "drone-vs-bird" dataset (2,920 images) from Roboflow Universe,
already in YOLO format, and fine-tunes your EXISTING Anti-UAV best.pt on it.
This adds small/distant/low-contrast drone examples on top of what your
model already learned, rather than starting over.

Dataset source (public, no application needed):
  https://universe.roboflow.com/dam-tpuul/drone-vs-bird-lanzg

Note: the OFFICIAL WOSDETC "Drone-vs-Bird" challenge dataset is larger and
video-based, but requires emailing wosdetc@googlegroups.com and signing a
data usage agreement before you get access - not a direct download. This
script uses the open Roboflow version instead so you can start immediately.

Prereqs:
  pip install ultralytics roboflow

You need a free Roboflow account + API key:
  1. Sign up at https://roboflow.com (free)
  2. Get your API key from https://app.roboflow.com/settings/api
  3. Pass it via --api-key, or set env var ROBOFLOW_API_KEY

Usage:
  python finetune_drone_vs_bird.py --api-key YOUR_KEY --base-weights "D:\\runs\\detect\\antiuav_drone\\weights\\best.pt" --epochs 60
"""

import argparse
import os
from pathlib import Path


def download_dataset(api_key: str, out_dir: Path):
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("dam-tpuul").project("drone-vs-bird-lanzg")
    # Grabs the latest published version automatically
    version = project.version(project.versions()[-1].version)
    dataset = version.download("yolov11", location=str(out_dir))
    return Path(dataset.location) / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                     help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    ap.add_argument("--data-dir", default="drone_vs_bird_data",
                     help="Where to download the dataset")
    ap.add_argument("--base-weights", required=True,
                     help="Path to your existing best.pt (Anti-UAV trained) to fine-tune from")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="drone_finetuned")
    ap.add_argument("--lr0", type=float, default=0.001,
                     help="Lower learning rate than a fresh training run, since we're "
                          "fine-tuning an already-trained model, not starting from ImageNet weights")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit(
            "No Roboflow API key provided. Get one free at "
            "https://app.roboflow.com/settings/api and pass it via --api-key, "
            "or set the ROBOFLOW_API_KEY environment variable."
        )

    print("Downloading drone-vs-bird dataset from Roboflow...")
    data_yaml = download_dataset(args.api_key, Path(args.data_dir))
    print(f"Dataset ready at: {data_yaml}")

    from ultralytics import YOLO

    print(f"Loading existing trained weights from: {args.base_weights}")
    model = YOLO(args.base_weights)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        lr0=args.lr0,          # smaller LR - we're refining, not training from scratch
        patience=20,
        mosaic=1.0,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        plots=True,
    )

    metrics = model.val()
    print("\nValidation results after fine-tuning:")
    print(metrics)
    print(f"\nFine-tuned weights saved at: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

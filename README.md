# oakd-lr-drone-detection-range

YOLOv11 drone detector fine-tuned on the drone-vs-bird dataset, with
real-world detection range testing using an OAK-D LR (OpenCV AI Kit -
Depth, Long Range) camera and synchronized drone GPS telemetry.

## Overview

- **Model**: YOLOv11s fine-tuned on the Roboflow drone-vs-bird dataset,
  detecting drones and distinguishing them from birds.
- **Real-world range testing**: Detection performance evaluated on video
  captured with an OAK-D LR camera, using drone GPS coordinates and the
  camera's GPS position to compute ground-truth distance per frame.

## Dataset

- **Drone-vs-Bird (Roboflow Universe)** — https://universe.roboflow.com/dam-tpuul/drone-vs-bird-lanzg

Check the dataset's license before redistributing raw data; this repo
includes only training/inference code and resulting model weights.

## Demo — Real-Time Drone Detection

https://github.com/user-attachments/assets/b8e243fd-2b6a-4bad-a79f-4ea09f861f92

## Pipeline

1. `finetune_drone_vs_bird.py` — downloads the drone-vs-bird dataset and
   trains/fine-tunes a YOLOv11 model on it.
2. `inference.py` — runs the trained model on images, video, or a live
   camera feed. Supports class filtering, hiding confidence scores, and
   renaming class labels (e.g. bird/drone).
3. `range_analysis.py` — computes camera-to-drone distance per frame from
   GPS logs and correlates it with detection success/confidence to
   characterize practical detection range.

## GPS Telemetry & Range Measurement

The OAK-D LR camera captures video frames while simultaneously receiving
drone telemetry — including live GPS position — over its link to the
drone. This telemetry is streamed in real time to a base station laptop,
where each video frame is timestamp-matched with the corresponding drone
GPS coordinate and the camera's own fixed GPS position.

For every frame, three values are logged:
- Camera GPS position (fixed, ground station)
- Drone GPS position (live telemetry)
- Straight-line distance between the two, computed via the haversine formula

This per-frame distance was then correlated with YOLO detection results
(success/failure, confidence) to determine the effective detection range
of the camera + model combination.

**Result**: Based on this camera-to-drone distance analysis, the system
reliably detected the drone up to approximately **80 meters**, under
clear daylight conditions.

Exact GPS coordinates are not published to avoid revealing the test
location; only derived distance measurements and detection outcomes are
reported here.

## Setup

```bash
pip install ultralytics opencv-python tqdm roboflow
```

## Usage

```bash
# Train / fine-tune on drone-vs-bird
python finetune_drone_vs_bird.py --api-key YOUR_ROBOFLOW_KEY --epochs 60

# Run inference
python inference.py --weights runs/detect/drone_finetuned/weights/best.pt --source your_video.mp4 --show --conf 0.35 --hide-conf --rename-classes bird,drone
```

## Results

- mAP50 ~0.79 overall (drone class mAP50 ~0.92)
- Real-world OAK-D LR range test: reliable detection up to approximately
  **80 meters**, under clear daylight conditions, based on GPS-derived
  camera-to-drone distance.

## Acknowledgments

- Roboflow drone-vs-bird dataset contributors
- Ultralytics YOLOv11

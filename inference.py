"""
inference.py
-------------
Runs your trained YOLOv11 drone detector on your own images and/or videos.

Prereqs:
  pip install ultralytics

Usage examples:
  # Single image
  python inference.py --weights runs/detect/antiuav_drone/weights/best.pt --source my_photo.jpg

  # Folder of images
  python inference.py --weights best.pt --source "C:\\Users\\nabee\\my_images"

  # Video file
  python inference.py --weights best.pt --source my_video.mp4

  # Webcam (device 0)
  python inference.py --weights best.pt --source 0

  # Lower confidence threshold if it's missing small/far drones
  python inference.py --weights best.pt --source my_video.mp4 --conf 0.15

Output:
  Annotated images/video are saved under runs/detect/predict*/ by default,
  and printed to console. Use --show to also pop up a live window for videos.
"""

import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained best.pt")
    ap.add_argument("--source", required=True,
                     help="Image path, folder path, video path, or webcam index (e.g. 0)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    ap.add_argument("--show", action="store_true", help="Show live preview window (video/webcam)")
    ap.add_argument("--save-txt", action="store_true", help="Also save YOLO-format txt detections")
    ap.add_argument("--hide-conf", action="store_true",
                     help="Hide the confidence number, show only the class label on boxes")
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                     help="Only show these class indices, e.g. --classes 1 to show only drones")
    ap.add_argument("--rename-classes", type=str, default=None,
                     help="Comma-separated names to relabel classes in order, e.g. "
                          "--rename-classes bird,drone maps class 0->bird, class 1->drone")
    args = ap.parse_args()

    model = YOLO(args.weights)

    if args.rename_classes:
        new_names = [n.strip() for n in args.rename_classes.split(",")]
        current_names = dict(model.names)
        for i, name in enumerate(new_names):
            if i in current_names:
                current_names[i] = name
        model.names = current_names
        if hasattr(model, "model") and hasattr(model.model, "names"):
            model.model.names = current_names

    # source can be int (webcam) passed as string; convert if it's a digit
    source = int(args.source) if args.source.isdigit() else args.source

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        show=args.show,
        save=True,
        save_txt=args.save_txt,
        show_conf=not args.hide_conf,
        classes=args.classes,
        stream=True,  # memory-efficient for videos/large folders
    )

    n_frames = 0
    n_detections = 0
    for r in results:
        n_frames += 1
        n_detections += len(r.boxes)

    print("\n" + "=" * 50)
    print(f"Frames/images processed : {n_frames}")
    print(f"Total drone detections  : {n_detections}")
    print("Annotated output saved under: runs/detect/predict*/")
    print("=" * 50)


if __name__ == "__main__":
    main()

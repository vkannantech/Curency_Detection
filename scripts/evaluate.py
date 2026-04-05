"""Evaluate a trained YOLO currency detection model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from currency_detection.config import DATASET_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO currency detector.")
    parser.add_argument("--weights", required=True, help="Path to trained weights file.")
    parser.add_argument("--data", default=str(DATASET_CONFIG), help="Path to dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=960, help="Validation image size.")
    parser.add_argument("--device", default="cpu", help="Device, for example cpu or 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    names = getattr(metrics, "names", {}) or {}
    maps = getattr(metrics.box, "maps", [])
    if names and maps:
        print("\nPer-class mAP50-95:")
        for index, score in enumerate(maps):
            print(f"- {names.get(index, index)}: {score:.4f}")



if __name__ == "__main__":
    main()

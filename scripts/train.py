"""Train a custom YOLO currency detector."""

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
    parser = argparse.ArgumentParser(description="Train a YOLO model for currency detection.")
    parser.add_argument("--model", default="yolo11m.pt", help="Base model or weights file.")
    parser.add_argument("--data", default=str(DATASET_CONFIG), help="Path to dataset YAML.")
    parser.add_argument(
        "--profile",
        default="max-accuracy",
        choices=["max-accuracy", "balanced", "pi-speed"],
        help="Training preset for accuracy or deployment speed.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device like 0, or cpu.")
    parser.add_argument("--project", default="runs/detect", help="Training output directory.")
    parser.add_argument("--name", default="currency_yolo", help="Run name.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--patience", type=int, default=30, help="Early stop patience.")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training.")
    parser.add_argument("--resume", action="store_true", help="Resume the last interrupted training run.")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze initial model layers.")
    return parser.parse_args()


def build_profile_settings(profile: str) -> dict:
    if profile == "pi-speed":
        return {
            "imgsz": 640,
            "close_mosaic": 10,
            "optimizer": "AdamW",
            "cos_lr": True,
            "degrees": 7.5,
            "translate": 0.08,
            "scale": 0.25,
            "shear": 2.0,
            "perspective": 0.0005,
            "fliplr": 0.5,
            "mosaic": 0.8,
            "mixup": 0.05,
        }
    if profile == "balanced":
        return {
            "imgsz": 832,
            "close_mosaic": 12,
            "optimizer": "AdamW",
            "cos_lr": True,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.35,
            "shear": 3.0,
            "perspective": 0.001,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.08,
        }
    return {
        "imgsz": 960,
        "close_mosaic": 15,
        "optimizer": "AdamW",
        "cos_lr": True,
        "degrees": 12.0,
        "translate": 0.12,
        "scale": 0.4,
        "shear": 4.0,
        "perspective": 0.001,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
    }


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    profile_settings = build_profile_settings(args.profile)
    imgsz = args.imgsz if args.imgsz else profile_settings["imgsz"]
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
        pretrained=True,
        optimizer=profile_settings["optimizer"],
        cos_lr=profile_settings["cos_lr"],
        close_mosaic=profile_settings["close_mosaic"],
        degrees=profile_settings["degrees"],
        translate=profile_settings["translate"],
        scale=profile_settings["scale"],
        shear=profile_settings["shear"],
        perspective=profile_settings["perspective"],
        fliplr=profile_settings["fliplr"],
        mosaic=profile_settings["mosaic"],
        mixup=profile_settings["mixup"],
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        amp=True,
        plots=True,
        freeze=args.freeze,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

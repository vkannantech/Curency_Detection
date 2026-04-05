"""Run the real-time currency camera assistant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from currency_detection.camera import CurrencyCameraAssistant, validate_weights_path
from currency_detection.config import CameraConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time currency detection on a camera.")
    parser.add_argument("--weights", required=True, help="Path to trained weights or Ultralytics model name.")
    parser.add_argument("--source", default="0", help="Camera index, video path, or stream URL.")
    parser.add_argument(
        "--profile",
        default="desktop",
        choices=["desktop", "pi", "pi-lite"],
        help="Preset tuned for desktop or Raspberry Pi performance.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", default="cpu", help="Inference device, such as cpu or 0.")
    parser.add_argument("--speak", action="store_true", help="Enable audio announcements.")
    parser.add_argument("--headless", action="store_true", help="Disable display window.")
    parser.add_argument("--max-fps", type=float, default=12.0, help="Limit processing FPS.")
    parser.add_argument("--cooldown", type=float, default=2.5, help="Seconds before repeating speech.")
    parser.add_argument("--width", type=int, default=1280, help="Camera width.")
    parser.add_argument("--height", type=int, default=720, help="Camera height.")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation for more accuracy.")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable frame preprocessing.")
    parser.add_argument("--smoothing-window", type=int, default=12, help="Frames to use for temporal consensus.")
    parser.add_argument("--stable-ratio", type=float, default=0.5, help="Fraction of history required to validate a class.")
    parser.add_argument("--min-confidence-sum", type=float, default=1.2, help="Minimum accumulated confidence across the window.")
    return parser.parse_args()


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.profile == "pi":
        args.imgsz = 640 if args.imgsz == 640 else args.imgsz
        args.max_fps = 10.0 if args.max_fps == 12.0 else args.max_fps
        args.width = 960 if args.width == 1280 else args.width
        args.height = 544 if args.height == 720 else args.height
    elif args.profile == "pi-lite":
        args.imgsz = 512 if args.imgsz == 640 else args.imgsz
        args.max_fps = 8.0 if args.max_fps == 12.0 else args.max_fps
        args.width = 854 if args.width == 1280 else args.width
        args.height = 480 if args.height == 720 else args.height
    return args


def main() -> None:
    args = apply_profile_defaults(parse_args())
    config = CameraConfig(
        weights=validate_weights_path(args.weights),
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        speak=args.speak,
        headless=args.headless,
        max_fps=args.max_fps,
        cooldown=args.cooldown,
        resolution_width=args.width,
        resolution_height=args.height,
        profile=args.profile,
        preprocess=not args.no_preprocess,
        use_tta=args.tta,
        smoothing_window=args.smoothing_window,
        stable_ratio=args.stable_ratio,
        min_confidence_sum=args.min_confidence_sum,
    )
    CurrencyCameraAssistant(config).run()


if __name__ == "__main__":
    main()

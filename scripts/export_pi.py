"""Export a trained model to a Raspberry Pi friendly format."""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained YOLO model for Raspberry Pi.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "openvino", "torchscript", "engine", "tflite", "ncnn", "mnn", "paddle"],
        help="Export format for deployment.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--half", action="store_true", help="Use half precision if supported.")
    parser.add_argument("--int8", action="store_true", help="Use int8 export when supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    result = model.export(format=args.format, imgsz=args.imgsz, half=args.half, int8=args.int8)
    print(f"Export complete: {result}")


if __name__ == "__main__":
    main()

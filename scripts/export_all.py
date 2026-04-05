"""Export one trained model to many deployment backends."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from currency_detection.engine import export_formats_for_target, list_exportable_engines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained model to multiple runtime formats.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument(
        "--target",
        default="pi",
        choices=["desktop", "pi", "mobile"],
        help="Deployment target that decides the default formats.",
    )
    parser.add_argument(
        "--formats",
        nargs="*",
        default=None,
        choices=list_exportable_engines(),
        help="Explicit export formats. If omitted, target defaults are used.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--half", action="store_true", help="Use half precision if supported.")
    parser.add_argument("--int8", action="store_true", help="Use int8 export if supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    formats = args.formats or export_formats_for_target(args.target)

    print(f"Exporting {args.weights} for target '{args.target}'")
    for export_format in formats:
        print(f"\n[export] {export_format}")
        result = model.export(
            format=export_format,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
        )
        print(f"Completed: {result}")


if __name__ == "__main__":
    main()

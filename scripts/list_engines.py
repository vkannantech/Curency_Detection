"""Print supported inference engines and deployment hints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from currency_detection.engine import (
    build_export_plan,
    engine_support_matrix,
    export_formats_for_target,
    list_supported_engines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List supported inference engines.")
    parser.add_argument(
        "--target",
        default="desktop",
        choices=["desktop", "pi", "mobile"],
        help="Show recommended engines for a deployment target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Supported engines:")
    for engine in list_supported_engines():
        print(f"- {engine}")

    print("\nSupport matrix:")
    print(engine_support_matrix())

    print(f"\nRecommended export plan for {args.target}:")
    for engine, reason in build_export_plan(args.target):
        print(f"- {engine}: {reason}")

    print("\nRecommended export formats:")
    for export_format in export_formats_for_target(args.target):
        print(f"- {export_format}")


if __name__ == "__main__":
    main()

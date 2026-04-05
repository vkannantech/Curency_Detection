"""Benchmark multiple backend weights on the same image or video frame."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from currency_detection.engine import build_runtime_hint, resolve_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple YOLO backend artifacts.")
    parser.add_argument("--weights", nargs="+", required=True, help="One or more backend weight paths.")
    parser.add_argument("--image", required=True, help="Image used for benchmarking.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--device", default="cpu", help="Inference device.")
    parser.add_argument("--runs", type=int, default=10, help="Benchmark iterations per backend.")
    return parser.parse_args()


def benchmark_one(weights: str, image, imgsz: int, conf: float, iou: float, device: str, runs: int) -> tuple[str, float]:
    spec = resolve_engine(weights, "auto")
    model = YOLO(spec.normalized_weights)
    timings: list[float] = []

    for _ in range(runs):
        started = time.perf_counter()
        model.predict(source=image, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        timings.append((time.perf_counter() - started) * 1000.0)

    median_ms = statistics.median(timings)
    print(f"\n[{spec.detected}] {weights}")
    print(f"Median latency: {median_ms:.2f} ms")
    print(build_runtime_hint(spec.detected))
    return spec.detected, median_ms


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    summary: list[tuple[str, float, str]] = []
    for weights in args.weights:
        engine, latency = benchmark_one(
            weights=weights,
            image=image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            runs=args.runs,
        )
        summary.append((engine, latency, weights))

    print("\nRanking by latency:")
    for rank, (engine, latency, weights) in enumerate(sorted(summary, key=lambda item: item[1]), start=1):
        print(f"{rank}. {engine} | {latency:.2f} ms | {weights}")


if __name__ == "__main__":
    main()

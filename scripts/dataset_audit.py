"""Inspect dataset health before training."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a YOLO currency dataset.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to dataset YAML.")
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def list_files(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(directory.glob(pattern)))
    return files


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    config = read_yaml(data_path)
    root = (data_path.parent / config["path"]).resolve() if not Path(config["path"]).is_absolute() else Path(config["path"])
    names = config["names"]
    class_names = [names[index] for index in sorted(names)]

    print(f"Dataset root: {root}")
    print(f"Classes: {len(class_names)}")
    for split in ("train", "val", "test"):
        image_dir = root / config[split]
        label_dir = root / config[split].replace("images", "labels")
        images = list_files(image_dir, ("*.jpg", "*.jpeg", "*.png"))
        labels = list_files(label_dir, ("*.txt",))
        print(f"\n[{split}]")
        print(f"Images: {len(images)}")
        print(f"Labels: {len(labels)}")

        class_counter: Counter[int] = Counter()
        missing_labels = 0
        for label_file in labels:
            lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                missing_labels += 1
                continue
            for line in lines:
                class_id = int(line.split()[0])
                class_counter[class_id] += 1

        if missing_labels:
            print(f"Empty label files: {missing_labels}")
        if class_counter:
            print("Class distribution:")
            for class_id in sorted(class_counter):
                print(f"- {class_names[class_id]}: {class_counter[class_id]}")


if __name__ == "__main__":
    main()

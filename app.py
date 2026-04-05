"""Gradio web app for the Blind Assist currency detector."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from currency_detection.labels import rupee_value_for_label, to_spoken_label


DEFAULT_WEIGHTS = "last.pt" if (ROOT_DIR / "last.pt").exists() else "yolo11n.pt"
_MODEL_CACHE: dict[str, YOLO] = {}


def get_model(weights: str) -> YOLO:
    if weights not in _MODEL_CACHE:
        _MODEL_CACHE[weights] = YOLO(weights)
    return _MODEL_CACHE[weights]


def predict_image(image: np.ndarray, weights: str, conf: float, imgsz: int) -> tuple[np.ndarray, str]:
    if image is None:
        raise gr.Error("Please upload an image first.")

    model = get_model(weights)
    results = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)
    result = results[0]
    annotated = result.plot()

    summary_parts: list[str] = []
    total_value = 0
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        for class_id, score in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            label = result.names[int(class_id)]
            summary_parts.append(f"{to_spoken_label(label)} ({float(score):.2f})")
            total_value += rupee_value_for_label(label)

    if not summary_parts:
        return annotated, "No currency detected."

    lines = [
        f"Estimated total value: Rs {total_value}",
        "Detections:",
        *[f"- {item}" for item in summary_parts],
    ]
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "\n".join(lines)


with gr.Blocks(title="Blind Assist Currency Detector") as demo:
    gr.Markdown(
        """
        # Blind Assist: Currency Detection
        Upload an image of coins or notes and the model will detect denominations.
        For best results, use your trained `last.pt` or `best.pt` weights.
        """
    )
    with gr.Row():
        image_input = gr.Image(type="numpy", label="Currency Image")
        image_output = gr.Image(type="numpy", label="Detection Result")

    with gr.Row():
        weights_input = gr.Textbox(value=DEFAULT_WEIGHTS, label="Weights File")
        conf_input = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.5, label="Confidence Threshold")
        imgsz_input = gr.Slider(minimum=320, maximum=1280, step=32, value=640, label="Inference Size")

    summary_output = gr.Textbox(label="Summary", lines=10)
    run_button = gr.Button("Detect Currency")

    run_button.click(
        fn=predict_image,
        inputs=[image_input, weights_input, conf_input, imgsz_input],
        outputs=[image_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch()

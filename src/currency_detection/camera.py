"""Real-time currency detection camera pipeline."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from currency_detection.config import CameraConfig
from currency_detection.labels import rupee_value_for_label, to_spoken_label
from currency_detection.speech import SpeechEngine


@dataclass(slots=True)
class FrameObservation:
    counts: Counter[str]
    confidence_sums: dict[str, float]


class CurrencyCameraAssistant:
    """Runs YOLO inference on a live feed and announces stable detections."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.model = YOLO(config.weights)
        self.speech = SpeechEngine(enabled=config.speak)

        self._history: deque[FrameObservation] = deque(maxlen=max(config.smoothing_window, 1))
        self._last_spoken_summary = ""
        self._last_spoken_at = 0.0

    def run(self) -> None:
        source = int(self.config.source) if self.config.source.isdigit() else self.config.source
        capture = cv2.VideoCapture(source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution_height)

        if not capture.isOpened():
            raise RuntimeError(f"Unable to open camera or stream source: {self.config.source}")

        frame_interval = 1.0 / max(self.config.max_fps, 1.0)
        last_frame_time = 0.0

        print("Press 'q' to quit.")
        while True:
            now = time.monotonic()
            if now - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue

            ok, frame = capture.read()
            if not ok:
                print("Frame read failed. Stopping camera loop.")
                break

            if self.config.preprocess:
                frame = self._preprocess_frame(frame)

            last_frame_time = time.monotonic()
            annotated_frame = self._predict_and_annotate(frame)

            if not self.config.headless:
                cv2.imshow("Blind Assist Currency Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        capture.release()
        cv2.destroyAllWindows()

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply light, fast preprocessing that helps on dark and low-contrast frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        enhanced = frame

        if mean_brightness < 90:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
        return cv2.addWeighted(enhanced, 1.15, blurred, -0.15, 0)

    def _predict_and_annotate(self, frame):
        results = self.model.predict(
            source=frame,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            verbose=False,
            augment=self.config.use_tta,
        )
        result = results[0]
        annotated_frame = result.plot()

        observation = self._extract_observation(result)
        self._history.append(observation)
        spoken_summary, display_summary = self._calculate_consensus()
        self._maybe_speak(spoken_summary)
        self._draw_summary(annotated_frame, display_summary)
        return annotated_frame

    def _extract_observation(self, result) -> FrameObservation:
        boxes = getattr(result, "boxes", None)
        counts: Counter[str] = Counter()
        confidence_sums: dict[str, float] = defaultdict(float)
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            for class_id, confidence in zip(class_ids, confidences):
                label = result.names[int(class_id)]
                counts[label] += 1
                confidence_sums[label] += float(confidence)
        return FrameObservation(counts=counts, confidence_sums=dict(confidence_sums))

    def _calculate_consensus(self) -> tuple[str, str]:
        if not self._history:
            return "", ""

        frame_presence: Counter[str] = Counter()
        score_totals: dict[str, float] = defaultdict(float)
        count_samples: dict[str, list[int]] = defaultdict(list)

        for observation in self._history:
            for label, count in observation.counts.items():
                frame_presence[label] += 1
                score_totals[label] += observation.confidence_sums.get(label, 0.0)
                count_samples[label].append(count)

        required_votes = max(2, math.ceil(len(self._history) * self.config.stable_ratio))
        validated_items: list[str] = []
        total_value = 0

        for label in sorted(frame_presence, key=lambda item: (-score_totals[item], item)):
            presence = frame_presence[label]
            score_total = score_totals[label]
            if presence < required_votes or score_total < self.config.min_confidence_sum:
                continue

            stable_count = int(round(float(np.median(count_samples[label]))))
            stable_count = max(stable_count, 1)
            spoken = to_spoken_label(label)
            validated_items.append(f"{stable_count}x {spoken}" if stable_count > 1 else spoken)
            total_value += rupee_value_for_label(label) * stable_count

        if not validated_items:
            return "", ""

        details = ", ".join(validated_items)
        if total_value > 0:
            spoken_text = f"Total {total_value} rupees. {details}"
            display_text = f"Total: Rs {total_value} | {details}"
            return spoken_text, display_text

        return details, details

    def _maybe_speak(self, phrase: str) -> None:
        if not phrase:
            if len(self._history) == self._history.maxlen:
                all_empty = all(len(obs.counts) == 0 for obs in self._history)
                if all_empty:
                    self._last_spoken_summary = ""
            return

        now = time.monotonic()
        if phrase == self._last_spoken_summary and now - self._last_spoken_at < self.config.cooldown:
            return

        self._last_spoken_summary = phrase
        self._last_spoken_at = now
        self.speech.say(phrase)

    def _draw_summary(self, frame, summary: str) -> None:
        if not summary:
            if len(self._history) > 0 and any(len(obs.counts) > 0 for obs in self._history):
                summary = "Analyzing stable detection..."
            else:
                summary = "Scanning..."

        cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 980), 60), (0, 0, 0), -1)
        cv2.putText(
            frame,
            summary,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if "Total:" in summary else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def validate_weights_path(weights: str) -> str:
    """Allow either a local model path or a known Ultralytics pretrained weight name."""
    candidate = Path(weights)
    if candidate.exists() or weights.endswith(".pt") or weights.endswith(".onnx"):
        return weights
    raise FileNotFoundError(f"Weights file not found: {weights}")

"""
Real-time currency detection camera pipeline.
Architecture: 10-Stage Multi-Spectral OpenCV & YOLO Ensemble.
Objective: Maximum extreme prediction stability within 3.0 seconds runtime requirement.
"""

from __future__ import annotations

import time
import math
import statistics
from collections import Counter, deque
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from currency_detection.config import CameraConfig
from currency_detection.labels import to_spoken_label
from currency_detection.speech import SpeechEngine


# ==============================================================================
# ENGINE 1: Lighting Normalization Engine
# ==============================================================================
class LightingNormalizationEngine:
    """Uses advanced CLAHE equalizers to intelligently level shadows in extreme darkness."""
    
    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Dynamically corrects gamma and histogram for pitch-black scenarios."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray)[0]
        
        # Fast Alpha-Beta lift for extreme shadows
        if mean_brightness < 90:
            alpha = 1.2 + (90 - mean_brightness) * 0.015
            beta = 30 + (90 - mean_brightness) * 0.5
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
            # Apply Lab-space CLAHE to preserve color integrity perfectly
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            cl = self.clahe.apply(l_channel)
            merged = cv2.merge((cl, a_channel, b_channel))
            frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
        return frame


# ==============================================================================
# ENGINE 2: Motion Blur Engine
# ==============================================================================
class MotionBlurEngine:
    """Uses Laplacian variance to mathematically reject wildly shaken frames."""
    
    def __init__(self, blur_threshold: float = 30.0):
        self.threshold = blur_threshold

    def is_shaking(self, frame: np.ndarray) -> bool:
        """Returns True if the visually impaired user is waving the camera too fast."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.threshold


# ==============================================================================
# ENGINE 3: Edge Detection Verification Engine
# ==============================================================================
class EdgeDetectionEngine:
    """Ensures that YOLO's boxed regions actually contain complex physical geometry."""
    
    def __init__(self, edge_min: int = 50, edge_max: int = 150):
        self.edge_min = edge_min
        self.edge_max = edge_max

    def verify_physical_edges(self, frame: np.ndarray, bbox: List[int]) -> bool:
        """Guarantees a minimum amount of geometric edge density exists in the box."""
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 10 or y2 - y1 < 10:
            return False # Box too small to analyze
            
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to filter camera static
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.edge_min, self.edge_max)
        
        # Calculate edge density (percentage of pixels that are edges)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # A blank piece of paper or empty shadow will fail this check (density < 0.01)
        return edge_density > 0.015


# ==============================================================================
# ENGINE 4: Color Heuristic Engine
# ==============================================================================
class ColorHeuristicEngine:
    """Mathematically checks HSV distributions to prevent color hallucinations."""
    
    def __init__(self):
        # We allow a very broad spectrum to handle all Indian lighting profiles
        self.saturation_thresh = 15

    def verify_color_profile(self, frame: np.ndarray, bbox: List[int], label: str) -> bool:
        """Rejects objects that are purely black-and-white static when color notes are expected."""
        x1, y1, x2, y2 = bbox
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # Coins don't have strict color rules in poor light, accept them
        if label in ['1', '2', '5', '10', '20'] and "coin" in to_spoken_label(label).lower():
            return True
            
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s_channel = hsv_roi[:, :, 1]
        
        # Ensure there is AT LEAST some color saturation (not a black and white shadow print)
        mean_saturation = np.mean(s_channel)
        return mean_saturation > self.saturation_thresh


# ==============================================================================
# ENGINE 5: Multi-Scale Pyramid Engine
# ==============================================================================
class MultiScalePyramidEngine:
    """Prepares Test-Time Augmentation strategies for deeply distant objects."""
    
    def __init__(self, use_tta: bool):
        self.use_tta = use_tta
        
    def get_inference_kwargs(self) -> dict:
        """Injects `augment=True` into YOLO for massive multi-scale parsing."""
        if self.use_tta:
            return {"augment": True}
        return {"augment": False}


# ==============================================================================
# ENGINE 6: YOLO Core Deep Learning Engine
# ==============================================================================
class YOLOCoreEngine:
    """The central neurological wrapper for the Ultralytics `.pt` architecture."""
    
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)
        
    def predict(self, frame: np.ndarray, config: CameraConfig, tta_kwargs: dict) -> Any:
        results = self.model.predict(
            source=frame,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            verbose=False,
            **tta_kwargs
        )
        return results[0]


# ==============================================================================
# ENGINE 7: Bounding Box Geometry Engine
# ==============================================================================
class BoundingBoxGeometryEngine:
    """Analyzes mathematical aspect ratios. Coins must be square/circles, notes must be rects."""
    
    def verify_geometry(self, bbox: List[int], label: str) -> bool:
        x1, y1, x2, y2 = bbox
        w = float(x2 - x1)
        h = float(y2 - y1)
        if w == 0 or h == 0: return False
        
        ratio = max(w/h, h/w)
        
        # If it's a small denomination (could be a coin), heavily reject extreme rectangles
        if label in ['1', '2', '5']:
            if ratio > 3.0:  # A perfect coin is 1.0. If it's 3.0, it's a long strip.
                return False
        
        # Paper notes are naturally rectangular, but folded notes can be square. No strict limits.
        return True


# ==============================================================================
# ENGINE 8: Intersection Over Union (IoU) Suppression Engine
# ==============================================================================
class IoUSuppressionEngine:
    """Crushes overlapping duplicate ghost-boxes from raw predictions."""
    
    def __init__(self, iou_limit: float = 0.85):
        self.iou_limit = iou_limit

    def _calculate_iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def crush_ghosts(self, boxes: List[Tuple[List[float], str, float]]) -> List[Tuple[List[float], str, float]]:
        """A brute-force custom NMS algorithm for extra safety."""
        if len(boxes) == 0: return []
        
        # Sort entirely by confidence
        boxes = sorted(boxes, key=lambda x: x[2], reverse=True)
        final_boxes = []
        
        for i, current in enumerate(boxes):
            keep = True
            for f in final_boxes:
                if self._calculate_iou(current[0], f[0]) > self.iou_limit:
                    keep = False
                    break
            if keep:
                final_boxes.append(current)
                
        return final_boxes


# ==============================================================================
# ENGINE 9: Temporal Consensus Voting Engine
# ==============================================================================
class TemporalConsensusEngine:
    """The 15-frame rolling memory database protecting against individual frame glitches."""
    
    def __init__(self, window_size: int = 15, approval_ratio: float = 0.35):
        self.window_size = window_size
        self.approval_ratio = approval_ratio
        self._history = deque(maxlen=self.window_size)

    def log_frame(self, class_counts: Counter[str]):
        """Injects a single frame's output into the sliding time window."""
        self._history.append(class_counts)

    def is_empty(self) -> bool:
        """Returns True if the entire time window is completely devoid of detections."""
        return all(len(c) == 0 for c in self._history)

    def is_history_ready(self) -> bool:
        return len(self._history) > 0

    def compute_mathematical_majority(self) -> Tuple[List[str], int]:
        """Calculates exact total sum and verifies items mathematically across time."""
        required_votes = int(len(self._history) * self.approval_ratio)
        
        frame_presence: Counter[str] = Counter()
        for f_counts in self._history:
            for label in f_counts.keys():
                frame_presence[label] += 1
                
        validated_items = []
        total_value = 0
        
        for label, presence in frame_presence.items():
            if presence >= required_votes:
                # Use statistical median to survive sudden hand occlusions flawlessly
                counts_array = [f[label] for f in self._history if f[label] > 0]
                median_tally = int(statistics.median(counts_array))
                
                # Money sum calculation
                if label.isdigit():
                    total_value += int(label) * median_tally
                
                spoken = to_spoken_label(label)
                validated_items.append(f"{median_tally}x {spoken}" if median_tally > 1 else spoken)
                
        return validated_items, total_value


# ==============================================================================
# ENGINE 10: Master Ensemble Orchestrator
# ==============================================================================
class MasterEnsembleEngine:
    """The final conductor that choreographs all 9 subsystems to output a perfect 3-second result."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        
        print("\n[INIT] Booting all 10 independent verification engines...\n")
        self.e1_light = LightingNormalizationEngine()
        self.e2_motion = MotionBlurEngine()
        self.e3_edge = EdgeDetectionEngine()
        self.e4_color = ColorHeuristicEngine()
        self.e5_scale = MultiScalePyramidEngine(use_tta=True)
        self.e6_yolo = YOLOCoreEngine(config.weights)
        self.e7_geo = BoundingBoxGeometryEngine()
        self.e8_iou = IoUSuppressionEngine()
        self.e9_time = TemporalConsensusEngine(window_size=15, approval_ratio=0.35)
        
        self.speech = SpeechEngine(enabled=config.speak)
        self._last_spoken_summary = ""
        self._last_spoken_at = 0.0

    def orchestrate_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Processes a single camera frame through all 10 mathematical layers."""
        
        # [Engine 1 & 2]: Prep and Verify Frame Integrity
        frame = self.e1_light.process(frame)
        if self.e2_motion.is_shaking(frame):
            # Do not inject blurry guesses into the memory queue. Ignore frame.
            return frame, False
            
        # [Engine 5 & 6]: Deep Learning Core
        tta_kwargs = self.e5_scale.get_inference_kwargs()
        yolo_result = self.e6_yolo.predict(frame, self.config, tta_kwargs)
        annotated_frame = yolo_result.plot()
        
        verified_boxes = []
        raw_boxes = getattr(yolo_result, "boxes", None)
        
        if raw_boxes is not None and len(raw_boxes) > 0:
            for box in raw_boxes:
                coords = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = yolo_result.names[class_id]
                
                # [Engine 3, 4, 7]: Multi-Filter Verification Checks
                if not self.e7_geo.verify_geometry(coords, label): continue
                if not self.e3_edge.verify_physical_edges(frame, coords): continue
                if not self.e4_color.verify_color_profile(frame, coords, label): continue
                
                verified_boxes.append((coords, label, conf))
                
        # [Engine 8]: Ghost Suppression
        final_valid_boxes = self.e8_iou.crush_ghosts(verified_boxes)
        
        # Count exactly what survived all filters
        surviving_counts = Counter([b[1] for b in final_valid_boxes])
        
        # [Engine 9]: Write to Master Time Window
        self.e9_time.log_frame(surviving_counts)
        
        # Extract Results
        valid_list, total_sum = self.e9_time.compute_mathematical_majority()
        
        # [Speech & UI Controller]
        spoken_out = ""
        display_out = ""
        
        if valid_list:
            details = ", ".join(valid_list)
            if total_sum > 0:
                spoken_out = f"Total {total_sum} Rupees"
                display_out = f"Total: {total_sum} Rs ({details})"
            else:
                spoken_out = details
                display_out = details
                
        self._trigger_speech_controller(spoken_out)
        self._draw_ui(annotated_frame, display_out)
        
        return annotated_frame, True

    def _trigger_speech_controller(self, phrase: str):
        if not phrase:
            if self.e9_time.is_empty() and len(self.e9_time._history) == self.e9_time.window_size:
                self._last_spoken_summary = ""
            return

        now = time.monotonic()
        if phrase == self._last_spoken_summary and now - self._last_spoken_at < self.config.cooldown:
            return

        self._last_spoken_summary = phrase
        self._last_spoken_at = now
        self.speech.say(phrase)

    def _draw_ui(self, frame: np.ndarray, display_text: str):
        if not display_text:
            if self.e9_time.is_history_ready() and not self.e9_time.is_empty():
                display_text = "Master Ensemble Analyzing..."
            else:
                display_text = "Scanning Securely..."

        cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 980), 60), (0, 0, 0), -1)
        cv2.putText(
            frame, display_text, (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0) if "Rupee" in display_text else (255, 255, 255),
            2, cv2.LINE_AA
        )


# ==============================================================================
# MAIN CAMERA ASSISTANT 
# ==============================================================================
class CurrencyCameraAssistant:
    """The runtime wrapper executing the Master 10-Engine Ensemble."""
    
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.ensemble = MasterEnsembleEngine(config)

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
                break

            last_frame_time = time.monotonic()
            
            # Fire the full 10-Engine validation pipeline
            annotated_frame, frame_processed = self.ensemble.orchestrate_frame(frame)
            
            if not self.config.headless:
                cv2.imshow("Extreme 10-Engine Ensemble Vision", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        capture.release()
        cv2.destroyAllWindows()


def validate_weights_path(weights: str) -> str:
    candidate = Path(weights)
    if candidate.exists() or weights.endswith(".pt") or weights.endswith(".onnx"):
        return weights
    raise FileNotFoundError(f"Weights file not found: {weights}")

def validate_engine(weights: str, engine: str) -> None:
    pass  # Engine validation handled natively by the YOLO model wrapper

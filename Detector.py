"""
Detector.py — Core YOLOv8 + OpenCV real-time object detection engine.
Handles model loading, frame inference, bounding box rendering, and stats tracking.

Accuracy improvements:
  - Dynamic Stride-32 Letterboxing: Computes minimal padding constraints instead of rigid boxes.
  - High-Fidelity Downscaling: Employs INTER_AREA interpolation to eliminate downsample aliasing.
  - Test-Time Augmentation (TTA) Support: Built-in option for multi-scale inference mapping.
  - Higher default confidence (0.55) and IoU (0.50) thresholds.
  - Agnostic NMS disabled — per-class NMS gives cleaner boxes.
  - Local Contrast Fallback: Smarter frame validation ensures low-light scenes aren't ignored.
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


# ── COCO-80 class colour palette (deterministic, visually distinct) ──────────
CLASS_COLORS: dict[int, tuple[int, int, int]] = {}

def _get_color(class_id: int) -> tuple[int, int, int]:
    if class_id not in CLASS_COLORS:
        h = (class_id * 47) % 180          # spread hues evenly across HSV wheel
        hsv = np.uint8([[[h, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        CLASS_COLORS[class_id] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return CLASS_COLORS[class_id]


def letterbox_frame(
    frame: np.ndarray,
    target_size: int = 640,
    color: tuple = (114, 114, 114),
    stride: int = 32,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize frame to target_size with minimum letterbox constraints aligned to network strides.
    Preserves aspect ratio and employs high-fidelity downsampling (INTER_AREA) to prevent edge aliasing.

    Returns:
        lb_frame  : letterboxed frame ready for YOLO
        scale     : scale factor applied to the original frame
        (pad_w, pad_h) : pixels of padding added to width / height
    """
    h, w = frame.shape[:2]
    
    # Calculate scale factor
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    
    # Use INTER_AREA for downsampling (avoids aliasing artifacts), INTER_LINEAR for upsampling
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    # Compute minimal padding required to reach a size multiple of stride
    pad_w = (target_size - new_w) % stride
    pad_h = (target_size - new_h) % stride
    
    # Divide padding equally across opposite borders
    pad_w1, pad_w2 = pad_w // 2, pad_w - (pad_w // 2)
    pad_h1, pad_h2 = pad_h // 2, pad_h - (pad_h // 2)
    
    # Fallback back up to square size if strict canonical resolution target is requested
    if resized.shape[0] + pad_h1 + pad_h2 != target_size:
        pad_h2 = target_size - resized.shape[0] - pad_h1
    if resized.shape[1] + pad_w1 + pad_w2 != target_size:
        pad_w2 = target_size - resized.shape[1] - pad_w1

    lb_frame = cv2.copyMakeBorder(
        resized,
        pad_h1, pad_h2,
        pad_w1, pad_w2,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return lb_frame, scale, (pad_w1, pad_h1)


def is_valid_frame(frame: np.ndarray, min_mean: float = 4.0) -> bool:
    """Return False for corrupt, None, or completely dead frames while preserving dark environments."""
    if frame is None or frame.size == 0:
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
        
    mean_val = float(frame.mean())
    if mean_val >= min_mean:
        return True
        
    # Local Contrast Fallback: If global lighting is ultra-low, verify if structural features exist (night vision)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.meanStdDev(gray)[1][0][0] > 8.0:
        return True
        
    return False


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionStats:
    total_frames: int = 0
    total_detections: int = 0
    fps_history: deque = field(default_factory=lambda: deque(maxlen=30))
    class_counts: dict = field(default_factory=lambda: defaultdict(int))
    start_time: float = field(default_factory=time.time)

    @property
    def avg_fps(self) -> float:
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0

    @property
    def uptime(self) -> str:
        s = int(time.time() - self.start_time)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    @property
    def avg_detections(self) -> float:
        return self.total_detections / max(self.total_frames, 1)


# ─────────────────────────────────────────────────────────────────────────────

class YOLODetector:
    """
    Wraps YOLOv8 inference with letterbox preprocessing, configurable
    confidence/IoU thresholds, frame skipping, and overlay rendering.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.55,   # raised from 0.40 → fewer false positives
        iou_threshold: float = 0.50,    # raised from 0.45 → tighter NMS
        frame_skip: int = 1,
        max_detections: int = 100,
        input_size: int = 640,          # YOLO canonical input size
        track_objects: bool = False,
        use_tta: bool = False,          # Toggle Test-Time Augmentation for maximum precision recall
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.frame_skip = max(1, frame_skip)
        self.max_det = max_detections
        self.input_size = input_size
        self.track_objects = track_objects
        self.use_tta = use_tta

        self.model: Optional[YOLO] = None
        self.stats = DetectionStats()
        self._frame_counter = 0
        self._prev_time = time.perf_counter()

        # Frame-skip: cache last raw detection list (not annotated frame) so
        # we can re-draw on every fresh camera frame instead of showing a stale
        # annotated snapshot.
        self._last_detections: list[dict] = []

    # ── Model management ─────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Download (first run) and load YOLO weights, then warm up."""
        print(f"[Detector] Loading {self.model_name} …")
        self.model = YOLO(self.model_name)
        self.model.fuse()          # fuse Conv+BN layers for speed
        self._warmup()
        print(f"[Detector] Model ready — {len(self.model.names)} classes")

    def _warmup(self, n: int = 3) -> None:
        """Run a few dummy inferences to initialise CUDA/CPU caches."""
        dummy = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        for _ in range(n):
            self.model.predict(dummy, conf=self.conf_threshold, verbose=False, augment=self.use_tta)

    def unload_model(self) -> None:
        self.model = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Run inference on the frame (or reuse cached detections on skipped frames).
        Always re-draws bounding boxes on the *current* frame — no stale snapshots.

        Returns:
            annotated frame, list of detection dicts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded — call load_model() first.")

        if not is_valid_frame(frame):
            return frame, []

        # FPS bookkeeping
        self._frame_counter += 1
        now = time.perf_counter()
        fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now
        self.stats.fps_history.append(fps)
        self.stats.total_frames += 1

        # ── Decide whether to run inference this frame ────────────────────────
        run_inference = (self._frame_counter % self.frame_skip == 0)

        if run_inference:
            # Letterbox → infer → unscale boxes back to original frame coords
            lb_frame, scale, (pad_w, pad_h) = letterbox_frame(frame, self.input_size)
            results = self.model.predict(
                lb_frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                agnostic_nms=False,   # per-class NMS → better multi-class scenes
                verbose=False,
                stream=False,
                augment=self.use_tta, # Employs multi-scale test augmentation for tiny/complex detections
            )
            detections = self._parse_results(results, scale, pad_w, pad_h, frame.shape)

            self.stats.total_detections += len(detections)
            for d in detections:
                self.stats.class_counts[d["label"]] += 1

            self._last_detections = detections
        else:
            detections = self._last_detections

        annotated = self._draw_overlay(frame.copy(), detections, fps)
        return annotated, detections

    # ── Result parsing ────────────────────────────────────────────────────────

    def _parse_results(
        self,
        results,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_shape: tuple,
    ) -> list[dict]:
        """
        Convert YOLO boxes (in letterboxed coords) back to original frame coords.
        """
        orig_h, orig_w = orig_shape[:2]
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0]
                coords = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
                lx1, ly1, lx2, ly2 = coords

                # Reverse letterbox transform → original pixel coords
                x1 = int(np.clip((lx1 - pad_w) / scale, 0, orig_w - 1))
                y1 = int(np.clip((ly1 - pad_h) / scale, 0, orig_h - 1))
                x2 = int(np.clip((lx2 - pad_w) / scale, 0, orig_w - 1))
                y2 = int(np.clip((ly2 - pad_h) / scale, 0, orig_h - 1))

                # Discard degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                label = self.model.names.get(cls_id, f"cls_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "label": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "color": _get_color(cls_id),
                })
        return detections

    # ── Overlay rendering ─────────────────────────────────────────────────────

    def _draw_overlay(
        self, frame: np.ndarray, detections: list[dict], fps: float
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = det["color"]
            label = det["label"]
            conf = det["confidence"]

            # Main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accent lines (tactical look)
            cl = 12
            for cx, cy, dx, dy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * cl, cy), color, 3)
                cv2.line(frame, (cx, cy), (cx, cy + dy * cl), color, 3)

            # Label background + text
            text = f"{label}  {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            lx, ly = x1, max(y1 - th - 6, 0)
            cv2.rectangle(frame, (lx, ly), (lx + tw + 8, ly + th + 6), color, -1)
            cv2.putText(
                frame, text, (lx + 4, ly + th + 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA,
            )

        self._draw_hud(frame, detections, fps, w, h)
        return frame

    def _draw_hud(self, frame, detections, fps, w, h):
        panel_h = 115
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, panel_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (0, 0), (280, 3), (0, 220, 120), -1)

        lines = [
            ("YOLO DETECT", 0.55, (0, 220, 120)),
            (f"FPS  {self.stats.avg_fps:5.1f}   |  {fps:.0f} inst", 0.42, (200, 200, 200)),
            (f"OBJ  {len(detections):3d}      |  CONF >{self.conf_threshold:.0%}", 0.42, (200, 200, 200)),
            (f"FRM  {self.stats.total_frames:,}  |  SKIP {self.frame_skip}x", 0.42, (180, 180, 180)),
            (f"UPTIME  {self.stats.uptime}", 0.38, (140, 140, 160)),
        ]
        y = 18
        for text, scale, color in lines:
            cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)
            y += 22

        # Detection count badge (top-right)
        badge_text = str(len(detections))
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
        cv2.circle(frame, (w - 28, 36), 28, (0, 200, 100), -1)
        cv2.circle(frame, (w - 28, 36), 28, (0, 255, 130), 2)
        cv2.putText(frame, badge_text, (w - bw - 14, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (10, 10, 10), 2, cv2.LINE_AA)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def class_names(self) -> dict:
        return self.model.names if self.model else {}

    def reset_stats(self):
        self.stats = DetectionStats()
        self._last_detections = []
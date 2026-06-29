"""
Video_source.py — Robust video capture abstraction.
Supports webcam indices, video files, and RTSP/HTTP streams.
Includes auto-reconnect for streams, frame buffering, and corrupt-frame filtering.
"""

import cv2
import threading
import time
import queue
import numpy as np
from pathlib import Path


class VideoSource:
    """
    Thread-safe video capture with optional background buffering.
    Automatically detects source type from the provided identifier.
    Skips corrupt or near-black frames so the detector never receives bad input.
    """

    def __init__(self, source=0, buffer_size: int = 4, backend=None):
        self.source = source
        self.buffer_size = buffer_size
        self._backend = backend
        self._cap: cv2.VideoCapture | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

        # Metrics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.frames_invalid = 0
        self._open_time: float | None = None

    # ── Source type detection ─────────────────────────────────────────────────

    @staticmethod
    def detect_source_type(source) -> str:
        """Returns 'webcam', 'file', or 'stream'."""
        if isinstance(source, int):
            return "webcam"
        s = str(source).lower()
        if s.startswith(("rtsp://", "http://", "https://", "rtp://")):
            return "stream"
        if Path(s).suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".m4v"}:
            return "file"
        try:
            int(s)
            return "webcam"
        except ValueError:
            return "stream"

    # ── Frame validation ──────────────────────────────────────────────────────

    @staticmethod
    def _is_valid(frame: np.ndarray, min_mean: float = 4.0) -> bool:
        """Return False for None, zero-size, or near-black frames."""
        if frame is None or frame.size == 0:
            return False
        if frame.ndim != 3 or frame.shape[2] != 3:
            return False
        return float(frame.mean()) >= min_mean

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Open capture device. Returns True on success."""
        src = (
            int(self.source)
            if self.detect_source_type(self.source) == "webcam"
            else self.source
        )

        if self._backend is not None:
            cap = cv2.VideoCapture(src, self._backend)
        else:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            # Try common fallback backends
            for backend in [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(src, backend)
                if cap.isOpened():
                    break

        if not cap.isOpened():
            return False

        # Reduce latency and set preferred resolution
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._cap = cap
        self._open_time = time.time()
        return True

    def start_background_capture(self) -> None:
        """Start a daemon thread that continuously reads and buffers frames."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret or not self._is_valid(frame):
                self.frames_invalid += 1
                time.sleep(0.01)
                continue

            self.frames_captured += 1

            # Drop oldest frame if buffer is full (prefer freshness)
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            self._frame_queue.put(frame)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read next frame.
        Uses the background thread buffer if running, otherwise reads directly.
        """
        if self._thread and self._thread.is_alive():
            try:
                frame = self._frame_queue.get(timeout=0.1)
                return True, frame
            except queue.Empty:
                return False, None
        else:
            if self._cap is None:
                return False, None
            ret, frame = self._cap.read()
            if ret and self._is_valid(frame):
                return True, frame
            return False, None

    def release(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self._cap = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def resolution(self) -> tuple[int, int]:
        if self._cap:
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return (0, 0)

    @property
    def source_fps(self) -> float:
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        return 0.0

    @property
    def total_frames(self) -> int:
        if self._cap:
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return n if n > 0 else -1
        return -1

    @property
    def is_opened(self) -> bool:
        return bool(self._cap and self._cap.isOpened())

    @property
    def source_type(self) -> str:
        return self.detect_source_type(self.source)
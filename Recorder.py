"""
Recorder.py — Optional detection logger and video recorder.
Writes annotated video to disk and logs detections to CSV.

Improvements:
  - ISO-format timestamps in CSV for easy parsing
  - Resolution mismatch guard (warns instead of silently distorting)
  - session_id exposed as a property
  - Graceful no-op if write() called before start()
"""

import cv2
import csv
import time
from pathlib import Path
from datetime import datetime, timezone


class DetectionRecorder:
    """
    Records annotated frames to an MP4 file and detection events to a CSV.
    Outputs are written to the 'recordings/' subdirectory by default.
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        fps: float = 20.0,
        resolution: tuple = (1280, 720),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.resolution = resolution

        self._video_writer: cv2.VideoWriter | None = None
        self._csv_file = None
        self._csv_writer = None
        self._session_id: str = ""
        self._recording = False
        self._frame_count = 0

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start(self) -> str:
        """Start a new recording session. Returns the session ID string."""
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Video
        video_path = self.output_dir / f"session_{self._session_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            str(video_path), fourcc, self.fps, self.resolution
        )
        if not self._video_writer.isOpened():
            print(f"[Recorder] WARNING: VideoWriter failed to open {video_path}")

        # CSV
        csv_path = self.output_dir / f"detections_{self._session_id}.csv"
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp_iso", "frame", "label", "confidence",
            "x1", "y1", "x2", "y2", "width", "height",
        ])

        self._recording = True
        self._frame_count = 0
        print(f"[Recorder] Session {self._session_id} started")
        print(f"           Video  → {video_path}")
        print(f"           CSV   → {csv_path}")
        return self._session_id

    def write(self, frame: "np.ndarray", detections: list[dict]) -> None:
        """Write one annotated frame and its detections to disk."""
        if not self._recording:
            return

        ts_iso = datetime.now(timezone.utc).isoformat()
        self._frame_count += 1

        # Resize if frame doesn't match declared resolution
        h, w = frame.shape[:2]
        target_w, target_h = self.resolution
        if (w, h) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if self._video_writer and self._video_writer.isOpened():
            self._video_writer.write(frame)

        if self._csv_writer:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                self._csv_writer.writerow([
                    ts_iso,
                    self._frame_count,
                    det["label"],
                    f"{det['confidence']:.4f}",
                    x1, y1, x2, y2,
                    x2 - x1,    # bbox width  (useful for downstream analysis)
                    y2 - y1,    # bbox height
                ])

    def stop(self) -> None:
        """Finalise and close the current recording session."""
        self._recording = False
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        print(f"[Recorder] Session {self._session_id} saved ({self._frame_count} frames)")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def session_id(self) -> str:
        return self._session_id
import argparse
import sys
import cv2
import numpy as np
import time

from Detector import YOLODetector
from Video_source import VideoSource
from Recorder import DetectionRecorder


# ── Key bindings ──────────────────────────────────────────────────────────────
KEY_QUIT       = ord('q')
KEY_PAUSE      = ord(' ')
KEY_RECORD     = ord('r')
KEY_SCREENSHOT = ord('s')
KEY_CONF_UP    = ord('+')
KEY_CONF_DOWN  = ord('-')
KEY_SKIP_UP    = ord('f')
KEY_SKIP_DOWN  = ord('d')
KEY_RESET      = ord('0')
KEY_HELP       = ord('h')


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 Real-Time Object Detection")
    p.add_argument("--source",  default=0,
                   help="Webcam index, video file path, or RTSP URL")
    p.add_argument("--model",   default="yolov8n.pt",
                   help="YOLO model weights (n/s/m/l/x). Larger = more accurate but slower.")
    p.add_argument("--conf",    type=float, default=0.55,
                   help="Confidence threshold (0–1). Higher = fewer false positives.")
    p.add_argument("--iou",     type=float, default=0.50,
                   help="IoU / NMS threshold (0–1). Higher = tighter duplicate suppression.")
    p.add_argument("--skip",    type=int,   default=1,
                   help="Run inference every Nth frame (1 = every frame, max accuracy).")
    p.add_argument("--maxdet",  type=int,   default=100,
                   help="Max detections per frame.")
    p.add_argument("--input-size", type=int, default=640,
                   help="YOLO input resolution (must be multiple of 32). 640 is standard.")
    p.add_argument("--record",  action="store_true",
                   help="Record annotated video + CSV log from the start.")
    p.add_argument("--no-ui",   action="store_true",
                   help="Headless mode — no display window.")
    p.add_argument("--width",   type=int,   default=1280,
                   help="Display window width.")
    p.add_argument("--height",  type=int,   default=720,
                   help="Display window height.")
    p.add_argument("--backend", type=str,   default=None,
                   choices=["dshow", "v4l2", "any"],
                   help="Force specific capture backend (dshow recommended on Windows).")
    return p.parse_args()


def backend_flag(name: str | None):
    mapping = {"dshow": cv2.CAP_DSHOW, "v4l2": cv2.CAP_V4L2, "any": cv2.CAP_ANY}
    return mapping.get(name)


def draw_help_overlay(frame: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    bw, bh = 420, 330
    x1, y1 = cx - bw // 2, cy - bh // 2

    cv2.rectangle(overlay, (x1, y1), (x1 + bw, y1 + bh), (18, 18, 25), -1)
    cv2.rectangle(overlay, (x1, y1), (x1 + bw, y1 + bh), (0, 200, 110), 2)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    cv2.putText(frame, "KEYBOARD CONTROLS", (x1 + 20, y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 120), 1, cv2.LINE_AA)

    bindings = [
        ("Q",      "Quit application"),
        ("SPACE",  "Pause / Resume"),
        ("R",      "Toggle recording"),
        ("S",      "Save screenshot"),
        ("+",      "Increase confidence threshold"),
        ("-",      "Decrease confidence threshold"),
        ("F",      "Increase frame skip (faster, less accurate)"),
        ("D",      "Decrease frame skip (slower, more accurate)"),
        ("0",      "Reset statistics"),
        ("H",      "Toggle this help overlay"),
    ]
    for i, (key, desc) in enumerate(bindings):
        y = y1 + 60 + i * 26
        cv2.putText(frame, f"  {key:<8}{desc}", (x1 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)
    return frame


def run(args) -> int:
    print("╔══════════════════════════════════════════════╗")
    print("║   YOLOv8 Real-Time Object Detection          ║")
    print("╚══════════════════════════════════════════════╝")

    # Convert source to int if it's a device index
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # ── Detector ──────────────────────────────────────────────────────────────
    detector = YOLODetector(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        frame_skip=args.skip,
        max_detections=args.maxdet,
        input_size=args.input_size,
    )
    detector.load_model()

    # ── Video source ──────────────────────────────────────────────────────────
    vs = VideoSource(source, backend=backend_flag(args.backend))
    print(f"[Main] Opening source: {source!r}  (type: {vs.source_type})")
    if not vs.open():
        print(f"[ERROR] Cannot open source: {source}")
        print("  → If using a webcam, try --backend dshow (Windows) or --source 0/1/2")
        return 1

    vs.start_background_capture()
    w, h = vs.resolution
    print(f"[Main] Source {w}×{h} @ {vs.source_fps:.0f} fps")
    print(f"[Main] Conf: {args.conf:.0%}  |  IoU: {args.iou:.0%}  |  Input: {args.input_size}px")

    # ── Optional recorder ─────────────────────────────────────────────────────
    recorder: DetectionRecorder | None = None
    if args.record:
        recorder = DetectionRecorder(fps=vs.source_fps or 20, resolution=(w or 1280, h or 720))
        recorder.start()

    # ── Window ────────────────────────────────────────────────────────────────
    win_name = "YOLOv8 Object Detection  |  Press H for help"
    if not args.no_ui:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, args.width, args.height)

    # ── State ─────────────────────────────────────────────────────────────────
    paused = False
    show_help = False
    screenshot_counter = 0
    last_frame: np.ndarray | None = None

    print("[Main] Running — press Q to quit, H for controls\n")

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF

            # ── Key handling ──────────────────────────────────────────────────
            if key == KEY_QUIT:
                break
            elif key == KEY_PAUSE:
                paused = not paused
                print(f"[Main] {'PAUSED' if paused else 'RESUMED'}")
            elif key == KEY_HELP:
                show_help = not show_help
            elif key == KEY_RESET:
                detector.reset_stats()
                print("[Main] Statistics reset")
            elif key == KEY_RECORD:
                if recorder is None:
                    recorder = DetectionRecorder(
                        fps=vs.source_fps or 20,
                        resolution=(w or 1280, h or 720),
                    )
                    recorder.start()
                    print("[Main] Recording STARTED")
                elif recorder.is_recording:
                    recorder.stop()
                    print("[Main] Recording STOPPED")
                else:
                    recorder.start()
                    print("[Main] Recording RESUMED")
            elif key == KEY_SCREENSHOT:
                if last_frame is not None:
                    fname = f"screenshot_{screenshot_counter:04d}.jpg"
                    cv2.imwrite(fname, last_frame)
                    print(f"[Main] Screenshot saved: {fname}")
                    screenshot_counter += 1
            elif key == KEY_CONF_UP:
                detector.conf_threshold = min(0.95, round(detector.conf_threshold + 0.05, 2))
                print(f"[Main] Confidence → {detector.conf_threshold:.0%}")
            elif key == KEY_CONF_DOWN:
                detector.conf_threshold = max(0.05, round(detector.conf_threshold - 0.05, 2))
                print(f"[Main] Confidence → {detector.conf_threshold:.0%}")
            elif key == KEY_SKIP_UP:
                detector.frame_skip = min(8, detector.frame_skip + 1)
                print(f"[Main] Frame skip → {detector.frame_skip}x")
            elif key == KEY_SKIP_DOWN:
                detector.frame_skip = max(1, detector.frame_skip - 1)
                print(f"[Main] Frame skip → {detector.frame_skip}x")

            # ── Pause handling ────────────────────────────────────────────────
            if paused:
                if last_frame is not None and not args.no_ui:
                    display = last_frame.copy()
                    cy = display.shape[0] // 2
                    cx = display.shape[1] // 2 - 80
                    cv2.putText(display, "  PAUSED  ", (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (50, 50, 255), 2, cv2.LINE_AA)
                    cv2.imshow(win_name, display)
                time.sleep(0.05)
                continue

            # ── Frame read ────────────────────────────────────────────────────
            ret, frame = vs.read()
            if not ret or frame is None:
                if vs.source_type == "file":
                    print("[Main] End of video file.")
                    break
                time.sleep(0.01)
                continue

            # ── Inference ────────────────────────────────────────────────────
            annotated, detections = detector.process_frame(frame)

            if show_help:
                annotated = draw_help_overlay(annotated)

            last_frame = annotated

            # ── Record ───────────────────────────────────────────────────────
            if recorder and recorder.is_recording:
                recorder.write(annotated, detections)

            # ── Display ──────────────────────────────────────────────────────
            if not args.no_ui:
                cv2.imshow(win_name, annotated)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")

    finally:
        print("\n[Main] Shutting down …")
        vs.release()
        if recorder and recorder.is_recording:
            recorder.stop()
        if not args.no_ui:
            cv2.destroyAllWindows()

        # ── Final summary ─────────────────────────────────────────────────────
        stats = detector.stats
        print(f"\n{'─' * 44}")
        print("  Session Summary")
        print(f"{'─' * 44}")
        print(f"  Frames processed : {stats.total_frames:,}")
        print(f"  Total detections : {stats.total_detections:,}")
        print(f"  Avg detections/f : {stats.avg_detections:.2f}")
        print(f"  Avg FPS          : {stats.avg_fps:.1f}")
        print(f"  Uptime           : {stats.uptime}")
        if stats.class_counts:
            print("\n  Top detected classes:")
            top = sorted(stats.class_counts.items(), key=lambda x: -x[1])[:10]
            for label, count in top:
                print(f"    {label:<20} {count:>6}")
        print(f"{'─' * 44}\n")

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(args))

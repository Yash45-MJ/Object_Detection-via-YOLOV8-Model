import argparse
import time
import numpy as np
import cv2
from ultralytics import YOLO

from Detector import letterbox_frame


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 inference speed benchmark")
    p.add_argument("--models",     nargs="+", default=["yolov8n.pt", "yolov8s.pt"])
    p.add_argument("--frames",     type=int,   default=50,
                   help="Number of inference frames per model.")
    p.add_argument("--conf",       type=float, default=0.55,
                   help="Confidence threshold (should match your live-detection config).")
    p.add_argument("--iou",        type=float, default=0.50,
                   help="IoU / NMS threshold.")
    p.add_argument("--width",      type=int,   default=1280,
                   help="Simulated camera frame width.")
    p.add_argument("--height",     type=int,   default=720,
                   help="Simulated camera frame height.")
    p.add_argument("--input-size", type=int,   default=640,
                   help="YOLO input resolution (must be multiple of 32).")
    p.add_argument("--warmup",     type=int,   default=5,
                   help="Warm-up inferences before timing (discarded).")
    return p.parse_args()


def benchmark_model(
    model_name: str,
    frames: int,
    conf: float,
    iou: float,
    cam_w: int,
    cam_h: int,
    input_size: int,
    warmup: int,
) -> dict:
    model = YOLO(model_name)
    model.fuse()

    # ── Warm-up — using letterboxed frames ───────────────────────────────────
    dummy_cam = np.random.randint(30, 200, (cam_h, cam_w, 3), dtype=np.uint8)
    dummy_lb, _, _ = letterbox_frame(dummy_cam, input_size)
    for _ in range(warmup):
        model.predict(dummy_lb, conf=conf, iou=iou, verbose=False)

    # ── Timed runs ───────────────────────────────────────────────────────────
    latencies = []
    for _ in range(frames):
        # Randomised "natural-ish" synthetic frame (avoids pathological all-noise)
        cam_frame = np.random.randint(30, 200, (cam_h, cam_w, 3), dtype=np.uint8)
        lb_frame, _, _ = letterbox_frame(cam_frame, input_size)

        t0 = time.perf_counter()
        model.predict(lb_frame, conf=conf, iou=iou, verbose=False)
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies)
    return {
        "model":    model_name,
        "frames":   frames,
        "avg_fps":  1.0 / np.mean(latencies),
        "avg_ms":   np.mean(latencies) * 1000,
        "min_ms":   np.min(latencies) * 1000,
        "max_ms":   np.max(latencies) * 1000,
        "p95_ms":   np.percentile(latencies, 95) * 1000,
        "std_ms":   np.std(latencies) * 1000,
    }


def run(args):
    print(f"\nYOLOv8 Benchmark")
    print(f"  Camera   : {args.width}×{args.height}")
    print(f"  Input    : {args.input_size}×{args.input_size} (letterboxed)")
    print(f"  Frames   : {args.frames} per model  |  Warm-up: {args.warmup}")
    print(f"  Conf     : {args.conf:.0%}  |  IoU: {args.iou:.0%}")
    print("=" * 72)
    print(f"{'Model':<22} {'FPS':>7} {'Avg ms':>8} {'Min ms':>8} {'P95 ms':>8} {'Std ms':>8}")
    print("-" * 72)

    results = []
    for model_name in args.models:
        try:
            r = benchmark_model(
                model_name,
                args.frames,
                args.conf,
                args.iou,
                args.width,
                args.height,
                args.input_size,
                args.warmup,
            )
            results.append(r)
            print(
                f"  {r['model']:<20} "
                f"{r['avg_fps']:>7.1f} "
                f"{r['avg_ms']:>8.1f} "
                f"{r['min_ms']:>8.1f} "
                f"{r['p95_ms']:>8.1f} "
                f"{r['std_ms']:>8.1f}"
            )
        except Exception as e:
            print(f"  {model_name:<20} ERROR: {e}")

    print("=" * 72)
    if results:
        fastest = max(results, key=lambda r: r["avg_fps"])
        print(f"\n  Fastest model : {fastest['model']}  ({fastest['avg_fps']:.1f} FPS avg)\n")


if __name__ == "__main__":
    run(parse_args())

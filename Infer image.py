"""
Infer_image.py — Run YOLOv8 detection on one or more static images.

Accuracy improvements over the original:
  - Letterbox preprocessing (via Detector) preserves aspect ratio
  - Bounding boxes mapped back to original image coords for correct output
  - Skips corrupt/unreadable frames gracefully
  - Saves per-image JSON summary alongside annotated images

Usage:
    python Infer_image.py image.jpg
    python Infer_image.py img1.jpg img2.png --conf 0.55
    python Infer_image.py *.jpg --output results/ --show
    python Infer_image.py image.jpg --model yolov8s.pt   # more accurate model
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from Detector import YOLODetector


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 static image inference")
    p.add_argument("images", nargs="+",
                   help="Image file paths (supports wildcards via shell expansion)")
    p.add_argument("--model",      default="yolov8n.pt",
                   help="YOLO model weights. Use yolov8s/m/l/x for more accuracy.")
    p.add_argument("--conf",       type=float, default=0.55,
                   help="Confidence threshold (default 0.55).")
    p.add_argument("--iou",        type=float, default=0.50,
                   help="IoU / NMS threshold (default 0.50).")
    p.add_argument("--input-size", type=int,   default=640,
                   help="YOLO input resolution (must be multiple of 32, default 640).")
    p.add_argument("--output",     default="output_images",
                   help="Directory for annotated outputs.")
    p.add_argument("--save-json",  action="store_true",
                   help="Also save a JSON summary of detections for each image.")
    p.add_argument("--show",       action="store_true",
                   help="Display each result window (requires a GUI).")
    return p.parse_args()


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate input size
    if args.input_size % 32 != 0:
        print(f"[WARN] --input-size {args.input_size} is not a multiple of 32; "
              f"rounding up to {((args.input_size + 31) // 32) * 32}.")
        args.input_size = ((args.input_size + 31) // 32) * 32

    detector = YOLODetector(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.input_size,
    )
    detector.load_model()

    # ── Resolve image paths ───────────────────────────────────────────────────
    paths = []
    for img_path in args.images:
        p = Path(img_path)
        if not p.exists():
            print(f"[SKIP] File not found: {img_path}")
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            print(f"[SKIP] Unsupported extension '{p.suffix}': {img_path}")
            continue
        paths.append(p)

    if not paths:
        print("[ERROR] No valid images to process.")
        return

    print(f"\nProcessing {len(paths)} image(s) …\n")
    ok_count = 0

    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"[SKIP] Cannot read image: {p}")
            continue

        orig_h, orig_w = frame.shape[:2]
        annotated, detections = detector.process_frame(frame)

        # ── Save annotated image ──────────────────────────────────────────────
        out_path = out_dir / f"{p.stem}_detected{p.suffix}"
        cv2.imwrite(str(out_path), annotated)

        # ── Console summary ───────────────────────────────────────────────────
        labels = [f"{d['label']}({d['confidence']:.0%})" for d in detections]
        print(f"[OK]  {p.name}")
        print(f"      Size     : {orig_w}×{orig_h}")
        print(f"      Found    : {len(detections)} object(s): {', '.join(labels) or 'none'}")
        print(f"      Saved to : {out_path}")

        # ── Optional JSON ─────────────────────────────────────────────────────
        if args.save_json:
            json_path = out_dir / f"{p.stem}_detections.json"
            summary = {
                "image": str(p),
                "model": args.model,
                "conf_threshold": args.conf,
                "iou_threshold": args.iou,
                "original_size": [orig_w, orig_h],
                "detections": [
                    {
                        "label": d["label"],
                        "confidence": round(d["confidence"], 4),
                        "bbox": list(d["bbox"]),   # [x1, y1, x2, y2]
                    }
                    for d in detections
                ],
            }
            json_path.write_text(json.dumps(summary, indent=2))
            print(f"      JSON     : {json_path}")

        print()

        # ── Optional display ──────────────────────────────────────────────────
        if args.show:
            cv2.imshow(str(p.name), annotated)
            print("      Press any key for next image …")
            cv2.waitKey(0)

        ok_count += 1

    if args.show:
        cv2.destroyAllWindows()

    print(f"Done. {ok_count}/{len(paths)} image(s) processed → {out_dir}/")


if __name__ == "__main__":
    run(parse_args())
#!/usr/bin/env python3
"""Run 100x YOLO segmentation inference and save overlays, masks, and CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = BASE / "images"
DEFAULT_WEIGHTS = BASE / "flake_seg_100x_best.pt"
DEFAULT_OUT_DIR = BASE / "segmentation_outputs_100x"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def color_for_detection(class_name: str, cls_id: int) -> tuple[int, int, int]:
    lowered = class_name.lower()
    if lowered == "good":
        return (60, 210, 80)
    if lowered == "bad":
        return (255, 140, 60)
    palette = [(0, 220, 255), (220, 80, 220), (255, 220, 80)]
    return palette[cls_id % len(palette)]


def draw_label(img_bgr: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x0 = max(0, min(x, img_bgr.shape[1] - tw - 8))
    y0 = max(th + 8, y)
    cv2.rectangle(img_bgr, (x0, y0 - th - 6), (x0 + tw + 6, y0 + 2), (20, 20, 20), -1)
    cv2.putText(img_bgr, text, (x0 + 3, y0 - 3), font, scale, color, thick, cv2.LINE_AA)


def save_result(
    *,
    image_path: Path,
    out_dir: Path,
    model: YOLO,
    conf: float,
    iou: float,
    save_instances: bool,
) -> list[dict[str, str | int | float]]:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"SKIP unreadable: {image_path}")
        return []

    h_img, w_img = img_bgr.shape[:2]
    result = model.predict(img_bgr, conf=conf, iou=iou, retina_masks=True, verbose=False)[0]

    overlay = img_bgr.copy()
    combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    rows: list[dict[str, str | int | float]] = []

    masks = None if result.masks is None else result.masks.data.cpu().numpy()
    boxes = [] if result.boxes is None else list(result.boxes)
    names = getattr(model, "names", {}) or {}

    instance_dir = out_dir / "instance_masks" / image_path.stem
    if save_instances:
        instance_dir.mkdir(parents=True, exist_ok=True)

    for idx, box in enumerate(boxes):
        if masks is None or idx >= len(masks):
            continue
        score = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = str(names.get(cls_id, f"class_{cls_id}"))
        color = color_for_detection(class_name, cls_id)

        mask = cv2.resize(masks[idx], (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        binary = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area_px = float(cv2.contourArea(contour))
        if area_px <= 0:
            continue

        combined_mask = cv2.bitwise_or(combined_mask, binary)
        tint = np.zeros_like(overlay)
        tint[:, :] = color
        overlay = np.where(binary[:, :, None] > 0, (0.65 * overlay + 0.35 * tint).astype(np.uint8), overlay)
        cv2.drawContours(overlay, [contour], -1, color, thickness=2)

        x, y, w, h = cv2.boundingRect(contour)
        draw_label(overlay, x, max(14, y - 4), f"{class_name} {score:.2f}", color)
        if save_instances:
            cv2.imwrite(str(instance_dir / f"{idx:03d}_{class_name}.png"), binary)

        rows.append({
            "image": image_path.name,
            "instance": idx,
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(score, 5),
            "area_px": round(area_px, 2),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

    overlay_dir = out_dir / "overlays"
    mask_dir = out_dir / "masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_dir / f"{image_path.stem}_overlay.png"), overlay)
    cv2.imwrite(str(mask_dir / f"{image_path.stem}_mask.png"), combined_mask)
    print(f"{image_path.name}: {len(rows)} mask(s)")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 100x flake mask inference.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-instances", action="store_true")
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {args.image_dir}")
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    image_paths = iter_images(args.image_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit(f"No images found in: {args.image_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    all_rows: list[dict[str, str | int | float]] = []
    for image_path in image_paths:
        all_rows.extend(save_result(
            image_path=image_path,
            out_dir=args.out_dir,
            model=model,
            conf=args.conf,
            iou=args.iou,
            save_instances=args.save_instances,
        ))

    summary_path = args.out_dir / "detections.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "instance", "class_id", "class_name", "confidence", "area_px", "x", "y", "w", "h"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Outputs saved to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

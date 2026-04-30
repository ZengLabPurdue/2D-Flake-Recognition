#!/usr/bin/env python3
"""
Export 100x contour annotations to a YOLO segmentation dataset.

The outline tool saves polygons as JSON in contour_annotations/. This script
converts those polygons into normalized YOLO segmentation labels:

    <class_id> x1 y1 x2 y2 ... xn yn
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


BASE = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = BASE / "images"
DEFAULT_ANN_DIR = BASE / "contour_annotations"
DEFAULT_OUT_DIR = BASE / "training_images_100x"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class ExportStats:
    annotations_seen: int = 0
    images_exported: int = 0
    labels_written: int = 0
    contours_written: int = 0
    skipped_no_image: int = 0
    skipped_empty: int = 0


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_image(image_dir: Path, image_name: str, ann_stem: str) -> Path | None:
    candidates = []
    if image_name:
        candidates.append(image_dir / image_name)
    candidates.extend(image_dir / f"{ann_stem}{ext}" for ext in IMAGE_EXTS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def polygon_to_yolo_line(
    polygon: list,
    *,
    width: int,
    height: int,
    class_id: int,
    decimals: int,
) -> str | None:
    points: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()

    for pt in polygon:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x = max(0.0, min(float(pt[0]), float(width - 1)))
        y = max(0.0, min(float(pt[1]), float(height - 1)))
        key = (round(x), round(y))
        if key in seen:
            continue
        seen.add(key)
        points.append((x / width, y / height))

    if len(points) < 3:
        return None

    coords = []
    for x, y in points:
        coords.append(f"{x:.{decimals}f}")
        coords.append(f"{y:.{decimals}f}")
    return f"{class_id} " + " ".join(coords)


def export_annotations(
    *,
    image_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    class_id: int,
    decimals: int,
    copy_images: bool,
    dry_run: bool,
) -> ExportStats:
    stats = ExportStats()
    ann_paths = sorted(ann_dir.glob("*.json"))

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for ann_path in ann_paths:
        stats.annotations_seen += 1
        data = _load_json(ann_path)
        image_name = str(data.get("image", ""))
        contours = data.get("contours", [])
        if not contours:
            stats.skipped_empty += 1
            continue

        img_path = find_image(image_dir, image_name, ann_path.stem)
        if img_path is None:
            stats.skipped_no_image += 1
            print(f"SKIP no image for {ann_path.name}")
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        lines = [
            line
            for contour in contours
            if (line := polygon_to_yolo_line(
                contour,
                width=width,
                height=height,
                class_id=class_id,
                decimals=decimals,
            ))
        ]
        if not lines:
            stats.skipped_empty += 1
            continue

        if not dry_run:
            (out_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            if copy_images:
                shutil.copy2(img_path, out_dir / img_path.name)

        stats.images_exported += 1
        stats.labels_written += 1
        stats.contours_written += len(lines)
        print(f"{img_path.name}: {len(lines)} contour(s)")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert 100x contour JSON files to YOLO segmentation labels.",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--ann-dir", type=Path, default=DEFAULT_ANN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {args.image_dir}")
    if not args.ann_dir.is_dir():
        raise SystemExit(f"Annotation directory not found: {args.ann_dir}")

    stats = export_annotations(
        image_dir=args.image_dir,
        ann_dir=args.ann_dir,
        out_dir=args.out_dir,
        class_id=args.class_id,
        decimals=args.decimals,
        copy_images=not args.labels_only,
        dry_run=args.dry_run,
    )

    print("\nExport summary")
    print(f"  Annotation files seen : {stats.annotations_seen}")
    print(f"  Images exported       : {stats.images_exported}")
    print(f"  Label files written   : {stats.labels_written}")
    print(f"  Contours written      : {stats.contours_written}")
    print(f"  Skipped no image      : {stats.skipped_no_image}")
    print(f"  Skipped empty/invalid : {stats.skipped_empty}")
    print(f"  Output directory      : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

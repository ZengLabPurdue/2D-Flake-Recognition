#!/usr/bin/env python3
"""
Batch process all images with the filtered-sensitive overlay pipeline using 2x2 binning.
Uses settings from batch_filtered_settings.json (batch_filtered_tuner_ui.py).

Input: images/x20_images/ (original, unbinned images)
Output: images/filtered_sensitive_overlays/ (compatible with flake_extraction_pipeline)
  - {stem}_binned_filtered_overlay.png
  - {stem}_binned_filtered_contours.png

Usage:
    python batch_filtered_sensitive_overlays_2x2.py [input_dir] [-o output_dir]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline import ContourPipeline

BATCH_FILTERED_SETTINGS_PATH = Path(__file__).resolve().parent / "batch_filtered_settings.json"
_DEFAULT_PARAMS = {
    "blur_sigma": 0.6,
    "combine_canny_low_range": False,
    "canny_low_low": 0,
    "canny_low_high": 100,
    "canny_low": 10,
    "canny_high": 50,
    "min_area": 0,
    "filter_nested_contours": False,
    "edge_method": "canny_h",
    "edge_methods": ["canny_h"],
    "use_h_channel": True,
    "preprocessing": "full",
    "bridge_gap_factor": 0.15,
    "force_close_factor": 0.3,
    "line_thickness": 4,
    "close_kernel_divisor": 100,
    "morph_margin": 15,
    "corner_radius": 2,
    "min_area_contour": 200,
    "filter_yellow": False,
    "filter_blue": False,
    "filter_good_flake": False,
    "yellow_v_min": 70,
    "blue_h_min": 100,
    "good_flake_g_max": 130,
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_filtered_overlay_params() -> dict:
    """Load params from batch_filtered_settings.json if it exists."""
    if not BATCH_FILTERED_SETTINGS_PATH.exists():
        return _DEFAULT_PARAMS.copy()
    try:
        with open(BATCH_FILTERED_SETTINGS_PATH) as f:
            loaded = json.load(f)
        out = _DEFAULT_PARAMS.copy()
        for k, v in loaded.items():
            if k in out:
                out[k] = v
        return out
    except Exception:
        return _DEFAULT_PARAMS.copy()


def save_filtered_overlay_params(params: dict) -> None:
    """Save params to batch_filtered_settings.json."""
    with open(BATCH_FILTERED_SETTINGS_PATH, "w") as f:
        json.dump(params, f, indent=2)


def process_one(input_path: Path, output_dir: Path, params: dict) -> bool:
    """Process a single image and save overlay + contours to output_dir."""
    if not input_path.exists():
        print(f"  Skip (not found): {input_path.name}")
        return False

    try:
        original_img = np.array(Image.open(input_path))
        if original_img.ndim == 2:
            original_img = np.stack([original_img, original_img, original_img], axis=-1)
        if original_img.shape[2] == 4:
            original_img = original_img[:, :, :3]

        pipeline = ContourPipeline()
        result = pipeline.run(original_img, params, return_edges=True, return_binary=False)
        filtered_contours = result["contours"]
        overlay = result["overlay"]
        edges = result.get("edges")
        orig_h, orig_w = original_img.shape[:2]

        # Contours-only canvas (flake_extraction reads this)
        contours_canvas = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        lt = max(2, int(min(orig_w, orig_h) / 500))
        if filtered_contours:
            cv2.drawContours(contours_canvas, filtered_contours, -1, (255, 255, 255), lt)

        stem = input_path.stem
        overlay_path = output_dir / f"{stem}_binned_filtered_overlay.png"
        contours_path = output_dir / f"{stem}_binned_filtered_contours.png"
        Image.fromarray(overlay).save(overlay_path)
        Image.fromarray(contours_canvas).save(contours_path)
        if edges is not None:
            if edges.ndim == 3:
                edges_gray = edges[:, :, 0]
            else:
                edges_gray = edges
            Image.fromarray(edges_gray).save(output_dir / f"{stem}_binned_filtered_edges.png")

        print(f"  ✓ {input_path.name} -> {len(filtered_contours)} contours")
        return True

    except Exception as e:
        print(f"  ✗ {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch filtered-sensitive overlay pipeline (uses contour tuner settings)."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("images/x20_images"),
        help="Input directory (default: images/x20_images)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("images/filtered_sensitive_overlays"),
        help="Output directory (default: images/filtered_sensitive_overlays)",
    )
    args = parser.parse_args()

    params = load_filtered_overlay_params()
    if BATCH_FILTERED_SETTINGS_PATH.exists():
        print("Using settings from batch_filtered_settings.json")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 1

    paths = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        print(f"No images in {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(paths)} images from {input_dir}")
    print(f"Output: {output_dir}/")
    print()

    ok = 0
    for p in paths:
        if process_one(p, output_dir, params):
            ok += 1

    print(f"\nDone. {ok}/{len(paths)} images processed -> {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

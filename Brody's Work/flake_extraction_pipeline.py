#!/usr/bin/env python3
"""
Extract individual flakes from images:
1. Load contours from filtered sensitive overlays (or compute if not available)
2. Build bounding box around each contour
3. Crop and run through background eliminator (extract_background) to isolate flake
4. Run color clustering to determine flake type
5. Filter out contours where a majority of interior pixels have HSV H in 25-45 (yellow)

Usage:
    python flake_extraction_pipeline.py [input_dir] [-o output_dir]
"""
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from batch_filtered_sensitive_overlays_2x2 import load_filtered_overlay_params
from extract_background import extract_background
from pipeline import ContourPipeline

MIN_AREA = 800
BBOX_PADDING = 20  # Pixels around each bounding box
ROBUST_CONTOURS_DIR = Path("robust_contours_and_masks_no_gap_close")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
N_COLOR_CLUSTERS = 3  # For k-means on flake colors

# HSV filter: remove contours where majority of interior pixels have H (hue) in yellow range
# OpenCV H 0-179: yellow ~25-45 (50-70 on 0-360 scale maps to 25-35)
HSV_HUE_LO = 25
HSV_HUE_HI = 45
MAJORITY_THRESHOLD = 0.5


def filter_hsv_contours(contours: list, img: np.ndarray) -> list:
    """Remove contours where a majority of interior pixels have HSV H in 25-45 (yellow)."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    h, w = img.shape[:2]
    img_u8 = np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]

    filtered = []
    for c in contours:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        interior_H = H[mask > 127]
        n = len(interior_H)
        if n < 30:
            filtered.append(c)
            continue
        in_range = np.sum((interior_H >= HSV_HUE_LO) & (interior_H <= HSV_HUE_HI))
        fraction = in_range / n
        if fraction >= MAJORITY_THRESHOLD:
            continue
        filtered.append(c)
    return filtered


def get_filtered_contours_and_edges(
    original_img: np.ndarray,
    stem: str,
    contours_source_dir: Path = None,
    apply_hsv_filter: bool = True,
) -> tuple[list, np.ndarray | None]:
    """
    Run pipeline (preprocess + edge detection) to get contours and edges fresh each time.
    Returns (contours, edges). edges may be at binned resolution.
    """
    orig = original_img.copy()
    if orig.ndim == 2:
        orig = np.stack([orig, orig, orig], axis=-1)
    if orig.shape[2] == 4:
        orig = orig[:, :, :3]
    orig = np.clip(orig, 0, 255).astype(np.uint8)
    params = load_filtered_overlay_params()
    pipeline = ContourPipeline()
    result = pipeline.run(orig, params, return_edges=False, return_binary=True)
    all_contours = result["contours"]
    # Use gap-closed binary (same edges that form contours) - raw edges have gaps
    binary = result.get("binary")
    if apply_hsv_filter:
        all_contours = filter_hsv_contours(all_contours, orig)
    return all_contours, binary


def _kmeans_simple(pixels: np.ndarray, n_clusters: int, max_iter: int = 50):
    """Simple k-means using numpy only. Returns cluster centers and labels."""
    n, d = pixels.shape
    rng = np.random.default_rng(42)
    centers = pixels[rng.choice(n, size=n_clusters, replace=False)].copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            pixels[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers, labels


def cluster_colors(flake_pixels_rgb: np.ndarray, n_clusters: int = N_COLOR_CLUSTERS):
    """Run k-means on flake pixel colors. Returns dominant colors and labels."""
    if len(flake_pixels_rgb) < n_clusters:
        return [], []
    pixels = np.clip(flake_pixels_rgb.astype(np.float32), 0, 255)
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if len(pixels) < n_clusters:
        return [], []
    n_clusters = min(n_clusters, len(pixels))
    centers, labels = _kmeans_simple(pixels, n_clusters)
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    counts = np.bincount(labels)
    order = np.argsort(-counts)
    return centers[order], labels


def process_image(input_path: Path, output_dir: Path, apply_hsv_filter: bool = True) -> int:
    """Process one image: extract flakes, run background eliminator, cluster colors."""
    if not input_path.exists():
        print(f"  Skip (not found): {input_path.name}")
        return False

    try:
        original_img = np.array(Image.open(input_path))
        if original_img.ndim == 2:
            original_img = np.stack([original_img] * 3, axis=-1)
        if original_img.shape[2] == 4:
            original_img = original_img[:, :, :3]
        orig_h, orig_w = original_img.shape[:2]

        contours, edges = get_filtered_contours_and_edges(
            original_img, stem=input_path.stem, apply_hsv_filter=apply_hsv_filter
        )
        if not contours:
            print(f"  {input_path.name}: no contours")
            return 0

        img_out_dir = output_dir / input_path.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        # Edges overlay (gap-closed binary - edges that form contours, so they connect)
        if edges is not None:
            bin_edges = (edges > 0) if edges.dtype == bool else (edges > 127)
            if bin_edges.shape[:2] != (orig_h, orig_w):
                bin_edges = cv2.resize(
                    bin_edges.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                ) > 0
            edges_overlay = original_img.copy()
            edges_overlay[bin_edges] = [0, 255, 255]
            Image.fromarray(edges_overlay).save(img_out_dir / "edges_overlay.png")

        # Contour overlay
        overlay = original_img.copy()
        line_thickness = max(2, int(min(orig_w, orig_h) / 500))
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), line_thickness)
        Image.fromarray(overlay).save(img_out_dir / "contour_overlay.png")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            robust_dir = ROBUST_CONTOURS_DIR / input_path.stem
            robust_bg_path = robust_dir / f"{input_path.stem}_masked_background_only.png"
            robust_mask_path = robust_dir / f"{input_path.stem}_mask_shapes.png"
            has_robust = robust_bg_path.exists() and robust_mask_path.exists()
            if has_robust:
                robust_bg_img = np.array(Image.open(robust_bg_path))
                if robust_bg_img.ndim == 2:
                    robust_bg_img = np.stack([robust_bg_img] * 3, axis=-1)
                robust_mask_img = np.array(Image.open(robust_mask_path))
                if robust_mask_img.ndim == 3:
                    robust_mask_img = robust_mask_img[:, :, 0]

            for idx, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                pad = BBOX_PADDING
                side = max(w, h) + 2 * pad
                cx, cy = x + w // 2, y + h // 2
                half = side // 2
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(orig_w, x1 + side)
                y2 = min(orig_h, y1 + side)
                x1 = max(0, x2 - side)
                y1 = max(0, y2 - side)
                crop = original_img[y1:y2, x1:x2].copy()

                crop_path = tmp_path / f"crop_{idx}.png"
                flake_dir = img_out_dir / "crops" / f"flake_{idx:03d}"
                flake_dir.mkdir(parents=True, exist_ok=True)

                Image.fromarray(crop).save(flake_dir / "crop.png")

                overlay_crop = crop.copy()
                rel_contour = contour.copy()
                rel_contour[:, 0, 0] -= x1
                rel_contour[:, 0, 1] -= y1
                line_thick = max(1, min(crop.shape[0], crop.shape[1]) // 100)
                cv2.drawContours(overlay_crop, [rel_contour], -1, (0, 255, 0), line_thick)
                Image.fromarray(overlay_crop).save(flake_dir / "overlay.png")

                if has_robust:
                    robust_bg_crop = robust_bg_img[y1:y2, x1:x2].copy()
                    robust_mask_crop = robust_mask_img[y1:y2, x1:x2]
                    bg_mask = (robust_mask_crop < 127)
                    n_bg = int(np.sum(bg_mask))
                    if n_bg < 10:
                        print(f"    flake_{idx:03d}: no background left after masking, using unmasked crop")
                        bg_crop_to_save = robust_bg_crop
                    else:
                        bg_crop_to_save = np.zeros_like(robust_bg_crop)
                        bg_crop_to_save[bg_mask] = robust_bg_crop[bg_mask]
                    Image.fromarray(bg_crop_to_save).save(flake_dir / "background_crop.png")

                import io
                import contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        Image.fromarray(crop).save(crop_path)
                        _, mask_inv, _ = extract_background(
                            crop_path,
                            output_path=tmp_path / f"bg_{idx}",
                            bin_factor=1,
                            canny_low=5,
                            canny_high=25,
                            min_area=5,
                            bridge_max_gap=30,
                            force_close_max_gap=40,
                        )
                        flake_mask = 255 - mask_inv
                    except Exception:
                        rel_contour = contour.copy()
                        rel_contour[:, 0, 0] -= x1
                        rel_contour[:, 0, 1] -= y1
                        flake_mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
                        cv2.fillPoly(flake_mask, [rel_contour], 255)

                flake_pixels = crop[flake_mask > 127]
                if len(flake_pixels) < 10:
                    continue

                centers, _ = cluster_colors(flake_pixels, N_COLOR_CLUSTERS)
                dominant = centers[0] if len(centers) > 0 else np.array([0, 0, 0])

                masked_flake = np.zeros_like(crop)
                masked_flake[flake_mask > 127] = crop[flake_mask > 127]
                Image.fromarray(masked_flake).save(flake_dir / "masked.png")
                # mask.png: flake region = black, background = original crop pixels
                mask_img = crop.copy()
                mask_img[flake_mask > 127] = 0
                Image.fromarray(mask_img).save(flake_dir / "mask.png")

                swatch = np.tile(dominant, (30, 30, 1)).astype(np.uint8)
                Image.fromarray(swatch).save(flake_dir / "color.png")

                color_log_path = img_out_dir / "flake_colors.csv"
                header = "flake_idx,r,g,b,bbox_x,bbox_y,bbox_w,bbox_h\n"
                row = f"{idx},{dominant[0]},{dominant[1]},{dominant[2]},{x},{y},{w},{h}\n"
                if idx == 0:
                    with open(color_log_path, "w") as f:
                        f.write(header)
                with open(color_log_path, "a") as f:
                    f.write(row)

        summary_path = img_out_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Image: {input_path.name}\n")
            f.write(f"Contours: {len(contours)}\n")

        print(f"  ✓ {input_path.name} -> {len(contours)} flakes")
        return len(contours)

    except Exception as e:
        print(f"  ✗ {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract flakes: contours → bbox crop → background eliminator → color clustering"
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
        default=Path("flake_extraction_output"),
        help="Output directory (default: flake_extraction_output)",
    )
    parser.add_argument(
        "--no-filter",
        dest="no_filter",
        action="store_true",
        help="Disable HSV filter; keep all contours",
    )
    parser.add_argument(
        "--only",
        metavar="NAME",
        help="Only process image(s) whose stem matches (e.g. 1-5-20)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 1

    paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.only:
        paths = [p for p in paths if p.stem == args.only]
        if not paths:
            print(f"No image with stem '{args.only}' in {input_dir}")
            return 1
    if not paths:
        print(f"No images in {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(paths)} images from {input_dir}")
    print(f"Output: {output_dir}/")
    print()

    apply_filter = not args.no_filter
    if not apply_filter:
        print("HSV filter: disabled (keeping all contours)")
    total = 0
    for p in paths:
        total += process_image(p, output_dir, apply_hsv_filter=apply_filter)

    print(f"\nDone. {total} flakes extracted -> {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

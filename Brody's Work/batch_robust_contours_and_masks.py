#!/usr/bin/env python3
"""
Batch process all images with robust contour detection and create masks.
Processes all images in images/x20_images/ and creates:
- Contours overlay
- Shape masks (white shapes on black background)
- Background masks (black shapes on white background)
- All intermediate outputs

Usage:
    python batch_robust_contours_and_masks.py [input_dir] [output_dir]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from canny_filter_tuner import compute_filtered_canny
from edge_to_contour_methods import (
    _ensure_binary,
    force_close_open_chains,
    smart_connect_endpoints_bridged,
    _find_endpoints,
)
from preprocess import clahe_then_nlmeans

# Settings file: written by contour_tuner_ui.py when you click "Update Now"
CONTOUR_SETTINGS_PATH = Path(__file__).resolve().parent / "robust_contours_settings.json"

_DEFAULT_PARAMS = {
    "blur_sigma": 0.1,
    "canny_low": 0,
    "canny_high": 50,
    "use_h_channel": True,
    "edge_method": "canny_h",
    "preprocessing": "full",
    "bridge_gap_factor": 0.25,
    "force_close_factor": 0.4,
    "line_thickness": 5,
    "close_kernel_divisor": 150,
    "morph_margin": 30,
    "corner_radius": 8,
    "binning": True,
}


def load_contour_params() -> dict:
    """Load params from robust_contours_settings.json if it exists, else return defaults."""
    if not CONTOUR_SETTINGS_PATH.exists():
        return _DEFAULT_PARAMS.copy()
    try:
        with open(CONTOUR_SETTINGS_PATH) as f:
            loaded = json.load(f)
        out = _DEFAULT_PARAMS.copy()
        for k, v in loaded.items():
            if k in out:
                out[k] = v
        return out
    except Exception:
        return _DEFAULT_PARAMS.copy()


def save_contour_params(params: dict) -> None:
    """Save params to robust_contours_settings.json for batch_robust_contours_and_masks.py."""
    with open(CONTOUR_SETTINGS_PATH, "w") as f:
        json.dump(params, f, indent=2)


def bin_image_by_2(arr: np.ndarray) -> np.ndarray:
    """
    Bin a 2D array by 2x2 (2x2 -> 1 pixel) using mean.
    If dimensions are not divisible by 2, trim to nearest multiple of 2.
    """
    h, w = arr.shape[:2]
    # Trim to dimensions divisible by 2
    h_trim = h - (h % 2)
    w_trim = w - (w % 2)
    arr = arr[:h_trim, :w_trim]

    if arr.ndim == 2:
        # Shape (H, W) -> (H/2, 2, W/2, 2) -> mean over (1, 3) -> (H/2, W/2)
        binned = arr.reshape(h_trim // 2, 2, w_trim // 2, 2).mean(axis=(1, 3))
        return binned
    else:
        # Multi-channel (e.g. RGB): bin each channel
        binned = np.stack(
            [bin_image_by_2(arr[:, :, c]) for c in range(arr.shape[2])], axis=-1
        )
        return binned


def compute_h_channel_edges(img: np.ndarray, blur_sigma: float = 0.6, canny_low: int = 10, canny_high: int = 50) -> np.ndarray:
    """Compute Canny edges on the H (Hue) channel in HSV. Returns binary uint8 (0 or 255)."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    blurred = cv2.GaussianBlur(h_channel, (0, 0), max(0.01, blur_sigma))
    return cv2.Canny(blurred, canny_low, canny_high)


def morph_extend_to_border(binary: np.ndarray, bin_h: int, bin_w: int, margin: int = 15, corner_radius: int = 2) -> np.ndarray:
    """Extend edges toward image border when they're within margin pixels. Helps close border-touching contours."""
    ext = binary.copy()
    if margin <= 0:
        return ext
    if np.any(binary[:, :margin] > 127):
        left_col = np.any(binary[:, :margin] > 127, axis=1)
        if np.any(left_col):
            y_min, y_max = np.where(left_col)[0][[0, -1]]
            ext[y_min : y_max + 1, 0] = 255
    if np.any(binary[:, -margin:] > 127):
        right_col = np.any(binary[:, -margin:] > 127, axis=1)
        if np.any(right_col):
            y_min, y_max = np.where(right_col)[0][[0, -1]]
            ext[y_min : y_max + 1, bin_w - 1] = 255
    if np.any(binary[:margin, :] > 127):
        top_row = np.any(binary[:margin, :] > 127, axis=0)
        if np.any(top_row):
            x_min, x_max = np.where(top_row)[0][[0, -1]]
            ext[0, x_min : x_max + 1] = 255
    if np.any(binary[-margin:, :] > 127):
        bottom_row = np.any(binary[-margin:, :] > 127, axis=0)
        if np.any(bottom_row):
            x_min, x_max = np.where(bottom_row)[0][[0, -1]]
            ext[bin_h - 1, x_min : x_max + 1] = 255

    # Corner fix: only when white pixels touch the actual corner (within corner_radius px), connect the corner
    cr = max(0, corner_radius)
    if cr > 0 and np.any(binary[:cr, :cr] > 127):
        ext[0, 0] = 255
    if cr > 0 and np.any(binary[:cr, -cr:] > 127):
        ext[0, bin_w - 1] = 255
    if cr > 0 and np.any(binary[-cr:, :cr] > 127):
        ext[bin_h - 1, 0] = 255
    if cr > 0 and np.any(binary[-cr:, -cr:] > 127):
        ext[bin_h - 1, bin_w - 1] = 255

    return ext


def filter_nested_contours(contours):
    """
    Remove nested contours - if a smaller contour is completely inside a larger one,
    keep only the larger contour.
    Returns filtered list of outermost contours only.
    """
    if not contours or len(contours) <= 1:
        return contours
    
    # Sort by area (largest first)
    contours_with_areas = [(cv2.contourArea(c), c) for c in contours]
    contours_with_areas.sort(reverse=True, key=lambda x: x[0])
    
    filtered_contours = []
    
    for i, (area_i, contour_i) in enumerate(contours_with_areas):
        is_nested = False
        
        # Check if this contour is inside any larger contour we've already kept
        for area_j, contour_j in contours_with_areas[:i]:
            # Get the center point of contour_i
            M = cv2.moments(contour_i)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check if the center point is inside the larger contour
            center_result = cv2.pointPolygonTest(contour_j, (cx, cy), False)
            
            if center_result >= 0:  # Center is inside or on boundary
                # Additional check: test if most points of contour_i are inside contour_j
                test_points = contour_i.reshape(-1, 2)
                points_inside = 0
                points_to_test = min(20, len(test_points))  # Test up to 20 points
                step = max(1, len(test_points) // points_to_test)
                
                for point in test_points[::step]:
                    # Contour points are (y, x), but pointPolygonTest expects (x, y)
                    result = cv2.pointPolygonTest(contour_j, (int(point[1]), int(point[0])), False)
                    if result >= 0:
                        points_inside += 1
                
                # If most points are inside, consider it nested
                if points_inside >= points_to_test * 0.8:  # 80% threshold
                    is_nested = True
                    break
        
        if not is_nested:
            filtered_contours.append(contour_i)
    
    return filtered_contours


def process_image_with_masks(
    input_path: Path,
    output_dir: Path,
    no_binning: bool = False,
    params: dict | None = None,
    no_gap_close: bool = False,
):
    """Process a single image with robust contour detection and create masks."""
    if params is None:
        params = load_contour_params()
    # CLI --no-binning overrides config binning
    use_binning = params.get("binning", True) and not no_binning

    if not input_path.exists():
        print(f"Error: Image not found: {input_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}")
    
    # Step 1: Load original image
    print("\n1. Loading image...")
    try:
        original_img = np.array(Image.open(input_path))
        if original_img.ndim == 2:
            original_img = np.stack([original_img, original_img, original_img], axis=-1)
        if original_img.shape[2] == 4:
            original_img = original_img[:, :, :3]
        orig_h, orig_w = original_img.shape[:2]
        print(f"   Original size: {orig_w}x{orig_h}")
    except Exception as e:
        print(f"   Error loading image: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Bin image (or skip if --no-binning)
    if not use_binning:
        print("\n2. No binning (full resolution)...")
        work_img = original_img.copy()
        bin_h, bin_w = orig_h, orig_w
        scale_x = 1.0
        scale_y = 1.0
    else:
        print("\n2. Binning image by 2x2...")
        try:
            work_img = bin_image_by_2(original_img)
            bin_h, bin_w = work_img.shape[:2]
            if work_img.dtype != np.uint8:
                work_img = np.clip(work_img, 0, 255).astype(np.uint8)
            print(f"   Binned size: {bin_w}x{bin_h}")
            scale_x = orig_w / bin_w
            scale_y = orig_h / bin_h
            print(f"   Scale factors: {scale_x:.2f}x{scale_y:.2f}")
            binned_path = output_dir / f"{input_path.stem}_2x2_binned.png"
            Image.fromarray(work_img).save(binned_path)
        except Exception as e:
            print(f"   Error binning: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Step 3: Preprocess (CLAHE + NL-means)
    print("\n3. Preprocessing (CLAHE + NL-means)...")
    try:
        preprocessed = clahe_then_nlmeans(work_img)
        preprocessed_path = output_dir / f"{input_path.stem}_preprocessed.png"
        Image.fromarray(preprocessed.astype(np.uint8)).save(preprocessed_path)
    except Exception as e:
        print(f"   Error preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Combined edge detection (Canny on preprocessed + H-channel in HSV)
    print("\n4. Edge detection (combined: Canny + H-channel)...")
    try:
        blur = params.get("blur_sigma", 0.1)
        cl, ch = params.get("canny_low", 0), params.get("canny_high", 50)
        edges_canny, n_white_canny, _ = compute_filtered_canny(
            preprocessed, blur_sigma=blur, canny_low=cl, canny_high=ch, min_area=0
        )
        edges = edges_canny.copy()
        if params.get("use_h_channel", True):
            edges_h = compute_h_channel_edges(work_img, blur_sigma=blur, canny_low=cl, canny_high=ch)
            edges = np.maximum(edges_canny, edges_h)
            n_h = int(np.sum(edges_h > 127))
        else:
            n_h = 0
        n_white = int(np.sum(edges > 127))
        edges_path = output_dir / f"{input_path.stem}_edges.png"
        Image.fromarray(edges).save(edges_path)
        print(f"   Canny: {n_white_canny:,} px, H-channel: {n_h:,} px, Combined: {n_white:,} px")
    except Exception as e:
        print(f"   Error detecting edges: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Find contours with very aggressive gap closing (on binned image)
    print("\n5. Finding contours (very robust gap closing - connecting ALL edges)...")
    try:
        # Calculate adaptive gap sizes based on BINNED image dimensions
        image_diagonal = int(np.hypot(bin_h, bin_w))
        bridge_f = params.get("bridge_gap_factor", 0.25)
        force_f = params.get("force_close_factor", 0.4)
        adaptive_bridge_gap = max(100, int(image_diagonal * bridge_f))
        adaptive_force_close_gap = max(200, int(image_diagonal * force_f))
        
        print(f"   Image diagonal: {image_diagonal}px")
        print(f"   Bridge gap: {adaptive_bridge_gap}px")
        print(f"   Force-close gap: {adaptive_force_close_gap}px")
        
        # Step 5a: Morphological closing
        binary = _ensure_binary(edges).copy()
        close_div = params.get("close_kernel_divisor", 150)
        close_kernel_size = max(5, int(min(bin_h, bin_w) / close_div))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        line_thick = params.get("line_thickness", 5)
        if not no_gap_close:
            # Step 5b: Smart bridging
            binary = smart_connect_endpoints_bridged(
                binary,
                max_gap=adaptive_bridge_gap,
                line_thickness=line_thick,
                direction_weight=2.0,
                extend_pixels=2
            )
            
            # Step 5c: Force close
            binary = force_close_open_chains(
                binary,
                line_thickness=line_thick,
                max_iterations=200,
                max_gap=adaptive_force_close_gap
            )
            
            # Step 5d: Final aggressive pass for any remaining endpoints
            remaining_endpoints = _find_endpoints(binary)
            if remaining_endpoints:
                h, w = binary.shape
                for y1, x1 in remaining_endpoints:
                    best_d = float("inf")
                    best_p = None
                    search_radius = min(adaptive_force_close_gap * 2, max(h, w))
                    y_min = max(0, y1 - search_radius)
                    y_max = min(h, y1 + search_radius + 1)
                    x_min = max(0, x1 - search_radius)
                    x_max = min(w, x1 + search_radius + 1)
                    
                    for py in range(y_min, y_max):
                        for px in range(x_min, x_max):
                            if binary[py, px] < 127:
                                continue
                            if py == y1 and px == x1:
                                continue
                            d = np.hypot(py - y1, px - x1)
                            if d < best_d and d > 0.5:
                                best_d = d
                                best_p = (py, px)
                    
                    if best_p:
                        cv2.line(binary, (x1, y1), (best_p[1], best_p[0]), 255, line_thick)
        
        # Step 5e: Extend edges to border (morph) for border-touching contours, then extract
        morph_margin = params.get("morph_margin", 30)
        corner_radius = params.get("corner_radius", 8)
        binary = morph_extend_to_border(binary, bin_h, bin_w, margin=morph_margin, corner_radius=corner_radius)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Ensure all contours are closed
        all_contours_binned = []
        for c in contours:
            if len(c) > 0:
                first = c[0][0]
                last = c[-1][0]
                if not np.array_equal(first, last):
                    closed = np.vstack([c, c[0].reshape(1, 1, 2)])
                    all_contours_binned.append(closed)
                else:
                    all_contours_binned.append(c)
        
        print(f"   Found {len(all_contours_binned)} contours before filtering")
        
        # Filter out nested contours - keep only outermost ones (on binned image)
        all_contours_binned = filter_nested_contours(all_contours_binned)
        
        print(f"   Found {len(all_contours_binned)} outermost contours after filtering")
        
        if not use_binning:
            all_contours = all_contours_binned
        else:
            print(f"   Scaling contours to original size ({orig_w}x{orig_h})...")
            all_contours = []
            for c in all_contours_binned:
                scaled = c.copy().astype(np.float32)
                scaled[:, 0, 0] *= scale_y  # y coordinates
                scaled[:, 0, 1] *= scale_x  # x coordinates
                scaled = scaled.astype(np.int32)
                all_contours.append(scaled)
        
        print(f"   {'Scaled' if use_binning else 'Using'} {len(all_contours)} contours{' to original size' if use_binning else ''}")
        
        if all_contours:
            areas = [cv2.contourArea(c) for c in all_contours]
            print(f"   Area range: {min(areas):.1f} - {max(areas):.1f} pixels")
            
            final_endpoints = _find_endpoints(binary)
            if final_endpoints:
                print(f"   Warning: {len(final_endpoints)} endpoints still remain")
            else:
                print(f"   ✓ All edges successfully connected!")
    except Exception as e:
        print(f"   Error finding contours: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Create masks (at original size)
    print("\n6. Creating masks...")
    try:
        # Shape mask: white shapes on black background
        shape_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if all_contours:
            cv2.fillPoly(shape_mask, all_contours, 255)
        
        shape_mask_path = output_dir / f"{input_path.stem}_mask_shapes.png"
        Image.fromarray(shape_mask).save(shape_mask_path)
        print(f"   Saved: {shape_mask_path.name}")
        
        # Background mask: black shapes on white background (inverse)
        background_mask = 255 - shape_mask
        background_mask_path = output_dir / f"{input_path.stem}_mask_background.png"
        Image.fromarray(background_mask).save(background_mask_path)
        print(f"   Saved: {background_mask_path.name}")
        
        # Masked image: original image with shapes masked out (showing background only)
        masked_image = original_img.copy()
        # Set shape regions to black (or transparent)
        for c in range(3):
            masked_image[:, :, c] = np.where(shape_mask > 127, 0, original_img[:, :, c])
        
        masked_image_path = output_dir / f"{input_path.stem}_masked_background_only.png"
        Image.fromarray(masked_image).save(masked_image_path)
        print(f"   Saved: {masked_image_path.name}")
        
        # Shapes only: original image with background masked out (showing shapes only)
        shapes_only = original_img.copy()
        for c in range(3):
            shapes_only[:, :, c] = np.where(shape_mask > 127, original_img[:, :, c], 0)
        
        shapes_only_path = output_dir / f"{input_path.stem}_shapes_only.png"
        Image.fromarray(shapes_only).save(shapes_only_path)
        print(f"   Saved: {shapes_only_path.name}")
        
    except Exception as e:
        print(f"   Error creating masks: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Create overlay (at original size)
    print("\n7. Creating overlay...")
    try:
        overlay = original_img.copy()
        if all_contours:
            line_thickness = max(2, int(min(orig_w, orig_h) / 500))
            cv2.drawContours(overlay, all_contours, -1, (0, 255, 0), line_thickness)
        
        overlay_path = output_dir / f"{input_path.stem}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        print(f"   Saved: {overlay_path.name}")
    except Exception as e:
        print(f"   Error creating overlay: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✓ Complete: {input_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images with robust contour detection and create masks."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("images/x20_images"),
        help="Input directory containing images (default: images/x20_images)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("robust_contours_and_masks"),
        help="Output directory (default: robust_contours_and_masks)"
    )
    parser.add_argument(
        "--no-binning",
        action="store_true",
        help="Skip 2x2 binning; process at full resolution"
    )
    parser.add_argument(
        "--no-gap-close",
        action="store_true",
        help="Skip gap closing / force-close (faster but contours may be broken)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_files = [
        p for p in input_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image(s) to process")
    print(f"Output directory: {args.output}")
    params = load_contour_params()
    if CONTOUR_SETTINGS_PATH.exists():
        print(f"Using settings from: {CONTOUR_SETTINGS_PATH.name}")
    
    # Create output directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_path in sorted(image_files):
        # Create subdirectory for each image
        img_output_dir = output_base / img_path.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        if process_image_with_masks(
            img_path, img_output_dir,
            no_binning=args.no_binning,
            params=params,
            no_gap_close=args.no_gap_close,
        ):
            successful += 1
        else:
            failed += 1
            print(f"\n✗ Failed to process: {img_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(image_files)}")
    print(f"  Output directory: {output_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

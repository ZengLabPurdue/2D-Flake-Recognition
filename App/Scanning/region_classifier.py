"""Flood-fill and color-match regions produced by contour detection."""

import colorsys
import json
import re
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CONTRAST_MATCH_THRESHOLD = 3
DEFAULT_REGION_FLOOD_FILL_THRESHOLD = 10
CLASS_COLOR_SATURATION = 0.58
CLASS_COLOR_LIGHTNESS = 0.72


def _timed_call(stats, name, function, *args):
    if stats is None:
        return function(*args)

    started_at = perf_counter()
    try:
        return function(*args)
    finally:
        elapsed = perf_counter() - started_at
        result = stats.setdefault(
            name,
            {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0},
        )
        result["calls"] += 1
        result["total_seconds"] += elapsed
        result["max_seconds"] = max(result["max_seconds"], elapsed)


def _read_rgb_triplet(value, field_name, class_name):
    if isinstance(value, dict):
        value = (value.get("red"), value.get("green"), value.get("blue"))
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, (int, float))
            or not np.isfinite(channel)
            for channel in value
        )
    ):
        raise ValueError(f"Profile class {class_name!r} has invalid {field_name}.")
    return np.asarray(value, dtype=np.float64)


def load_profile_classes(profile_path, contrast_threshold=None):
    """Load the class names, signed RGB contrasts, and matching tolerances."""
    profile_path = Path(profile_path)
    if profile_path.is_dir():
        profile_path = profile_path / "profile.json"
    if contrast_threshold is not None and (
        isinstance(contrast_threshold, bool)
        or not isinstance(contrast_threshold, (int, float))
        or not np.isfinite(contrast_threshold)
        or not 0 <= contrast_threshold <= 255
    ):
        raise ValueError("The contrast threshold must be between 0 and 255.")

    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Profile was not found: {profile_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read profile: {profile_path}") from exc

    saved_classes = profile_data.get("classes")
    if not isinstance(saved_classes, list):
        raise ValueError(f"Profile has an invalid class list: {profile_path}")

    profile_classes = []
    for class_index, saved_class in enumerate(saved_classes):
        if not isinstance(saved_class, dict):
            raise ValueError(f"Profile class has invalid data: {profile_path}")
        name = saved_class.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Profile class has an invalid name: {profile_path}")
        name = name.strip()

        try:
            contrast_rgb = _read_rgb_triplet(
                saved_class["contrast_rgb"],
                "RGB contrast",
                name,
            )
            saved_threshold = saved_class.get("contrast_threshold")
            if saved_threshold is None:
                saved_threshold = saved_class["flood_fill"]["threshold"]
            tolerance = (
                float(saved_threshold)
                if contrast_threshold is None
                else float(contrast_threshold)
            )
        except KeyError as exc:
            raise ValueError(
                f"Profile class {name!r} is missing RGB contrast data or a threshold."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Profile class {name!r} has invalid color data.") from exc

        if np.any(contrast_rgb < -255) or np.any(contrast_rgb > 255):
            raise ValueError(f"Profile class {name!r} has an invalid RGB contrast.")
        if not np.isfinite(tolerance) or not 0 <= tolerance <= 255:
            raise ValueError(f"Profile class {name!r} has an invalid tolerance.")
        profile_classes.append({
            "name": name,
            "contrast_rgb": contrast_rgb,
            "tolerance": tolerance,
            "class_index": class_index,
        })

    if not profile_classes:
        raise ValueError(f"Profile contains no color classes: {profile_path}")
    names = [item["name"].casefold() for item in profile_classes]
    if len(names) != len(set(names)):
        raise ValueError("Profile class names must be unique.")
    return profile_classes


def match_profile_class(contrast_rgb, profile_classes):
    """Return the nearest class whose RGB difference is inside its tolerance."""
    matches = []
    for profile_class in profile_classes:
        difference = np.abs(contrast_rgb - profile_class["contrast_rgb"])
        if np.all(difference <= profile_class["tolerance"]):
            matches.append((
                float(np.linalg.norm(difference)),
                profile_class,
                difference,
            ))
    return min(matches, key=lambda match: match[0]) if matches else None


def generate_class_colors(profile_classes, color_seed=None):
    """Generate randomly assigned soft class colors with well-spaced hues."""
    if not profile_classes:
        return {}
    random = np.random.default_rng(color_seed)
    hue_offset = float(random.random())
    hue_step = 1.0 / len(profile_classes)
    hues = np.asarray([
        (hue_offset + index * hue_step) % 1.0
        for index in range(len(profile_classes))
    ])
    random.shuffle(hues)
    colors = {}
    for index, profile_class in enumerate(profile_classes):
        hue = float(hues[index])
        rgb = colorsys.hls_to_rgb(
            hue,
            CLASS_COLOR_LIGHTNESS,
            CLASS_COLOR_SATURATION,
        )
        colors[profile_class["name"]] = tuple(
            int(round(channel * 255)) for channel in rgb
        )
    return colors


def contour_mask(image_shape, contours):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def _contour_depths(hierarchy):
    depths = np.zeros(len(hierarchy), dtype=np.int32)
    for index in range(len(hierarchy)):
        parent = hierarchy[index][3]
        while parent != -1:
            depths[index] += 1
            parent = hierarchy[parent][3]
    return depths


def _deepest_contour_at_point(point, contour_indices, contours, depths):
    containing = [
        index
        for index in contour_indices
        if cv2.pointPolygonTest(contours[index], point, False) >= 0
    ]
    return max(containing, key=lambda index: depths[index]) if containing else contour_indices[0]


def _flood_fill_region(gray_image, seed_point, allowed_mask, threshold):
    """Run fixed-range, 8-connected flood fill without changing the source."""
    if not allowed_mask[seed_point[1], seed_point[0]]:
        return np.zeros(gray_image.shape, dtype=bool)
    flood_mask = np.ones(
        (gray_image.shape[0] + 2, gray_image.shape[1] + 2),
        dtype=np.uint8,
    )
    flood_mask[1:-1, 1:-1] = np.where(allowed_mask, 0, 1).astype(np.uint8)
    flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(
        gray_image,
        flood_mask,
        seed_point,
        0,
        threshold,
        threshold,
        flags,
    )
    return flood_mask[1:-1, 1:-1] == 255


def classify_contour_regions(
    image_rgb,
    edge_mask,
    contours,
    hierarchy,
    all_external_contours,
    external_indices,
    contour_indices_by_external,
    profile_classes,
    flood_fill_threshold,
    benchmark_stats=None,
):
    """Flood every contour seed independently and retain overlapping classes."""
    height, width = edge_mask.shape
    external_contours = [contours[index] for index in external_indices]
    external_mask = contour_mask(image_rgb.shape, external_contours)
    all_external_mask = contour_mask(image_rgb.shape, all_external_contours)
    background_pixels = image_rgb[all_external_mask == 0]
    if len(background_pixels) == 0:
        raise ValueError("The detected contours cover the entire image background.")
    background_color = np.rint(np.mean(background_pixels, axis=0))

    all_region_mask = np.zeros((height, width), dtype=np.uint8)
    region_overlap_count_map = np.zeros((height, width), dtype=np.uint16)
    all_internal_mask = np.zeros((height, width), dtype=np.uint8)
    matched_region_mask = np.zeros((height, width), dtype=np.uint8)
    matched_internal_mask = np.zeros((height, width), dtype=np.uint8)
    class_masks = {}
    region_results = []
    depths = _timed_call(
        benchmark_stats,
        "contour_depths",
        _contour_depths,
        hierarchy,
    )
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    for external_index in external_indices:
        external_contour = contours[external_index]
        x, y, contour_width, contour_height = cv2.boundingRect(external_contour)
        x1, y1 = max(0, x - 1), max(0, y - 1)
        x2 = min(width, x + contour_width + 1)
        y2 = min(height, y + contour_height + 1)

        local_contour = external_contour.copy()
        local_contour[:, 0, 0] -= x1
        local_contour[:, 0, 1] -= y1
        local_external_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.drawContours(local_external_mask, [local_contour], -1, 255, cv2.FILLED)
        allowed = (
            (local_external_mask > 0)
            & (edge_mask[y1:y2, x1:x2] == 0)
        ).astype(np.uint8)
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            allowed,
            connectivity=8,
        )

        components = []
        contour_indices = contour_indices_by_external[external_index]
        for component_label in range(1, component_count):
            component_mask = labels == component_label
            component_size = int(stats[component_label, cv2.CC_STAT_AREA])
            if component_size == 0:
                continue
            seed_x, seed_y = (int(round(value)) for value in centroids[component_label])
            if (
                not 0 <= seed_x < component_mask.shape[1]
                or not 0 <= seed_y < component_mask.shape[0]
                or not component_mask[seed_y, seed_x]
            ):
                seed_y, seed_x = np.argwhere(component_mask)[0]
                seed_x, seed_y = int(seed_x), int(seed_y)
            global_seed = (seed_x + x1, seed_y + y1)
            source_index = _timed_call(
                benchmark_stats,
                "deepest_contour_at_point",
                _deepest_contour_at_point,
                global_seed,
                contour_indices,
                contours,
                depths,
            )
            components.append({
                "pixel_count": component_size,
                "global_seed": global_seed,
                "source_contour_index": source_index,
                "raw_depth": int(depths[source_index]),
            })

        if not components:
            continue
        outer_depth = min(component["raw_depth"] for component in components)
        components.sort(
            key=lambda component: (
                component["raw_depth"],
                component["pixel_count"],
            ),
            reverse=True,
        )

        for component in components:
            global_seed = component["global_seed"]
            local_seed = (global_seed[0] - x1, global_seed[1] - y1)
            region_mask = _timed_call(
                benchmark_stats,
                "flood_fill_region",
                _flood_fill_region,
                np.ascontiguousarray(gray_image[y1:y2, x1:x2]),
                local_seed,
                local_external_mask > 0,
                flood_fill_threshold,
            )
            pixel_count = int(np.count_nonzero(region_mask))
            if pixel_count == 0:
                continue

            region_y, region_x = np.nonzero(region_mask)
            bounding_box = {
                "x": int(x1 + region_x.min()),
                "y": int(y1 + region_y.min()),
                "width": int(region_x.max() - region_x.min() + 1),
                "height": int(region_y.max() - region_y.min() + 1),
            }

            depth_difference = component["raw_depth"] - outer_depth
            nesting_depth = max(0, (depth_difference + 1) // 2)
            region_type = "external" if nesting_depth == 0 else "internal"
            region_pixels = image_rgb[y1:y2, x1:x2][region_mask]
            average_color = np.mean(region_pixels, axis=0)
            contrast = np.rint(average_color - background_color)
            match = (
                _timed_call(
                    benchmark_stats,
                    "match_profile_class",
                    match_profile_class,
                    contrast,
                    profile_classes,
                )
                if profile_classes
                else None
            )

            all_region_mask[y1:y2, x1:x2][region_mask] = 255
            region_overlap_count_map[y1:y2, x1:x2][region_mask] += 1
            if region_type == "internal":
                all_internal_mask[y1:y2, x1:x2][region_mask] = 255

            matched_class = match[1] if match else None
            if matched_class:
                matched_region_mask[y1:y2, x1:x2][region_mask] = 255
                if region_type == "internal":
                    matched_internal_mask[y1:y2, x1:x2][region_mask] = 255
                class_mask = class_masks.get(matched_class["name"])
                if class_mask is None:
                    class_mask = np.zeros((height, width), dtype=np.uint8)
                    class_masks[matched_class["name"]] = class_mask
                class_mask[y1:y2, x1:x2][region_mask] = 255

            region_results.append({
                "seed_point": global_seed,
                "region_type": region_type,
                "source_contour_index": int(component["source_contour_index"]),
                "nesting_depth": nesting_depth,
                "seed_component_pixel_count": component["pixel_count"],
                "average_color_rgb": tuple(
                    int(round(channel)) for channel in average_color
                ),
                "contrast_rgb": tuple(int(round(channel)) for channel in contrast),
                "matched_class": matched_class["name"] if matched_class else None,
                "matched_color_rgb": (
                    matched_class["display_color_rgb"] if matched_class else None
                ),
                "match_threshold": matched_class["tolerance"] if matched_class else None,
                "match_distance": match[0] if match else None,
                "channel_difference_rgb": (
                    tuple(float(value) for value in match[2]) if match else None
                ),
                "pixel_count": pixel_count,
                "bounding_box": bounding_box,
            })

    class_index_map = np.zeros((height, width), dtype=np.int32)
    class_overlap_count_map = np.zeros((height, width), dtype=np.uint16)
    for profile_class in profile_classes:
        class_mask = class_masks.get(profile_class["name"])
        if class_mask is None:
            continue
        selected = class_mask > 0
        class_overlap_count_map[selected] += 1
        class_index_map[selected] = profile_class["class_index"] + 1

    return {
        "external_mask": external_mask,
        "all_external_mask": all_external_mask,
        "all_region_mask": all_region_mask,
        "region_overlap_count_map": region_overlap_count_map,
        "all_internal_mask": all_internal_mask,
        "matched_region_mask": matched_region_mask,
        "matched_internal_mask": matched_internal_mask,
        "class_index_map": class_index_map,
        "class_masks": class_masks,
        "class_overlap_count_map": class_overlap_count_map,
        "background_color_rgb": tuple(int(channel) for channel in background_color),
        "region_results": region_results,
    }


def create_classified_image(image_shape, external_mask, class_index_map, profile_classes):
    """Create a black-background RGB class map with white unmatched flakes."""
    image = np.zeros(image_shape, dtype=np.uint8)
    image[external_mask > 0] = (255, 255, 255)
    for profile_class in profile_classes:
        image[class_index_map == profile_class["class_index"] + 1] = profile_class[
            "display_color_rgb"
        ]
    return image


def _load_class_legend_font(font_size):
    """Load a crisp Git Bash-style monospace font with portable fallbacks."""
    font_candidates = (
        Path("C:/Windows/Fonts/lucon.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    )
    for font_path in font_candidates:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), font_size)
    for font_name in ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default(size=font_size)


def draw_class_legend(image_rgb, profile_classes, region_results=None):
    """Draw a compact monospace legend linked to each matching region."""
    if not profile_classes:
        return image_rgb.copy()

    result = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(result)
    scale = max(1.0, min(image_rgb.shape[:2]) / 700.0)
    font_size = max(14, int(round(18 * scale)))
    font = _load_class_legend_font(font_size)
    line_height = max(16, int(round(font_size * 1.15)))
    line_width = max(1, int(round(scale / 2)))
    x = max(10, int(round(14 * scale)))
    y = max(8, int(round(12 * scale)))

    regions_by_class = {}
    for region in region_results or ():
        class_name = region.get("matched_class")
        bounding_box = region.get("bounding_box")
        if class_name is None or not bounding_box:
            continue
        center = (
            int(round(bounding_box["x"] + bounding_box["width"] / 2)),
            int(round(bounding_box["y"] + bounding_box["height"] / 2)),
        )
        regions_by_class.setdefault(class_name, []).append(center)

    label_layout = []
    for index, profile_class in enumerate(profile_classes):
        text_y = y + index * line_height
        if text_y + font_size >= image_rgb.shape[0] - 5:
            break
        class_name = profile_class["name"]
        color = tuple(int(channel) for channel in profile_class["display_color_rgb"])
        display_name = re.sub(r"\bclass\b", "Class", class_name, flags=re.IGNORECASE)
        text_bounds = draw.textbbox((x, text_y), display_name, font=font)
        line_start = (
            text_bounds[2] + max(4, int(round(5 * scale))),
            (text_bounds[1] + text_bounds[3]) // 2,
        )
        label_layout.append((class_name, display_name, color, text_y, line_start))

    for class_name, _, color, _, line_start in label_layout:
        for region_center in regions_by_class.get(class_name, ()):
            draw.line(
                (line_start, region_center),
                fill=color,
                width=line_width,
            )

    for _, display_name, color, text_y, _ in label_layout:
        draw.text((x, text_y), display_name, font=font, fill=color)

    return np.asarray(result).copy()


def _seed_inside_contour(contour, allowed_mask):
    candidates = (contour_mask(allowed_mask.shape, [contour]) > 0) & allowed_mask
    ys, xs = np.nonzero(candidates)
    if len(xs) == 0:
        return None
    moments = cv2.moments(contour)
    if moments["m00"]:
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        if (
            0 <= center[0] < allowed_mask.shape[1]
            and 0 <= center[1] < allowed_mask.shape[0]
            and candidates[center[1], center[0]]
        ):
            return center
    x, y, width, height = cv2.boundingRect(contour)
    center_x = x + width / 2
    center_y = y + height / 2
    closest = np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2)
    return int(xs[closest]), int(ys[closest])


def internal_region_mask(edge_mask, contour_groups):
    """Build the legacy geometric mask used by the production scanner."""
    height, width = edge_mask.shape
    result = np.zeros((height, width), dtype=np.uint8)
    processed = np.zeros((height, width), dtype=np.uint8)
    for external_contour, internal_contours in contour_groups:
        x, y, contour_width, contour_height = cv2.boundingRect(external_contour)
        x1, y1 = max(0, x - 1), max(0, y - 1)
        x2 = min(width, x + contour_width + 1)
        y2 = min(height, y + contour_height + 1)
        local_external = external_contour.copy()
        local_external[:, 0] -= (x1, y1)
        allowed = (
            (contour_mask((y2 - y1, x2 - x1), [local_external]) > 0)
            & (edge_mask[y1:y2, x1:x2] == 0)
            & (processed[y1:y2, x1:x2] == 0)
        )
        for internal_contour in internal_contours:
            local_contour = internal_contour.copy()
            local_contour[:, 0] -= (x1, y1)
            seed = _seed_inside_contour(local_contour, allowed)
            if seed is None:
                continue
            flood_image = np.zeros(allowed.shape, dtype=np.uint8)
            flood_mask = np.ones(
                (allowed.shape[0] + 2, allowed.shape[1] + 2),
                dtype=np.uint8,
            )
            flood_mask[1:-1, 1:-1] = np.where(allowed, 0, 1).astype(np.uint8)
            cv2.floodFill(flood_image, flood_mask, seed, 255, flags=4)
            region = flood_image == 255
            result[y1:y2, x1:x2][region] = 255
            processed[y1:y2, x1:x2][region] = 255
            allowed[region] = False
    return result


def background_color_rgb(image_bgr, flake_mask):
    """Return mean RGB background outside the supplied flake mask."""
    pixels = image_bgr[flake_mask == 0]
    if len(pixels) == 0:
        raise ValueError("The contour mask covers the entire image background.")
    blue, green, red = np.mean(pixels, axis=0)
    return int(round(red)), int(round(green)), int(round(blue))


def region_contrast_rgb(image_bgr, region_mask, flake_mask):
    """Return signed RGB region contrast relative to the supplied background."""
    if region_mask is None or region_mask.shape != image_bgr.shape[:2]:
        raise ValueError("The region mask does not match the image.")
    if not np.any(region_mask > 0):
        raise ValueError("The selected region is empty.")
    blue, green, red = np.mean(image_bgr[region_mask > 0], axis=0)
    background = background_color_rgb(image_bgr, flake_mask)
    return tuple(
        int(round(region_channel - background_channel))
        for region_channel, background_channel in zip((red, green, blue), background)
    )

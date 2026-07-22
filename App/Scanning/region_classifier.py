"""Flood-fill and color-match regions produced by contour detection."""

import colorsys
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CONTRAST_MATCH_THRESHOLD = 3
DEFAULT_REGION_FLOOD_FILL_THRESHOLD = 10
CLASS_COLOR_SATURATION = 0.58
CLASS_COLOR_LIGHTNESS = 0.72
FILTER_BAD_COLOR = "bad_color"
FILTER_INTENSITY_RANGE = "intensity_range"
FILTER_COLOR_DISTANCE = "color_distance"
FILTER_TYPES = {
    FILTER_BAD_COLOR,
    FILTER_INTENSITY_RANGE,
    FILTER_COLOR_DISTANCE,
}
LEGEND_TOP_LEFT = "top_left"
LEGEND_TOP_RIGHT = "top_right"
LEGEND_BOTTOM_LEFT = "bottom_left"
LEGEND_BOTTOM_RIGHT = "bottom_right"
LEGEND_POSITIONS = {
    LEGEND_TOP_LEFT,
    LEGEND_TOP_RIGHT,
    LEGEND_BOTTOM_LEFT,
    LEGEND_BOTTOM_RIGHT,
}
DEFAULT_LEGEND_POSITION = LEGEND_TOP_LEFT


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
        not isinstance(value, (list, tuple, np.ndarray))
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


def _read_size_requirement(value, label):
    if value is None:
        return None, None
    if not isinstance(value, dict):
        raise ValueError(f"{label} size requirement must be an object or null.")
    result = []
    for key in ("minimum_size_um", "maximum_size_um"):
        size = value.get(key)
        if size is not None and (
            isinstance(size, bool)
            or not isinstance(size, (int, float))
            or not np.isfinite(size)
            or size <= 0
        ):
            raise ValueError(f"{label} has an invalid {key} value.")
        result.append(None if size is None else float(size))
    minimum, maximum = result
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{label} minimum size cannot exceed its maximum size.")
    return minimum, maximum


def load_profile_configuration(profile_path, contrast_threshold=None):
    """Load matching classes and profile-wide region requirements."""
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

    if not isinstance(profile_data, dict):
        raise ValueError(f"Profile has invalid data: {profile_path}")
    saved_classes = profile_data.get("classes")
    saved_filters = profile_data.get("filters", [])
    if not isinstance(saved_classes, list):
        raise ValueError(f"Profile has an invalid class list: {profile_path}")
    if not isinstance(saved_filters, list):
        raise ValueError(f"Profile has an invalid filter list: {profile_path}")
    profile_version = profile_data.get("version", 1)

    profile_minimum, profile_maximum = _read_size_requirement(
        profile_data.get("size_requirement"),
        "Profile",
    )
    profile_classes = []
    for saved_class in saved_classes:
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
        identify = saved_class.get("identify", True)
        reject = profile_version == 3 and saved_class.get("reject", False)
        if not isinstance(identify, bool) or not isinstance(reject, bool):
            raise ValueError(f"Profile class {name!r} has invalid matching flags.")
        group = saved_class.get("group")
        if group is None:
            group = ""
        if not isinstance(group, str):
            raise ValueError(f"Profile class {name!r} has an invalid group.")
        minimum, maximum = _read_size_requirement(
            saved_class.get("size_requirement"),
            f"Profile class {name!r}",
        )
        if not reject or identify:
            profile_classes.append({
                "name": name,
                "contrast_rgb": contrast_rgb,
                "tolerance": tolerance,
                # Keep display-map indexes contiguous when a version-three
                # reject-only class is migrated into a filter.
                "class_index": len(profile_classes),
                "group": group.strip(),
                "identify": identify,
                "minimum_size_um": minimum,
                "maximum_size_um": maximum,
            })
        if reject:
            saved_filters.append({
                "name": f"Filter {sum(item.get('_legacy_filter', False) for item in saved_filters) + 1}",
                "type": FILTER_BAD_COLOR,
                "contrast_rgb": contrast_rgb,
                "tolerance": tolerance,
                "_legacy_filter": True,
            })

    names = [item["name"].casefold() for item in profile_classes]
    if len(names) != len(set(names)):
        raise ValueError("Profile class names must be unique.")
    profile_filters = []
    for filter_index, saved_filter in enumerate(saved_filters, start=1):
        if not isinstance(saved_filter, dict):
            raise ValueError(f"Profile filter {filter_index} has invalid data.")
        name = saved_filter.get("name")
        filter_type = saved_filter.get("type")
        if not isinstance(name, str) or not name.strip() or filter_type not in FILTER_TYPES:
            raise ValueError(f"Profile filter {filter_index} is invalid.")
        item = {"name": name.strip(), "type": filter_type}
        if filter_type == FILTER_INTENSITY_RANGE:
            intensity_range = saved_filter.get("intensity_range")
            if not isinstance(intensity_range, dict):
                raise ValueError(f"Profile filter {name!r} has an invalid intensity range.")
            minimum = intensity_range.get("minimum")
            maximum = intensity_range.get("maximum")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 <= value <= 255
                for value in (minimum, maximum)
            ) or minimum > maximum:
                raise ValueError(f"Profile filter {name!r} has an invalid intensity range.")
            item["minimum_intensity"] = float(minimum)
            item["maximum_intensity"] = float(maximum)
        else:
            try:
                item["contrast_rgb"] = _read_rgb_triplet(
                    saved_filter["contrast_rgb"],
                    "RGB contrast",
                    name,
                )
            except KeyError as exc:
                raise ValueError(
                    f"Profile filter {name!r} is missing RGB contrast data."
                ) from exc
            if np.any(item["contrast_rgb"] < -255) or np.any(
                item["contrast_rgb"] > 255
            ):
                raise ValueError(
                    f"Profile filter {name!r} has an invalid RGB contrast."
                )
            if filter_type == FILTER_BAD_COLOR:
                tolerance = saved_filter.get("tolerance")
                if tolerance is None:
                    try:
                        tolerance = saved_filter["flood_fill"]["threshold"]
                    except (KeyError, TypeError) as exc:
                        raise ValueError(
                            f"Profile filter {name!r} has no color tolerance."
                        ) from exc
                if (
                    isinstance(tolerance, bool)
                    or not isinstance(tolerance, (int, float))
                    or not np.isfinite(tolerance)
                    or not 0 <= tolerance <= 255
                ):
                    raise ValueError(f"Profile filter {name!r} has an invalid tolerance.")
                item["tolerance"] = float(tolerance)
            else:
                distance = saved_filter.get("distance_threshold")
                if (
                    isinstance(distance, bool)
                    or not isinstance(distance, (int, float))
                    or not np.isfinite(distance)
                    or distance < 0
                ):
                    raise ValueError(
                        f"Profile filter {name!r} has an invalid color distance."
                    )
                item["distance_threshold"] = float(distance)
        profile_filters.append(item)

    filter_names = [item["name"].casefold() for item in profile_filters]
    if len(filter_names) != len(set(filter_names)):
        raise ValueError("Profile filter names must be unique.")

    return {
        "classes": profile_classes,
        "filters": profile_filters,
        "minimum_size_um": profile_minimum,
        "maximum_size_um": profile_maximum,
    }


def load_profile_classes(profile_path, contrast_threshold=None):
    """Load class matching data while retaining the former public API."""
    return load_profile_configuration(profile_path, contrast_threshold)["classes"]


def _size_matches(size_um, minimum, maximum):
    if size_um is None:
        return True
    if minimum is not None and size_um < minimum:
        return False
    if maximum is not None and size_um > maximum:
        return False
    return True


def match_profile_class(contrast_rgb, profile_classes, size_um=None):
    """Return the nearest enabled class inside its color and size limits."""
    matches = []
    for profile_class in profile_classes:
        if not profile_class.get("identify", True):
            continue
        if not _size_matches(
            size_um,
            profile_class.get("minimum_size_um"),
            profile_class.get("maximum_size_um"),
        ):
            continue
        difference = np.abs(contrast_rgb - profile_class["contrast_rgb"])
        if np.all(difference <= profile_class["tolerance"]):
            matches.append((
                float(np.linalg.norm(difference)),
                profile_class,
                difference,
            ))
    return min(matches, key=lambda match: match[0]) if matches else None


def match_profile_filter(contrast_rgb, average_intensity, profile_filters):
    """Return the first automatic rejection filter matched by a region."""
    for profile_filter in profile_filters:
        filter_type = profile_filter["type"]
        if filter_type == FILTER_INTENSITY_RANGE:
            if (
                profile_filter["minimum_intensity"]
                <= average_intensity
                <= profile_filter["maximum_intensity"]
            ):
                return average_intensity, profile_filter, None
            continue

        difference = np.abs(contrast_rgb - profile_filter["contrast_rgb"])
        distance = float(np.linalg.norm(difference))
        if filter_type == FILTER_BAD_COLOR:
            if np.all(difference <= profile_filter["tolerance"]):
                return distance, profile_filter, difference
        elif distance > profile_filter["distance_threshold"]:
            return distance, profile_filter, difference
    return None


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


def _validate_classification_size_inputs(
    profile_classes,
    pixel_size_um,
    minimum_size_um,
    maximum_size_um,
):
    if pixel_size_um is not None and (
        isinstance(pixel_size_um, bool)
        or not isinstance(pixel_size_um, (int, float))
        or not np.isfinite(pixel_size_um)
        or pixel_size_um <= 0
    ):
        raise ValueError("Pixel size must be a positive number of micrometers.")
    class_size_required = any(
        item.get("minimum_size_um") is not None
        or item.get("maximum_size_um") is not None
        for item in profile_classes
    )
    if (
        minimum_size_um is not None
        or maximum_size_um is not None
        or class_size_required
    ) and pixel_size_um is None:
        raise ValueError("A calibrated pixel size is required by the profile size limits.")


def _prepare_classification_workspace(
    image_rgb,
    edge_mask,
    contours,
    all_external_contours,
    external_indices,
):
    height, width = edge_mask.shape
    external_contours = [contours[index] for index in external_indices]
    external_mask = contour_mask(image_rgb.shape, external_contours)
    all_external_mask = contour_mask(image_rgb.shape, all_external_contours)
    background_pixels = image_rgb[all_external_mask == 0]
    if len(background_pixels) == 0:
        raise ValueError("The detected contours cover the entire image background.")
    background_color = np.rint(np.mean(background_pixels, axis=0))
    rejected_region_mask = np.zeros((height, width), dtype=np.uint8)
    return {
        "height": height,
        "width": width,
        "external_mask": external_mask,
        "all_external_mask": all_external_mask,
        "background_color": background_color,
        "all_region_mask": np.zeros((height, width), dtype=np.uint8),
        "region_overlap_count_map": np.zeros((height, width), dtype=np.uint16),
        "all_internal_mask": np.zeros((height, width), dtype=np.uint8),
        "matched_region_mask": np.zeros((height, width), dtype=np.uint8),
        "matched_internal_mask": np.zeros((height, width), dtype=np.uint8),
        "rejected_region_mask": rejected_region_mask,
        "filtered_region_mask": rejected_region_mask,
        "class_masks": {},
        "region_results": [],
    }


def _connected_components(allowed):
    return cv2.connectedComponentsWithStats(allowed, connectivity=8)


def _describe_seed_components(
    component_count,
    labels,
    component_stats,
    centroids,
    x1,
    y1,
    contour_indices,
    contours,
    depths,
    benchmark_stats,
):
    components = []
    for component_label in range(1, component_count):
        component_size = int(component_stats[component_label, cv2.CC_STAT_AREA])
        if component_size == 0:
            continue
        seed_x, seed_y = (
            int(round(value)) for value in centroids[component_label]
        )
        if (
            not 0 <= seed_x < labels.shape[1]
            or not 0 <= seed_y < labels.shape[0]
            or labels[seed_y, seed_x] != component_label
        ):
            left = int(component_stats[component_label, cv2.CC_STAT_LEFT])
            top = int(component_stats[component_label, cv2.CC_STAT_TOP])
            component_width = int(
                component_stats[component_label, cv2.CC_STAT_WIDTH]
            )
            component_height = int(
                component_stats[component_label, cv2.CC_STAT_HEIGHT]
            )
            relative_y, relative_x = np.argwhere(
                labels[
                    top:top + component_height,
                    left:left + component_width,
                ] == component_label
            )[0]
            seed_x = left + int(relative_x)
            seed_y = top + int(relative_y)
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
            "component_label": component_label,
            "pixel_count": component_size,
            "global_seed": global_seed,
            "source_contour_index": source_index,
            "raw_depth": int(depths[source_index]),
            "bounding_box": {
                "x": int(x1 + component_stats[component_label, cv2.CC_STAT_LEFT]),
                "y": int(y1 + component_stats[component_label, cv2.CC_STAT_TOP]),
                "width": int(component_stats[component_label, cv2.CC_STAT_WIDTH]),
                "height": int(component_stats[component_label, cv2.CC_STAT_HEIGHT]),
            },
        })
    return components


def _extract_external_components(
    external_index,
    contours,
    edge_mask,
    contour_indices_by_external,
    depths,
    width,
    height,
    benchmark_stats,
):
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
    component_count, labels, component_stats, centroids = _timed_call(
        benchmark_stats,
        "connected_components",
        _connected_components,
        allowed,
    )
    components = _timed_call(
        benchmark_stats,
        "describe_seed_components",
        _describe_seed_components,
        component_count,
        labels,
        component_stats,
        centroids,
        x1,
        y1,
        contour_indices_by_external[external_index],
        contours,
        depths,
        benchmark_stats,
    )
    if not components:
        return None
    outer_depth = min(component["raw_depth"] for component in components)
    components.sort(
        key=lambda component: (
            component["raw_depth"],
            component["pixel_count"],
        ),
        reverse=True,
    )
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "local_external_mask": local_external_mask,
        "component_labels": labels,
        "components": components,
        "outer_depth": outer_depth,
    }


def _prepare_flood_inputs(gray_image, external_region, local_seed):
    return (
        np.ascontiguousarray(gray_image[
            external_region["y1"]:external_region["y2"],
            external_region["x1"]:external_region["x2"],
        ]),
        external_region["local_external_mask"] > 0,
        local_seed,
    )


def _contour_component_mask(component_labels, component_label):
    return component_labels == component_label


def _measure_contour_color(
    image_rgb,
    component_mask,
    x1,
    y1,
    x2,
    y2,
    background_color,
):
    mask = component_mask.astype(np.uint8) * 255
    average_color = np.asarray(
        cv2.mean(image_rgb[y1:y2, x1:x2], mask=mask)[:3],
        dtype=np.float64,
    )
    return {
        "average_color": average_color,
        "contrast": np.rint(average_color - background_color),
        "average_intensity": float(np.mean(average_color)),
    }


def _measure_contour_geometry(
    component,
    pixel_size_um,
    minimum_size_um,
    maximum_size_um,
):
    bounding_box = component["bounding_box"]
    size_um = (
        max(bounding_box["width"], bounding_box["height"]) * pixel_size_um
        if pixel_size_um is not None
        else None
    )
    return {
        "pixel_count": component["pixel_count"],
        "bounding_box": bounding_box,
        "size_um": size_um,
        "inside_profile_size": _size_matches(
            size_um,
            minimum_size_um,
            maximum_size_um,
        ),
    }


def _measure_region_geometry(
    region_mask,
    x1,
    y1,
    pixel_size_um,
    minimum_size_um,
    maximum_size_um,
):
    pixel_count = int(np.count_nonzero(region_mask))
    if pixel_count == 0:
        return None
    region_y, region_x = np.nonzero(region_mask)
    bounding_box = {
        "x": int(x1 + region_x.min()),
        "y": int(y1 + region_y.min()),
        "width": int(region_x.max() - region_x.min() + 1),
        "height": int(region_y.max() - region_y.min() + 1),
    }
    size_um = (
        max(bounding_box["width"], bounding_box["height"]) * pixel_size_um
        if pixel_size_um is not None
        else None
    )
    return {
        "pixel_count": pixel_count,
        "bounding_box": bounding_box,
        "size_um": size_um,
        "inside_profile_size": _size_matches(
            size_um,
            minimum_size_um,
            maximum_size_um,
        ),
    }


def _match_region(
    color_measurement,
    geometry,
    profile_classes,
    profile_filters,
    benchmark_stats,
):
    filter_match = (
        _timed_call(
            benchmark_stats,
            "match_profile_filter",
            match_profile_filter,
            color_measurement["contrast"],
            color_measurement["average_intensity"],
            profile_filters or (),
        )
        if geometry["inside_profile_size"] and profile_filters
        else None
    )
    match = (
        _timed_call(
            benchmark_stats,
            "match_profile_class",
            match_profile_class,
            color_measurement["contrast"],
            profile_classes,
            geometry["size_um"],
        )
        if (
            profile_classes
            and geometry["inside_profile_size"]
            and filter_match is None
        )
        else None
    )
    return filter_match, match


def _update_region_masks(
    workspace,
    region_mask,
    x1,
    y1,
    x2,
    y2,
    region_type,
    matched_filter,
    matched_class,
):
    workspace["all_region_mask"][y1:y2, x1:x2][region_mask] = 255
    workspace["region_overlap_count_map"][y1:y2, x1:x2][region_mask] += 1
    if region_type == "internal":
        workspace["all_internal_mask"][y1:y2, x1:x2][region_mask] = 255
    if matched_filter:
        workspace["filtered_region_mask"][y1:y2, x1:x2][region_mask] = 255
    if not matched_class:
        return
    workspace["matched_region_mask"][y1:y2, x1:x2][region_mask] = 255
    if region_type == "internal":
        workspace["matched_internal_mask"][y1:y2, x1:x2][region_mask] = 255
    class_mask = workspace["class_masks"].get(matched_class["name"])
    if class_mask is None:
        class_mask = np.zeros(
            (workspace["height"], workspace["width"]),
            dtype=np.uint8,
        )
        workspace["class_masks"][matched_class["name"]] = class_mask
    class_mask[y1:y2, x1:x2][region_mask] = 255


def _make_region_result(
    component,
    region_type,
    nesting_depth,
    color_measurement,
    geometry,
    matched_filter,
    matched_class,
    filter_match,
    match,
    flood_filled,
):
    average_color = color_measurement["average_color"]
    contrast = color_measurement["contrast"]
    size_um = geometry["size_um"]
    return {
        "seed_point": component["global_seed"],
        "region_type": region_type,
        "source_contour_index": int(component["source_contour_index"]),
        "nesting_depth": nesting_depth,
        "seed_component_pixel_count": component["pixel_count"],
        "classification_source": "contour_interior",
        "flood_filled": flood_filled,
        "average_color_rgb": tuple(
            int(round(channel)) for channel in average_color
        ),
        "average_intensity": color_measurement["average_intensity"],
        "contrast_rgb": tuple(int(round(channel)) for channel in contrast),
        "matched_class": matched_class["name"] if matched_class else None,
        "matched_group": (
            (matched_class.get("group") or None) if matched_class else None
        ),
        "filtered": matched_filter is not None,
        "filtered_by": matched_filter["name"] if matched_filter else None,
        "filter_type": matched_filter["type"] if matched_filter else None,
        "rejected": matched_filter is not None,
        "rejected_by_class": None,
        "rejected_by_group": None,
        "size_um": float(size_um) if size_um is not None else None,
        "inside_profile_size": geometry["inside_profile_size"],
        "matched_color_rgb": (
            matched_class["display_color_rgb"] if matched_class else None
        ),
        "match_threshold": matched_class["tolerance"] if matched_class else None,
        "match_distance": (
            match[0] if match else (filter_match[0] if filter_match else None)
        ),
        "channel_difference_rgb": (
            tuple(float(value) for value in match[2])
            if match and match[2] is not None
            else (
                tuple(float(value) for value in filter_match[2])
                if filter_match and filter_match[2] is not None
                else None
            )
        ),
        "pixel_count": geometry["pixel_count"],
        "bounding_box": geometry["bounding_box"],
    }


def _build_class_index_maps(height, width, profile_classes, class_masks):
    class_index_map = np.zeros((height, width), dtype=np.int32)
    class_overlap_count_map = np.zeros((height, width), dtype=np.uint16)
    for profile_class in profile_classes:
        class_mask = class_masks.get(profile_class["name"])
        if class_mask is None:
            continue
        selected = class_mask > 0
        class_overlap_count_map[selected] += 1
        class_index_map[selected] = profile_class["class_index"] + 1
    return class_index_map, class_overlap_count_map


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
    pixel_size_um=None,
    minimum_size_um=None,
    maximum_size_um=None,
    profile_filters=None,
):
    """Classify contour interiors first, then flood-fill identified regions."""
    _timed_call(
        benchmark_stats,
        "validate_classification_inputs",
        _validate_classification_size_inputs,
        profile_classes,
        pixel_size_um,
        minimum_size_um,
        maximum_size_um,
    )
    workspace = _timed_call(
        benchmark_stats,
        "prepare_classification_workspace",
        _prepare_classification_workspace,
        image_rgb,
        edge_mask,
        contours,
        all_external_contours,
        external_indices,
    )
    depths = _timed_call(
        benchmark_stats,
        "contour_depths",
        _contour_depths,
        hierarchy,
    )
    gray_image = _timed_call(
        benchmark_stats,
        "convert_to_grayscale",
        cv2.cvtColor,
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    for external_index in external_indices:
        external_region = _timed_call(
            benchmark_stats,
            "extract_external_components",
            _extract_external_components,
            external_index,
            contours,
            edge_mask,
            contour_indices_by_external,
            depths,
            workspace["width"],
            workspace["height"],
            benchmark_stats,
        )
        if external_region is None:
            continue
        x1 = external_region["x1"]
        y1 = external_region["y1"]
        x2 = external_region["x2"]
        y2 = external_region["y2"]
        for component in external_region["components"]:
            global_seed = component["global_seed"]
            local_seed = (global_seed[0] - x1, global_seed[1] - y1)
            component_mask = _timed_call(
                benchmark_stats,
                "build_contour_interior_mask",
                _contour_component_mask,
                external_region["component_labels"],
                component["component_label"],
            )
            color_measurement = _timed_call(
                benchmark_stats,
                "measure_contour_color",
                _measure_contour_color,
                image_rgb,
                component_mask,
                x1,
                y1,
                x2,
                y2,
                workspace["background_color"],
            )
            contour_geometry = _timed_call(
                benchmark_stats,
                "measure_contour_geometry",
                _measure_contour_geometry,
                component,
                pixel_size_um,
                minimum_size_um,
                maximum_size_um,
            )
            depth_difference = (
                component["raw_depth"] - external_region["outer_depth"]
            )
            nesting_depth = max(0, (depth_difference + 1) // 2)
            region_type = "external" if nesting_depth == 0 else "internal"
            filter_match, match = _timed_call(
                benchmark_stats,
                "match_region",
                _match_region,
                color_measurement,
                contour_geometry,
                profile_classes,
                profile_filters,
                benchmark_stats,
            )
            matched_filter = filter_match[1] if filter_match else None
            matched_class = match[1] if match else None
            region_mask = None
            result_geometry = contour_geometry
            flood_filled = False
            if matched_class is not None:
                gray_region, allowed_mask, local_seed = _timed_call(
                    benchmark_stats,
                    "prepare_flood_inputs",
                    _prepare_flood_inputs,
                    gray_image,
                    external_region,
                    local_seed,
                )
                region_mask = _timed_call(
                    benchmark_stats,
                    "flood_fill_region",
                    _flood_fill_region,
                    gray_region,
                    local_seed,
                    allowed_mask,
                    flood_fill_threshold,
                )
                flooded_geometry = _timed_call(
                    benchmark_stats,
                    "measure_flooded_region_geometry",
                    _measure_region_geometry,
                    region_mask,
                    x1,
                    y1,
                    pixel_size_um,
                    minimum_size_um,
                    maximum_size_um,
                )
                if flooded_geometry is not None:
                    # Matching size is deliberately based on the full contour
                    # interior. The flooded mask only refines output geometry.
                    flooded_geometry["size_um"] = contour_geometry["size_um"]
                    flooded_geometry["inside_profile_size"] = contour_geometry[
                        "inside_profile_size"
                    ]
                    result_geometry = flooded_geometry
                    flood_filled = True
            elif matched_filter is not None:
                # Filters reject before region creation. Their contour interior
                # is retained only so preview output can show what was removed.
                region_mask = component_mask

            if region_mask is not None:
                _timed_call(
                    benchmark_stats,
                    "update_region_masks",
                    _update_region_masks,
                    workspace,
                    region_mask,
                    x1,
                    y1,
                    x2,
                    y2,
                    region_type,
                    matched_filter,
                    matched_class,
                )
            workspace["region_results"].append(_timed_call(
                benchmark_stats,
                "build_region_result",
                _make_region_result,
                component,
                region_type,
                nesting_depth,
                color_measurement,
                result_geometry,
                matched_filter,
                matched_class,
                filter_match,
                match,
                flood_filled,
            ))

    class_index_map, class_overlap_count_map = _timed_call(
        benchmark_stats,
        "build_class_index_maps",
        _build_class_index_maps,
        workspace["height"],
        workspace["width"],
        profile_classes,
        workspace["class_masks"],
    )

    return {
        "external_mask": workspace["external_mask"],
        "all_external_mask": workspace["all_external_mask"],
        "all_region_mask": workspace["all_region_mask"],
        "region_overlap_count_map": workspace["region_overlap_count_map"],
        "all_internal_mask": workspace["all_internal_mask"],
        "matched_region_mask": workspace["matched_region_mask"],
        "matched_internal_mask": workspace["matched_internal_mask"],
        "rejected_region_mask": workspace["rejected_region_mask"],
        "filtered_region_mask": workspace["filtered_region_mask"],
        "class_index_map": class_index_map,
        "class_masks": workspace["class_masks"],
        "class_overlap_count_map": class_overlap_count_map,
        "background_color_rgb": tuple(
            int(channel) for channel in workspace["background_color"]
        ),
        "region_results": workspace["region_results"],
    }


def create_classified_image(image_shape, external_mask, class_index_map, profile_classes):
    """Create a black-background RGB class map with white unmatched flakes."""
    image = np.zeros(image_shape, dtype=np.uint8)
    image[external_mask > 0] = (255, 255, 255)
    for profile_class in profile_classes:
        if not profile_class.get("identify", True):
            continue
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


def draw_class_legend(
    image_rgb,
    profile_classes,
    region_results=None,
    position=DEFAULT_LEGEND_POSITION,
):
    """Draw a compact linked legend in any corner of the classified image."""
    if not isinstance(position, str) or position not in LEGEND_POSITIONS:
        choices = ", ".join(sorted(LEGEND_POSITIONS))
        raise ValueError(f"Legend position must be one of: {choices}.")
    display_classes = [
        item
        for item in profile_classes
        if item.get("identify", True)
    ]
    if not display_classes:
        return image_rgb.copy()

    result = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(result)
    scale = max(1.0, min(image_rgb.shape[:2]) / 700.0)
    font_size = max(14, int(round(18 * scale)))
    font = _load_class_legend_font(font_size)
    line_height = max(16, int(round(font_size * 1.15)))
    line_width = max(1, int(round(scale / 2)))
    horizontal_margin = max(10, int(round(14 * scale)))
    vertical_margin = max(8, int(round(12 * scale)))
    line_gap = max(4, int(round(5 * scale)))
    on_right = position in (LEGEND_TOP_RIGHT, LEGEND_BOTTOM_RIGHT)
    on_bottom = position in (LEGEND_BOTTOM_LEFT, LEGEND_BOTTOM_RIGHT)

    maximum_lines = max(
        0,
        (image_rgb.shape[0] - 2 * vertical_margin) // line_height,
    )
    display_classes = display_classes[:maximum_lines]
    if not display_classes:
        return image_rgb.copy()
    block_height = len(display_classes) * line_height
    start_y = (
        image_rgb.shape[0] - vertical_margin - block_height
        if on_bottom
        else vertical_margin
    )

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
    for index, profile_class in enumerate(display_classes):
        text_y = start_y + index * line_height
        class_name = profile_class["name"]
        color = tuple(int(channel) for channel in profile_class["display_color_rgb"])
        display_name = (
            f"{profile_class['group']} / {class_name}"
            if profile_class.get("group")
            else class_name
        ).upper()
        unpositioned_bounds = draw.textbbox((0, 0), display_name, font=font)
        text_width = unpositioned_bounds[2] - unpositioned_bounds[0]
        text_x = (
            max(horizontal_margin, image_rgb.shape[1] - horizontal_margin - text_width)
            if on_right
            else horizontal_margin
        )
        text_bounds = draw.textbbox((text_x, text_y), display_name, font=font)
        line_start = (
            text_bounds[0] - line_gap if on_right else text_bounds[2] + line_gap,
            (text_bounds[1] + text_bounds[3]) // 2,
        )
        label_layout.append(
            (class_name, display_name, color, (text_x, text_y), line_start)
        )

    for class_name, _, color, _, line_start in label_layout:
        for region_center in regions_by_class.get(class_name, ()):
            draw.line(
                (line_start, region_center),
                fill=color,
                width=line_width,
            )

    for _, display_name, color, text_position, _ in label_layout:
        draw.text(text_position, display_name, font=font, fill=color)

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

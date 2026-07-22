"""Contour detection and output assembly for flake recognition."""

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import region_classifier
except ImportError:
    import region_classifier


def _project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "App").is_dir():
            return parent
    return Path(__file__).resolve().parent


AN_TEST_PROFILE_PATH = _project_root() / "App" / "Profiles" / "An_Test" / "profile.json"
DEFAULT_AREA_THRESHOLD = 100
DEFAULT_CONTRAST_MATCH_THRESHOLD = region_classifier.DEFAULT_CONTRAST_MATCH_THRESHOLD
DEFAULT_REGION_FLOOD_FILL_THRESHOLD = (
    region_classifier.DEFAULT_REGION_FLOOD_FILL_THRESHOLD
)
LEGEND_TOP_LEFT = region_classifier.LEGEND_TOP_LEFT
LEGEND_TOP_RIGHT = region_classifier.LEGEND_TOP_RIGHT
LEGEND_BOTTOM_LEFT = region_classifier.LEGEND_BOTTOM_LEFT
LEGEND_BOTTOM_RIGHT = region_classifier.LEGEND_BOTTOM_RIGHT
LEGEND_POSITIONS = region_classifier.LEGEND_POSITIONS
DEFAULT_LEGEND_POSITION = region_classifier.DEFAULT_LEGEND_POSITION


@dataclass(slots=True)
class ContourAnalysis:
    """Compatibility result used by the existing App scanning workflow."""

    edge_mask: np.ndarray
    all_external_contours: list[np.ndarray]
    external_contours: list[np.ndarray]
    internal_contours: list[np.ndarray]
    contour_groups: list[tuple[np.ndarray, list[np.ndarray]]]
    flake_mask: np.ndarray
    internal_mask: np.ndarray | None
    benchmark: dict | None = None


def _validate_image(image_bgr):
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("A three-channel BGR image is required.")


def _timed_call(stats, name, function, *args, **kwargs):
    if stats is None:
        return function(*args, **kwargs)
    started_at = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        elapsed = perf_counter() - started_at
        item = stats.setdefault(
            name,
            {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0},
        )
        item["calls"] += 1
        item["total_seconds"] += elapsed
        item["max_seconds"] = max(item["max_seconds"], elapsed)


def _benchmark_result(stats, wall_time_seconds):
    functions = {}
    for name, item in stats.items():
        calls = item["calls"]
        functions[name] = {
            "calls": calls,
            "total_seconds": item["total_seconds"],
            "average_seconds": item["total_seconds"] / calls,
            "max_seconds": item.get("max_seconds", item["total_seconds"] / calls),
        }
    return {
        "label": "find_flakes",
        "wall_time_seconds": wall_time_seconds,
        "functions": functions,
    }


def _print_benchmark(result):
    print("\nfind_flakes benchmark (inclusive timings)")
    print(
        f"{'Function':<30} {'Calls':>9} {'Total ms':>12} "
        f"{'Average ms':>12} {'Max ms':>12}"
    )
    print("-" * 79)
    ordered = sorted(
        result["functions"].items(),
        key=lambda item: item[1]["total_seconds"],
        reverse=True,
    )
    for name, item in ordered:
        print(
            f"{name:<30} {item['calls']:>9,d} "
            f"{item['total_seconds'] * 1000:>12.3f} "
            f"{item['average_seconds'] * 1000:>12.3f} "
            f"{item['max_seconds'] * 1000:>12.3f}"
        )
    print("-" * 79)
    print(f"{'Total wall time':<40} {result['wall_time_seconds'] * 1000:>12.3f}")


def _perimeter_point(distance, width, height):
    top_length = width - 1
    side_length = height - 1
    perimeter = 2 * (top_length + side_length)
    distance %= perimeter
    if distance <= top_length:
        return distance, 0
    distance -= top_length
    if distance <= side_length:
        return width - 1, distance
    distance -= side_length
    if distance <= top_length:
        return width - 1 - distance, height - 1
    return 0, height - 1 - (distance - top_length)


def _close_border_touching_edges(edge_mask, border_width=3):
    """Connect edge components through the image border."""
    height, width = edge_mask.shape
    if height < 2 or width < 2:
        return edge_mask
    closed = edge_mask.copy()
    _, labels = cv2.connectedComponents(edge_mask, connectivity=8)
    perimeter = 2 * ((width - 1) + (height - 1))
    border_width = max(1, min(border_width, height, width))
    distances_by_label = {}

    def add_distances(label_values, distances):
        for label, distance in zip(label_values, distances):
            if label:
                distances_by_label.setdefault(int(label), set()).add(
                    int(distance) % perimeter
                )

    ys, xs = np.nonzero(labels[:border_width, :])
    add_distances(labels[ys, xs], xs)
    ys, xs = np.nonzero(labels[:, width - border_width:])
    add_distances(labels[ys, width - border_width + xs], width - 1 + ys)
    ys, xs = np.nonzero(labels[height - border_width:, :])
    actual_ys = height - border_width + ys
    add_distances(
        labels[actual_ys, xs],
        width - 1 + height - 1 + width - 1 - xs,
    )
    ys, xs = np.nonzero(labels[:, :border_width])
    add_distances(
        labels[ys, xs],
        width - 1 + height - 1 + width - 1 + height - 1 - ys,
    )

    for distances in distances_by_label.values():
        distances = sorted(distances)
        if len(distances) < 2:
            continue
        gaps = [
            (
                (distances[(index + 1) % len(distances)] - distance) % perimeter,
                distance,
                distances[(index + 1) % len(distances)],
            )
            for index, distance in enumerate(distances)
        ]
        _, gap_start, gap_end = max(gaps, key=lambda item: item[0])
        arc_length = (gap_start - gap_end) % perimeter
        for offset in range(arc_length + 1):
            x, y = _perimeter_point(gap_end + offset, width, height)
            closed[y, x] = 255
    return closed


def _edge_mask(image_bgr, edge_threshold):
    red_green = image_bgr[:, :, (2, 1)]
    smoothed = cv2.GaussianBlur(red_green, (5, 5), 0)
    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude_squared = np.einsum("ijk,ijk->ij", gradient_x, gradient_x)
    magnitude_squared += np.einsum("ijk,ijk->ij", gradient_y, gradient_y)
    binary = np.where(
        magnitude_squared >= float(edge_threshold) ** 2,
        255,
        0,
    ).astype(np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _close_border_touching_edges(cleaned)


def _outermost_parent(index, hierarchy):
    parent = hierarchy[index][3]
    while parent != -1 and hierarchy[parent][3] != -1:
        parent = hierarchy[parent][3]
    return parent


def _detect_contours(image_bgr, edge_threshold, area_threshold, stats=None):
    edge_mask = _timed_call(stats, "edge_mask", _edge_mask, image_bgr, edge_threshold)
    contours, hierarchy = cv2.findContours(
        edge_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        hierarchy = np.empty((0, 4), dtype=np.int32)
        all_external_indices = []
        external_indices = []
    else:
        hierarchy = hierarchy[0]
        all_external_indices = [
            index for index, item in enumerate(hierarchy) if item[3] == -1
        ]
        external_indices = [
            index
            for index in all_external_indices
            if cv2.contourArea(contours[index]) >= area_threshold
        ]

    external_set = set(external_indices)
    internal_by_external = {index: [] for index in external_indices}
    indices_by_external = {index: [index] for index in external_indices}
    for index, contour in enumerate(contours):
        if len(hierarchy) == 0 or hierarchy[index][3] == -1:
            continue
        outer_parent = _outermost_parent(index, hierarchy)
        if outer_parent in external_set:
            internal_by_external[outer_parent].append(contour)
            indices_by_external[outer_parent].append(index)

    contour_groups = [
        (contours[index], internal_by_external[index])
        for index in external_indices
    ]
    return {
        "edge_mask": edge_mask,
        "contours": contours,
        "hierarchy": hierarchy,
        "all_external_indices": all_external_indices,
        "external_indices": external_indices,
        "all_external_contours": [contours[index] for index in all_external_indices],
        "external_contours": [contours[index] for index in external_indices],
        "internal_contours": [
            contour for _, internal in contour_groups for contour in internal
        ],
        "contour_groups": contour_groups,
        "contour_indices_by_external": indices_by_external,
    }


def analyze_contours(
    image_bgr,
    edge_threshold=10,
    area_threshold=500,
    fill_internal_regions=True,
):
    """Return the legacy contour analysis used by App scanning code."""
    _validate_image(image_bgr)
    if edge_threshold < 0 or area_threshold < 0:
        raise ValueError("Contour thresholds cannot be negative.")
    detection = _detect_contours(image_bgr, edge_threshold, area_threshold)
    flake_mask = region_classifier.contour_mask(
        image_bgr.shape,
        detection["all_external_contours"],
    )
    internal_mask = (
        region_classifier.internal_region_mask(
            detection["edge_mask"],
            detection["contour_groups"],
        )
        if fill_internal_regions
        else None
    )
    return ContourAnalysis(
        edge_mask=detection["edge_mask"],
        all_external_contours=detection["all_external_contours"],
        external_contours=detection["external_contours"],
        internal_contours=detection["internal_contours"],
        contour_groups=detection["contour_groups"],
        flake_mask=flake_mask,
        internal_mask=internal_mask,
    )


def mask_flakes(image_bgr, flake_mask):
    """Return a BGR copy with every detected flake set to black."""
    _validate_image(image_bgr)
    if flake_mask.shape != image_bgr.shape[:2]:
        raise ValueError("The flake mask does not match the image.")
    result = image_bgr.copy()
    result[flake_mask > 0] = 0
    return result


def _analysis_flake_mask(analysis):
    if isinstance(analysis, dict):
        return analysis.get("all_external_mask", analysis.get("flake_mask"))
    return analysis.flake_mask


def background_color_rgb(image_bgr, analysis=None):
    """Return mean RGB background while retaining the former public API."""
    _validate_image(image_bgr)
    if analysis is None:
        analysis = analyze_contours(image_bgr, fill_internal_regions=False)
    return region_classifier.background_color_rgb(
        image_bgr,
        _analysis_flake_mask(analysis),
    )


def region_contrast_rgb(image_bgr, region_mask, analysis=None):
    """Return signed RGB region contrast while retaining the former public API."""
    _validate_image(image_bgr)
    if analysis is None:
        analysis = analyze_contours(image_bgr, fill_internal_regions=False)
    return region_classifier.region_contrast_rgb(
        image_bgr,
        region_mask,
        _analysis_flake_mask(analysis),
    )


def _create_contour_edge_map(image_shape, external_contours, internal_contours):
    external_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    internal_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if external_contours:
        cv2.drawContours(external_mask, external_contours, -1, 255, 1)
    if internal_contours:
        cv2.drawContours(internal_mask, internal_contours, -1, 255, 1)
    edge_map = np.zeros(image_shape, dtype=np.uint8)
    edge_map[external_mask > 0] = (0, 255, 0)
    edge_map[internal_mask > 0] = (255, 0, 0)
    return edge_map, external_mask, internal_mask


def _show_results(image_rgb, contour_edge_map, contour_image, classified_image):
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, image, title in zip(
        axes.ravel(),
        (image_rgb, contour_edge_map, contour_image, classified_image),
        (
            "Original",
            "External (Green) / Internal (Red) Edges",
            "Detected Flakes",
            "Pixel Classification",
        ),
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def find_flakes(
    image_bgr,
    edge_threshold=10,
    area_threshold=DEFAULT_AREA_THRESHOLD,
    display=False,
    return_details=False,
    profile_path=None,
    contrast_threshold=DEFAULT_CONTRAST_MATCH_THRESHOLD,
    region_flood_fill_threshold=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    color_seed=None,
    draw_legend=True,
    benchmark=False,
    legacy_mask=False,
    pixel_size_um=None,
    profile_configuration=None,
    legend_position=DEFAULT_LEGEND_POSITION,
):
    """Detect flakes and return either class labels or the legacy masked image.

    The default result is the RGB class map used by Flake Recognition. Set
    ``legacy_mask=True`` for the production scanner's BGR image with flakes set
    to black. ``legend_position`` accepts top_left, top_right, bottom_left, or
    bottom_right. Both modes return the same filtered external contour list.
    """
    _validate_image(image_bgr)
    if edge_threshold < 0 or area_threshold < 0:
        raise ValueError("Contour thresholds cannot be negative.")
    if (
        isinstance(region_flood_fill_threshold, bool)
        or not isinstance(region_flood_fill_threshold, (int, float))
        or not 0 <= region_flood_fill_threshold <= 255
    ):
        raise ValueError("The region flood-fill threshold must be between 0 and 255.")

    stats = {} if benchmark else None
    wall_started_at = perf_counter()
    detection = _timed_call(
        stats,
        "detect_contours",
        _detect_contours,
        image_bgr,
        edge_threshold,
        area_threshold,
        stats,
    )

    if legacy_mask:
        flake_mask = region_classifier.contour_mask(
            image_bgr.shape,
            detection["all_external_contours"],
        )
        result_image = _timed_call(stats, "mask_flakes", mask_flakes, image_bgr, flake_mask)
        analysis = None
        if return_details:
            internal_mask = _timed_call(
                stats,
                "internal_region_mask",
                region_classifier.internal_region_mask,
                detection["edge_mask"],
                detection["contour_groups"],
            )
            analysis = ContourAnalysis(
                edge_mask=detection["edge_mask"],
                all_external_contours=detection["all_external_contours"],
                external_contours=detection["external_contours"],
                internal_contours=detection["internal_contours"],
                contour_groups=detection["contour_groups"],
                flake_mask=flake_mask,
                internal_mask=internal_mask,
            )
        if display:
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
        if benchmark:
            benchmark_data = _benchmark_result(stats, perf_counter() - wall_started_at)
            _print_benchmark(benchmark_data)
            if analysis is not None:
                analysis.benchmark = benchmark_data
        if return_details:
            return result_image, detection["external_contours"], analysis
        return result_image, detection["external_contours"]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if profile_path is not None and profile_configuration is not None:
        raise ValueError("Provide either a profile path or profile configuration, not both.")
    if profile_configuration is None:
        profile_configuration = (
            _timed_call(
                stats,
                "load_profile_configuration",
                region_classifier.load_profile_configuration,
                profile_path,
                contrast_threshold,
            )
            if profile_path
            else {
                "classes": [],
                "filters": [],
                "minimum_size_um": None,
                "maximum_size_um": None,
            }
        )
    if not isinstance(profile_configuration, dict):
        raise ValueError("Profile configuration must be an object.")
    profile_classes = profile_configuration["classes"]
    profile_filters = profile_configuration.get("filters", [])
    class_colors = _timed_call(
        stats,
        "generate_class_colors",
        region_classifier.generate_class_colors,
        profile_classes,
        color_seed,
    )
    for profile_class in profile_classes:
        profile_class["display_color_rgb"] = class_colors[profile_class["name"]]

    classification = _timed_call(
        stats,
        "classify_contour_regions",
        region_classifier.classify_contour_regions,
        image_rgb,
        detection["edge_mask"],
        detection["contours"],
        detection["hierarchy"],
        detection["all_external_contours"],
        detection["external_indices"],
        detection["contour_indices_by_external"],
        profile_classes,
        region_flood_fill_threshold,
        stats,
        pixel_size_um,
        profile_configuration["minimum_size_um"],
        profile_configuration["maximum_size_um"],
        profile_filters,
    )
    classified_without_legend = _timed_call(
        stats,
        "create_classified_image",
        region_classifier.create_classified_image,
        image_rgb.shape,
        classification["external_mask"],
        classification["class_index_map"],
        profile_classes,
    )
    classified_image = (
        _timed_call(
            stats,
            "draw_class_legend",
            region_classifier.draw_class_legend,
            classified_without_legend,
            profile_classes,
            classification["region_results"],
            legend_position,
        )
        if draw_legend
        else classified_without_legend
    )

    contour_image = image_rgb.copy()
    cv2.drawContours(contour_image, detection["external_contours"], -1, (0, 255, 0), 2)
    cv2.drawContours(contour_image, detection["internal_contours"], -1, (255, 0, 0), 2)
    contour_edge_map, external_edge_mask, internal_edge_mask = _timed_call(
        stats,
        "create_contour_edge_map",
        _create_contour_edge_map,
        image_rgb.shape,
        detection["external_contours"],
        detection["internal_contours"],
    )
    if display:
        _show_results(image_rgb, contour_edge_map, contour_image, classified_image)

    details = None
    if return_details:
        details = {
            "image_rgb": image_rgb,
            "closed_edges": detection["edge_mask"],
            "contour_edge_map": contour_edge_map,
            "external_edge_mask": external_edge_mask,
            "internal_edge_mask": internal_edge_mask,
            "all_external_contours": detection["all_external_contours"],
            "internal_contours": detection["internal_contours"],
            "internal_contours_by_external": detection["contour_groups"],
            "external_mask": classification["external_mask"],
            "all_external_mask": classification["all_external_mask"],
            "all_region_mask": classification["all_region_mask"],
            "region_overlap_count_map": classification["region_overlap_count_map"],
            "matched_region_mask": classification["matched_region_mask"],
            "rejected_region_mask": classification["rejected_region_mask"],
            "filtered_region_mask": classification["filtered_region_mask"],
            "class_index_map": classification["class_index_map"],
            "class_masks": classification["class_masks"],
            "class_overlap_count_map": classification["class_overlap_count_map"],
            "class_stack_order": [item["name"] for item in profile_classes],
            "filter_order": [item["name"] for item in profile_filters],
            "legend_position": legend_position if draw_legend else None,
            "background_color_rgb": classification["background_color_rgb"],
            "region_flood_fill_threshold": region_flood_fill_threshold,
            "pixel_size_um": pixel_size_um,
            "profile_minimum_size_um": profile_configuration["minimum_size_um"],
            "profile_maximum_size_um": profile_configuration["maximum_size_um"],
            "region_results": classification["region_results"],
            "class_colors_rgb": class_colors,
            "class_pixel_counts": {
                item["name"]: int(np.count_nonzero(
                    classification["class_index_map"] == item["class_index"] + 1
                ))
                for item in profile_classes
            },
            "class_layer_pixel_counts": {
                item["name"]: (
                    int(np.count_nonzero(classification["class_masks"][item["name"]]))
                    if item["name"] in classification["class_masks"]
                    else 0
                )
                for item in profile_classes
            },
            "internal_mask": classification["matched_internal_mask"],
            "all_internal_mask": classification["all_internal_mask"],
            "internal_region_results": [
                item
                for item in classification["region_results"]
                if item["region_type"] == "internal"
            ],
            "profile_path": str(profile_path) if profile_path else None,
            "classified_image": classified_image,
            "classified_image_without_legend": classified_without_legend,
            "contour_img": contour_image,
        }

    if benchmark:
        benchmark_data = _benchmark_result(stats, perf_counter() - wall_started_at)
        _print_benchmark(benchmark_data)
        if details is not None:
            details["benchmark"] = benchmark_data
    if return_details:
        return classified_image, detection["external_contours"], details
    return classified_image, detection["external_contours"]


def floodfill_internal_contours(
    image_bgr,
    edge_threshold=10,
    area_threshold=DEFAULT_AREA_THRESHOLD,
    display=False,
    profile_path=None,
    contrast_threshold=DEFAULT_CONTRAST_MATCH_THRESHOLD,
    region_flood_fill_threshold=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    color_seed=None,
    draw_legend=True,
    benchmark=False,
    pixel_size_um=None,
    legend_position=DEFAULT_LEGEND_POSITION,
):
    """Return the classified image and a mask of all matched regions."""
    classified_image, _, details = find_flakes(
        image_bgr,
        edge_threshold=edge_threshold,
        area_threshold=area_threshold,
        display=False,
        return_details=True,
        profile_path=profile_path,
        contrast_threshold=contrast_threshold,
        region_flood_fill_threshold=region_flood_fill_threshold,
        color_seed=color_seed,
        draw_legend=draw_legend,
        legend_position=legend_position,
        benchmark=benchmark,
        pixel_size_um=pixel_size_um,
    )
    segmented_mask = details["matched_region_mask"]
    filled_regions, _ = cv2.findContours(
        segmented_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if display:
        plt.imshow(classified_image)
        plt.axis("off")
        plt.show()
    return classified_image, segmented_mask, filled_regions


if __name__ == "__main__":
    from tkinter import filedialog

    image_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not image_path:
        raise SystemExit("No image selected.")
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    classified_image, _, details = find_flakes(
        image_bgr,
        return_details=True,
        profile_path=AN_TEST_PROFILE_PATH,
        benchmark=True,
    )
    output_path = f"{os.path.splitext(image_path)[0]}_An_Test_pixel_labels.png"
    edge_output_path = f"{os.path.splitext(image_path)[0]}_contour_edges.png"
    if not cv2.imwrite(output_path, cv2.cvtColor(classified_image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save pixel-label image: {output_path}")
    if not cv2.imwrite(
        edge_output_path,
        cv2.cvtColor(details["contour_edge_map"], cv2.COLOR_RGB2BGR),
    ):
        raise OSError(f"Could not save contour-edge image: {edge_output_path}")
    matched = sum(item["matched_class"] is not None for item in details["region_results"])
    print(f"Saved pixel labels to: {output_path}")
    print(f"Saved contour edges to: {edge_output_path}")
    print(f"Matched {matched} of {len(details['region_results'])} contour regions.")

"""Standalone benchmark and 3D contrast viewer for the App contour pipeline.

The contour detection and region classification functions exposed by this file
come directly from ``App/Scanning``. The code below is only a test harness and
visualizer, which prevents this standalone copy from drifting away from the
implementation used during a real scan.

Run with an image path::

    python contour_finder.py sample.png

Run without a path to choose an image interactively. Use ``--help`` for profile,
threshold, plot-export, and non-interactive options.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_DIRECTORY = REPOSITORY_ROOT / "App"
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))

from Scanning import contour_finder as _app_contour_finder  # noqa: E402
from Scanning import region_classifier as _app_region_classifier  # noqa: E402


APP_CONTOUR_IMPLEMENTATION_PATH = Path(_app_contour_finder.__file__).resolve()
APP_CLASSIFIER_IMPLEMENTATION_PATH = Path(_app_region_classifier.__file__).resolve()
IDENTIFIED_POINT_COLOR = "#ff00ff"
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

# Preserve the former import surface for other experiments in this directory.
# Every production function now resolves to the App implementation.
_APP_EXPORTS = [
    name
    for name in dir(_app_contour_finder)
    if not name.startswith("__")
]
globals().update({
    name: getattr(_app_contour_finder, name)
    for name in _APP_EXPORTS
})


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read a color image while supporting spaces and non-ASCII paths."""
    path = Path(path)
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, cv2.error) as exc:
        raise OSError(f"Could not read image: {path}") from exc
    if image is None:
        raise OSError(f"Could not read image: {path}")
    return image


def _actual_point_colors(region_results) -> np.ndarray:
    colors = np.asarray(
        [region["average_color_rgb"] for region in region_results],
        dtype=np.float64,
    )
    return np.clip(colors / 255.0, 0.0, 1.0)


def _region_category(region) -> str:
    if region.get("filtered", False):
        return "filtered"
    if region.get("matched_class") is not None:
        return "identified"
    return "unmatched"


def _set_equal_contrast_limits(axis, contrast_sets) -> None:
    nonempty = [values for values in contrast_sets if len(values)]
    if not nonempty:
        return
    values = np.concatenate(nonempty, axis=0)
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(5.0, float(np.max(maximum - minimum)) / 2.0) * 1.08
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plot_contour_contrasts_3d(
    region_results,
    profile_configuration=None,
    *,
    display=True,
    save_path=None,
    annotate_references=True,
    benchmark_seconds=None,
):
    """Plot all classified regions in signed RGB-contrast space.

    Unmatched circles and filtered X markers use each region's measured average
    RGB color. Every class-identified region is a vivid magenta star, making
    class membership visible without a plot legend. Hollow diamonds and squares
    mark configured class and color-filter reference contrasts respectively.
    """
    regions = list(region_results)
    configuration = profile_configuration or {"classes": [], "filters": []}
    contrasts = (
        np.asarray([region["contrast_rgb"] for region in regions], dtype=np.float64)
        if regions
        else np.empty((0, 3), dtype=np.float64)
    )
    point_colors = (
        _actual_point_colors(regions)
        if regions
        else np.empty((0, 3), dtype=np.float64)
    )
    categories = np.asarray([_region_category(region) for region in regions])

    figure = plt.figure(figsize=(11, 8.5))
    axis = figure.add_subplot(111, projection="3d")
    category_styles = {
        "unmatched": {"marker": "o", "size": 24, "linewidth": 0.35},
        "filtered": {"marker": "X", "size": 62, "linewidth": 0.75},
        "identified": {"marker": "*", "size": 105, "linewidth": 0.9},
    }
    for category in ("unmatched", "filtered", "identified"):
        selected = categories == category
        if not np.any(selected):
            continue
        style = category_styles[category]
        colors = (
            IDENTIFIED_POINT_COLOR
            if category == "identified"
            else point_colors[selected]
        )
        axis.scatter(
            contrasts[selected, 0],
            contrasts[selected, 1],
            contrasts[selected, 2],
            c=colors,
            marker=style["marker"],
            s=style["size"],
            edgecolors="black",
            linewidths=style["linewidth"],
            depthshade=False,
            label=category.title(),
        )

    reference_contrasts = []
    for profile_class in configuration.get("classes", ()):
        if not profile_class.get("identify", True):
            continue
        contrast = np.asarray(profile_class["contrast_rgb"], dtype=np.float64)
        reference_contrasts.append(contrast)
        axis.scatter(
            [contrast[0]],
            [contrast[1]],
            [contrast[2]],
            marker="D",
            s=115,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            depthshade=False,
        )
        if annotate_references:
            group = profile_class.get("group")
            label = f"{group} / {profile_class['name']}" if group else profile_class["name"]
            axis.text(contrast[0], contrast[1], contrast[2], f"  [C][I] {label}")

    for profile_filter in configuration.get("filters", ()):
        if "contrast_rgb" not in profile_filter:
            continue
        contrast = np.asarray(profile_filter["contrast_rgb"], dtype=np.float64)
        reference_contrasts.append(contrast)
        axis.scatter(
            [contrast[0]],
            [contrast[1]],
            [contrast[2]],
            marker="s",
            s=105,
            facecolors="none",
            edgecolors="#555555",
            linewidths=1.5,
            depthshade=False,
        )
        if annotate_references:
            axis.text(
                contrast[0],
                contrast[1],
                contrast[2],
                f"  [F] {profile_filter['name']}",
                color="#444444",
            )

    axis.scatter(
        [0],
        [0],
        [0],
        c="black",
        marker="+",
        s=60,
        linewidths=1.2,
        depthshade=False,
    )
    reference_array = (
        np.asarray(reference_contrasts, dtype=np.float64).reshape(-1, 3)
        if reference_contrasts
        else np.empty((0, 3), dtype=np.float64)
    )
    _set_equal_contrast_limits(axis, (contrasts, reference_array, np.zeros((1, 3))))

    counts = Counter(categories.tolist())
    axis.set_xlabel("Red contrast (region - background)")
    axis.set_ylabel("Green contrast (region - background)")
    axis.set_zlabel("Blue contrast (region - background)")
    timing = (
        f" | processing: {benchmark_seconds:.3f} s"
        if benchmark_seconds is not None
        else ""
    )
    axis.set_title(
        "Contour-region RGB contrasts\n"
        f"{len(regions):,} total | {counts['identified']:,} identified | "
        f"{counts['filtered']:,} filtered | {counts['unmatched']:,} unmatched"
        f"{timing}\nMagenta stars are class-identified regions"
    )
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved 3D contrast plot to: {save_path.resolve()}")
    if display:
        plt.show()
    return figure


def print_region_summary(region_results) -> None:
    regions = list(region_results)
    counts = Counter(_region_category(region) for region in regions)
    classes = Counter(
        region["matched_class"]
        for region in regions
        if region.get("matched_class") is not None
    )
    filters = Counter(
        region["filtered_by"]
        for region in regions
        if region.get("filtered_by") is not None
    )
    print("\nRegion summary")
    print("-" * 48)
    print(f"Total regions : {len(regions):,}")
    print(f"Identified    : {counts['identified']:,}")
    print(f"Filtered      : {counts['filtered']:,}")
    print(f"Unmatched     : {counts['unmatched']:,}")
    for name, count in sorted(classes.items()):
        print(f"  [C][I] {name}: {count:,}")
    for name, count in sorted(filters.items()):
        print(f"  [F] {name}: {count:,}")


def print_region_points(region_results) -> None:
    """Print every plotted point in a compact, copyable text format."""
    print("\nindex  state       class/filter               contrast RGB       actual RGB")
    print("-" * 88)
    for index, region in enumerate(region_results, start=1):
        state = _region_category(region)
        label = region.get("matched_class") or region.get("filtered_by") or "-"
        contrast = tuple(region["contrast_rgb"])
        actual = tuple(region["average_color_rgb"])
        print(
            f"{index:>5}  {state:<11} {label:<26.26} "
            f"{str(contrast):<20} {actual}"
        )


def benchmark_and_plot(
    image_bgr,
    profile_path=None,
    *,
    edge_threshold=10,
    area_threshold=DEFAULT_AREA_THRESHOLD,
    contrast_threshold=DEFAULT_CONTRAST_MATCH_THRESHOLD,
    region_flood_fill_threshold=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    pixel_size_um=None,
    legend_position=DEFAULT_LEGEND_POSITION,
    display=True,
    save_plot=None,
    annotate_references=True,
    print_points=False,
):
    """Benchmark the production pipeline and plot every resulting contrast."""
    classified_image, contours, details = _app_contour_finder.find_flakes(
        image_bgr,
        edge_threshold=edge_threshold,
        area_threshold=area_threshold,
        return_details=True,
        profile_path=profile_path,
        contrast_threshold=contrast_threshold,
        region_flood_fill_threshold=region_flood_fill_threshold,
        color_seed=0,
        benchmark=True,
        pixel_size_um=pixel_size_um,
        legend_position=legend_position,
    )
    configuration = (
        _app_region_classifier.load_profile_configuration(
            profile_path,
            contrast_threshold,
        )
        if profile_path is not None
        else {"classes": [], "filters": []}
    )
    print_region_summary(details["region_results"])
    if print_points:
        print_region_points(details["region_results"])
    figure = plot_contour_contrasts_3d(
        details["region_results"],
        configuration,
        display=display,
        save_path=save_plot,
        annotate_references=annotate_references,
        benchmark_seconds=details["benchmark"]["wall_time_seconds"],
    )
    return classified_image, contours, details, figure


def _choose_image_path() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title="Choose an Image to Benchmark",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def _argument_parser() -> argparse.ArgumentParser:
    default_profile = Path(AN_TEST_PROFILE_PATH)
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the App contour classifier and plot every region in "
            "signed RGB-contrast space using its measured color."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Image to analyze. A file picker opens when omitted.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=default_profile if default_profile.is_file() else None,
        help="Profile JSON or profile directory (default: App/Profiles/An_Test).",
    )
    parser.add_argument("--edge-threshold", type=float, default=10)
    parser.add_argument("--area-threshold", type=float, default=DEFAULT_AREA_THRESHOLD)
    parser.add_argument(
        "--contrast-threshold",
        type=float,
        default=DEFAULT_CONTRAST_MATCH_THRESHOLD,
    )
    parser.add_argument(
        "--flood-threshold",
        type=float,
        default=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Micrometers per pixel, required by profiles with size limits.",
    )
    parser.add_argument(
        "--legend-position",
        choices=sorted(LEGEND_POSITIONS),
        default=DEFAULT_LEGEND_POSITION,
    )
    parser.add_argument("--save-plot", type=Path, default=None)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive Matplotlib window.",
    )
    parser.add_argument(
        "--no-reference-labels",
        action="store_true",
        help="Hide class/filter names beside their reference markers.",
    )
    parser.add_argument(
        "--print-points",
        action="store_true",
        help="Also print every plotted contrast and actual RGB value.",
    )
    return parser


# Previous single-image ``main`` is intentionally disabled while the directory
# benchmark is active. ``benchmark_and_plot`` remains available for interactive
# single-image experiments.
#
# def main(argv=None) -> int:
#     arguments = _argument_parser().parse_args(argv)
#     image_path = arguments.image or _choose_image_path()
#     ...
#     benchmark_and_plot(image_bgr, profile_path, ...)
#     return 0


def _choose_directory_path() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askdirectory(
            title="Choose an Image Directory to Benchmark"
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def _directory_argument_parser() -> argparse.ArgumentParser:
    default_profile = Path(AN_TEST_PROFILE_PATH)
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the production contour classifier over every supported "
            "image in a directory without plotting or saving output images."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Directory to benchmark. A directory picker opens when omitted.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=default_profile if default_profile.is_file() else None,
        help="Profile JSON or profile directory (default: App/Profiles/An_Test).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in nested directories.",
    )
    parser.add_argument("--edge-threshold", type=float, default=10)
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=DEFAULT_AREA_THRESHOLD,
    )
    parser.add_argument(
        "--contrast-threshold",
        type=float,
        default=DEFAULT_CONTRAST_MATCH_THRESHOLD,
    )
    parser.add_argument(
        "--flood-threshold",
        type=float,
        default=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Micrometers per pixel, required by profiles with size limits.",
    )
    return parser


def _discover_images(directory: Path, recursive: bool) -> list[Path]:
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: str(path.relative_to(directory)).casefold(),
    )


def _merge_benchmark(aggregate, benchmark) -> None:
    for name, measurement in benchmark["functions"].items():
        destination = aggregate.setdefault(
            name,
            {"calls": 0, "total_seconds": 0.0, "max_seconds": 0.0},
        )
        destination["calls"] += measurement["calls"]
        destination["total_seconds"] += measurement["total_seconds"]
        destination["max_seconds"] = max(
            destination["max_seconds"],
            measurement["max_seconds"],
        )


def _print_aggregate_benchmark(aggregate, pipeline_seconds) -> None:
    print("\nDirectory benchmark (inclusive timings across successful images)")
    print(
        f"{'Function':<30} {'Calls':>9} {'Total ms':>12} "
        f"{'Average ms':>12} {'Max ms':>12}"
    )
    print("-" * 79)
    for name, measurement in sorted(
        aggregate.items(),
        key=lambda item: item[1]["total_seconds"],
        reverse=True,
    ):
        calls = measurement["calls"]
        print(
            f"{name:<30} {calls:>9,d} "
            f"{measurement['total_seconds'] * 1000:>12.3f} "
            f"{measurement['total_seconds'] / calls * 1000:>12.3f} "
            f"{measurement['max_seconds'] * 1000:>12.3f}"
        )
    print("-" * 79)
    print(f"{'Combined pipeline time':<40} {pipeline_seconds * 1000:>12.3f}")


def benchmark_directory(
    directory,
    profile_path=None,
    *,
    recursive=False,
    edge_threshold=10,
    area_threshold=DEFAULT_AREA_THRESHOLD,
    contrast_threshold=DEFAULT_CONTRAST_MATCH_THRESHOLD,
    region_flood_fill_threshold=DEFAULT_REGION_FLOOD_FILL_THRESHOLD,
    pixel_size_um=None,
) -> int:
    """Benchmark all images in a directory without displaying or saving them."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory was not found: {directory}")
    profile_path = Path(profile_path).resolve() if profile_path is not None else None
    if profile_path is not None and not profile_path.exists():
        raise FileNotFoundError(f"Profile was not found: {profile_path}")

    image_paths = _discover_images(directory, recursive)
    if not image_paths:
        print(f"No supported images found in: {directory}")
        return 1

    print(f"Contour implementation  : {APP_CONTOUR_IMPLEMENTATION_PATH}")
    print(f"Classifier implementation: {APP_CLASSIFIER_IMPLEMENTATION_PATH}")
    print(f"Image directory         : {directory}")
    print(f"Profile                 : {profile_path if profile_path else 'none'}")
    print(f"Images discovered       : {len(image_paths):,}")

    aggregate = {}
    combined_pipeline_seconds = 0.0
    successful = 0
    failed = 0
    total_regions = 0
    total_identified = 0
    total_filtered = 0
    batch_started_at = perf_counter()

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(directory)
        prefix = f"[{index:>{len(str(len(image_paths)))}}/{len(image_paths)}]"
        print(f"{prefix} Processing {relative_path} ...", flush=True)
        try:
            image_bgr = read_image_bgr(image_path)
            # find_flakes prints an individual benchmark table. Suppress it so
            # directory mode can emit one status line and one aggregate table.
            with redirect_stdout(io.StringIO()):
                _, _, details = _app_contour_finder.find_flakes(
                    image_bgr,
                    edge_threshold=edge_threshold,
                    area_threshold=area_threshold,
                    return_details=True,
                    profile_path=profile_path,
                    contrast_threshold=contrast_threshold,
                    region_flood_fill_threshold=region_flood_fill_threshold,
                    color_seed=0,
                    draw_legend=False,
                    benchmark=True,
                    pixel_size_um=pixel_size_um,
                )
            benchmark = details["benchmark"]
            regions = details["region_results"]
            identified = sum(
                region["matched_class"] is not None for region in regions
            )
            filtered = sum(region.get("filtered", False) for region in regions)
            pipeline_seconds = benchmark["wall_time_seconds"]
            _merge_benchmark(aggregate, benchmark)
            combined_pipeline_seconds += pipeline_seconds
            successful += 1
            total_regions += len(regions)
            total_identified += identified
            total_filtered += filtered
            print(
                f"{prefix} Completed {relative_path} | "
                f"{pipeline_seconds:.3f} s | {len(regions):,} regions | "
                f"{identified:,} identified | {filtered:,} filtered",
                flush=True,
            )
        except Exception as exc:  # Continue benchmarking the remaining images.
            failed += 1
            print(
                f"{prefix} FAILED {relative_path} | "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    batch_seconds = perf_counter() - batch_started_at
    if aggregate:
        _print_aggregate_benchmark(aggregate, combined_pipeline_seconds)
    print("\nDirectory summary")
    print(f"Successful images : {successful:,}")
    print(f"Failed images     : {failed:,}")
    print(f"Total regions     : {total_regions:,}")
    print(f"Identified regions: {total_identified:,}")
    print(f"Filtered regions  : {total_filtered:,}")
    print(f"Batch elapsed time: {batch_seconds:.3f} s")
    return 0 if successful and not failed else 1


def main(argv=None) -> int:
    arguments = _directory_argument_parser().parse_args(argv)
    directory = arguments.directory or _choose_directory_path()
    if directory is None:
        print("No directory selected.")
        return 1
    return benchmark_directory(
        directory,
        arguments.profile,
        recursive=arguments.recursive,
        edge_threshold=arguments.edge_threshold,
        area_threshold=arguments.area_threshold,
        contrast_threshold=arguments.contrast_threshold,
        region_flood_fill_threshold=arguments.flood_threshold,
        pixel_size_um=arguments.pixel_size,
    )


__all__ = _APP_EXPORTS + [
    "APP_CONTOUR_IMPLEMENTATION_PATH",
    "APP_CLASSIFIER_IMPLEMENTATION_PATH",
    "read_image_bgr",
    "plot_contour_contrasts_3d",
    "print_region_summary",
    "print_region_points",
    "benchmark_and_plot",
    "benchmark_directory",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

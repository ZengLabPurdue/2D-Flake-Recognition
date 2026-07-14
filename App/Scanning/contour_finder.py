from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class ContourAnalysis:
    """Masks and contour hierarchy produced by the flake edge detector."""

    edge_mask: np.ndarray
    all_external_contours: list[np.ndarray]
    external_contours: list[np.ndarray]
    internal_contours: list[np.ndarray]
    contour_groups: list[tuple[np.ndarray, list[np.ndarray]]]
    flake_mask: np.ndarray
    internal_mask: np.ndarray | None


def _validate_image(image_bgr):
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("A three-channel BGR image is required.")


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
    """Close an edge through the image border when it leaves the frame."""
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

    ys, xs = np.nonzero(labels[:, width - border_width :])
    add_distances(labels[ys, width - border_width + xs], width - 1 + ys)

    ys, xs = np.nonzero(labels[height - border_width :, :])
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

        gaps = []
        for index, distance in enumerate(distances):
            next_distance = distances[(index + 1) % len(distances)]
            gaps.append(((next_distance - distance) % perimeter, distance, next_distance))

        _, gap_start, gap_end = max(gaps, key=lambda item: item[0])
        arc_length = (gap_start - gap_end) % perimeter
        for offset in range(arc_length + 1):
            x, y = _perimeter_point(gap_end + offset, width, height)
            closed[y, x] = 255

    return closed


def _edge_mask(image_bgr, edge_threshold):
    # The established detector uses the red and green Sobel magnitudes together.
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


def _contour_mask(image_shape, contours):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def _seed_inside_contour(contour, allowed_mask):
    contour_mask = _contour_mask(allowed_mask.shape, [contour])
    candidates = (contour_mask > 0) & allowed_mask
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


def _internal_region_mask(edge_mask, contour_groups):
    """Flood-fill the enclosed regions represented by internal contours."""
    height, width = edge_mask.shape
    internal_mask = np.zeros((height, width), dtype=np.uint8)
    processed_mask = np.zeros((height, width), dtype=np.uint8)

    for external_contour, internal_contours in contour_groups:
        x, y, contour_width, contour_height = cv2.boundingRect(external_contour)
        x1, y1 = max(0, x - 1), max(0, y - 1)
        x2 = min(width, x + contour_width + 1)
        y2 = min(height, y + contour_height + 1)

        local_external = external_contour.copy()
        local_external[:, 0] -= (x1, y1)
        external_mask = _contour_mask((y2 - y1, x2 - x1), [local_external])

        local_edges = edge_mask[y1:y2, x1:x2]
        local_internal = internal_mask[y1:y2, x1:x2]
        local_processed = processed_mask[y1:y2, x1:x2]
        allowed = (
            (external_mask > 0)
            & (local_edges == 0)
            & (local_processed == 0)
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
            if np.any(region):
                local_internal[region] = 255
                local_processed[region] = 255
                allowed[region] = False

    return internal_mask


def analyze_contours(
    image_bgr,
    edge_threshold=10,
    area_threshold=500,
    fill_internal_regions=True,
):
    """Detect external flakes, their internal contours, and their pixel masks."""
    _validate_image(image_bgr)
    if edge_threshold < 0:
        raise ValueError("The edge threshold cannot be negative.")
    if area_threshold < 0:
        raise ValueError("The area threshold cannot be negative.")

    edge_mask = _edge_mask(image_bgr, edge_threshold)
    contours, hierarchy = cv2.findContours(
        edge_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None:
        return ContourAnalysis(
            edge_mask,
            [],
            [],
            [],
            [],
            (
                np.zeros(image_bgr.shape[:2], dtype=np.uint8)
                if fill_internal_regions
                else None
            ),
            np.zeros(image_bgr.shape[:2], dtype=np.uint8),
        )

    hierarchy = hierarchy[0]
    all_external_indices = [index for index, item in enumerate(hierarchy) if item[3] == -1]
    external_indices = [
        index
        for index in all_external_indices
        if cv2.contourArea(contours[index]) >= area_threshold
    ]
    external_index_set = set(external_indices)
    groups_by_external = {index: [] for index in external_indices}

    for index, contour in enumerate(contours):
        if hierarchy[index][3] == -1:
            continue
        outer_parent = _outermost_parent(index, hierarchy)
        if outer_parent in external_index_set:
            groups_by_external[outer_parent].append(contour)

    all_external = [contours[index] for index in all_external_indices]
    external = [contours[index] for index in external_indices]
    contour_groups = [
        (contours[index], groups_by_external[index])
        for index in external_indices
    ]
    internal = [contour for _, group in contour_groups for contour in group]

    return ContourAnalysis(
        edge_mask=edge_mask,
        all_external_contours=all_external,
        external_contours=external,
        internal_contours=internal,
        contour_groups=contour_groups,
        flake_mask=_contour_mask(image_bgr.shape, all_external),
        internal_mask=(
            _internal_region_mask(edge_mask, contour_groups)
            if fill_internal_regions
            else None
        ),
    )


def mask_flakes(image_bgr, flake_mask):
    """Return a copy of the BGR image with detected flakes set to black."""
    _validate_image(image_bgr)
    if flake_mask.shape != image_bgr.shape[:2]:
        raise ValueError("The flake mask does not match the image.")
    background = image_bgr.copy()
    background[flake_mask > 0] = 0
    return background


def background_color_rgb(image_bgr, analysis=None):
    """Return the mean RGB background after masking every detected flake."""
    _validate_image(image_bgr)
    analysis = analysis or analyze_contours(image_bgr, fill_internal_regions=False)
    background_pixels = image_bgr[analysis.flake_mask == 0]
    if len(background_pixels) == 0:
        raise ValueError("The contour mask covers the entire image background.")

    blue, green, red = np.mean(background_pixels, axis=0)
    return int(round(red)), int(round(green)), int(round(blue))


def region_contrast_rgb(image_bgr, region_mask, analysis=None):
    """Return signed RGB contrast as region color minus background color."""
    _validate_image(image_bgr)
    if region_mask is None or region_mask.shape != image_bgr.shape[:2]:
        raise ValueError("The region mask does not match the image.")
    if not np.any(region_mask > 0):
        raise ValueError("The selected region is empty.")

    blue, green, red = np.mean(image_bgr[region_mask > 0], axis=0)
    background = background_color_rgb(image_bgr, analysis)
    region = (red, green, blue)
    return tuple(
        int(round(region_channel - background_channel))
        for region_channel, background_channel in zip(region, background)
    )


def find_flakes(image_bgr, edge_threshold=10, area_threshold=500, return_details=False):
    """Return the flake-masked BGR image and area-filtered external contours."""
    analysis = analyze_contours(
        image_bgr,
        edge_threshold,
        area_threshold,
        fill_internal_regions=return_details,
    )
    background = mask_flakes(image_bgr, analysis.flake_mask)
    if return_details:
        return background, analysis.external_contours, analysis
    return background, analysis.external_contours

import os
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog


AN_TEST_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "App"
    / "Profiles"
    / "An_Test"
    / "profile.json"
)

def _load_profile_classes(profile_path):
    profile_path = Path(profile_path)

    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Profile was not found: {profile_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read profile: {profile_path}") from exc

    profile_classes = []

    for profile_class in profile_data.get("classes", []):
        try:
            color = profile_class["average_color_rgb"]
            average_color_rgb = np.array(
                [color["red"], color["green"], color["blue"]],
                dtype=np.float64,
            )
            tolerance = int(profile_class["flood_fill"]["threshold"])
            name = str(profile_class["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Profile class has invalid color data: {profile_path}") from exc

        if np.any(average_color_rgb < 0) or np.any(average_color_rgb > 255):
            raise ValueError(f"Profile class {name!r} has an invalid average color.")
        if not 0 <= tolerance <= 255:
            raise ValueError(f"Profile class {name!r} has an invalid tolerance.")

        profile_classes.append({
            "name": name,
            "average_color_rgb": average_color_rgb,
            "tolerance": tolerance,
        })

    if not profile_classes:
        raise ValueError(f"Profile contains no color classes: {profile_path}")

    return profile_classes

def _match_profile_class(average_color_rgb, profile_classes):
    matches = []

    for profile_class in profile_classes:
        channel_difference = np.abs(
            average_color_rgb - profile_class["average_color_rgb"]
        )

        if np.all(channel_difference <= profile_class["tolerance"]):
            matches.append((np.linalg.norm(channel_difference), profile_class))

    if not matches:
        return None

    return min(matches, key=lambda match: match[0])[1]

def _perimeter_point(distance, width, height):
    top_len = width - 1
    right_len = height - 1
    bottom_len = width - 1
    perimeter = 2 * (top_len + right_len)
    distance %= perimeter

    if distance <= top_len:
        return distance, 0

    distance -= top_len
    if distance <= right_len:
        return width - 1, distance

    distance -= right_len
    if distance <= bottom_len:
        return width - 1 - distance, height - 1

    distance -= bottom_len
    return 0, height - 1 - distance

def _close_border_touching_edges(edge_mask, border_width=3):
    height, width = edge_mask.shape

    if height < 2 or width < 2:
        return edge_mask

    closed = edge_mask.copy()
    _, labels = cv2.connectedComponents(edge_mask, connectivity=8)
    perimeter = 2 * ((width - 1) + (height - 1))
    border_width = max(1, min(border_width, height, width))
    border_distances_by_label = {}

    def add_distances(label_values, distances):
        for label, distance in zip(label_values, distances):
            if label == 0:
                continue

            border_distances_by_label.setdefault(int(label), set()).add(int(distance) % perimeter)

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

    for border_distances in border_distances_by_label.values():
        border_distances = sorted(border_distances)

        if len(border_distances) < 2:
            continue

        gaps = []
        for i, distance in enumerate(border_distances):
            next_distance = border_distances[(i + 1) % len(border_distances)]
            gap = (next_distance - distance) % perimeter
            gaps.append((gap, distance, next_distance))

        _, gap_start, gap_end = max(gaps, key=lambda item: item[0])
        arc_length = (gap_start - gap_end) % perimeter

        for offset in range(arc_length + 1):
            x, y = _perimeter_point(gap_end + offset, width, height)
            closed[y, x] = 255

    return closed

def _get_outer_parent(index, hierarchy):
    parent = hierarchy[index][3]

    while parent != -1 and hierarchy[parent][3] != -1:
        parent = hierarchy[parent][3]

    return parent

def _find_seed_in_contour(contour, allowed_mask):
    contour_mask = np.zeros(allowed_mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    candidates = (contour_mask > 0) & allowed_mask
    ys, xs = np.nonzero(candidates)

    if len(xs) == 0:
        return None

    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        if 0 <= cy < allowed_mask.shape[0] and 0 <= cx < allowed_mask.shape[1]:
            if candidates[cy, cx]:
                return cx, cy

    x, y, w, h = cv2.boundingRect(contour)
    target_x = x + w / 2
    target_y = y + h / 2
    closest_index = np.argmin((xs - target_x) ** 2 + (ys - target_y) ** 2)

    return int(xs[closest_index]), int(ys[closest_index])

def _find_internal_region_pixels(image_rgb, closed_edges, contour_groups, profile_classes=None):
    height, width = closed_edges.shape
    internal_mask = np.zeros((height, width), dtype=np.uint8)
    processed_mask = np.zeros((height, width), dtype=np.uint8)
    region_results = []

    for external_contour, internal_contours in contour_groups:
        x, y, w, h = cv2.boundingRect(external_contour)
        x1 = max(0, x - 1)
        y1 = max(0, y - 1)
        x2 = min(width, x + w + 1)
        y2 = min(height, y + h + 1)
        local_height = y2 - y1
        local_width = x2 - x1

        local_external_contour = external_contour.copy()
        local_external_contour[:, 0, 0] -= x1
        local_external_contour[:, 0, 1] -= y1

        external_mask = np.zeros((local_height, local_width), dtype=np.uint8)
        cv2.drawContours(external_mask, [local_external_contour], -1, 255, thickness=cv2.FILLED)

        local_closed_edges = closed_edges[y1:y2, x1:x2]
        local_internal_mask = internal_mask[y1:y2, x1:x2]
        local_processed_mask = processed_mask[y1:y2, x1:x2]
        allowed_mask = (external_mask > 0) & (local_closed_edges == 0) & (local_processed_mask == 0)

        for internal_contour in internal_contours:
            local_internal_contour = internal_contour.copy()
            local_internal_contour[:, 0, 0] -= x1
            local_internal_contour[:, 0, 1] -= y1

            seed = _find_seed_in_contour(local_internal_contour, allowed_mask)

            if seed is None:
                continue

            flood_image = np.zeros((local_height, local_width), dtype=np.uint8)
            flood_mask = np.zeros((local_height + 2, local_width + 2), dtype=np.uint8)
            flood_mask[1:-1, 1:-1] = np.where(allowed_mask, 0, 1).astype(np.uint8)

            cv2.floodFill(flood_image, flood_mask, seed, 255, flags=4)

            region_mask = flood_image == 255

            if not np.any(region_mask):
                continue

            region_pixels = image_rgb[y1:y2, x1:x2][region_mask]
            average_color_rgb = np.mean(region_pixels, axis=0)
            matched_class = (
                _match_profile_class(average_color_rgb, profile_classes)
                if profile_classes is not None
                else None
            )
            is_match = profile_classes is None or matched_class is not None

            if is_match:
                local_internal_mask[region_mask] = 255

            local_processed_mask[region_mask] = 255
            allowed_mask[region_mask] = False

            region_results.append({
                "seed_point": (seed[0] + x1, seed[1] + y1),
                "average_color_rgb": tuple(
                    int(round(channel)) for channel in average_color_rgb
                ),
                "matched_class": matched_class["name"] if matched_class else None,
                "pixel_count": int(np.count_nonzero(region_mask)),
            })

    return internal_mask, processed_mask, region_results

def _create_classified_image(image_shape, external_contours, internal_mask):
    """Return an RGB image: background black, external white, internal red."""
    height, width = image_shape[:2]
    external_mask = np.zeros((height, width), dtype=np.uint8)

    if external_contours:
        cv2.drawContours(external_mask, external_contours, -1, 255, thickness=cv2.FILLED)

    classified_image = np.zeros((height, width, 3), dtype=np.uint8)
    classified_image[external_mask > 0] = (255, 255, 255)
    classified_image[internal_mask > 0] = (255, 0, 0)

    return classified_image, external_mask

def find_flakes(
    image_bgr,
    edge_threshold=10,
    area_threshold=500,
    display=False,
    return_details=False,
    profile_path=None,
):
    """Classify pixels and return ``(classified_rgb, external_contours)``.

    Background pixels are black, pixels inside an area-filtered external contour
    are white, and matching internal regions are red. When ``profile_path`` is
    provided, nonmatching internal regions remain white.
    """

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    profile_classes = _load_profile_classes(profile_path) if profile_path else None
    
    R = image_bgr[:, :, 2]
    G = image_bgr[:, :, 1]
    
    process_image = np.stack((R, G), axis=2)
    
    smoothed = cv2.GaussianBlur(process_image, (5, 5), 0)
    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(np.sum(grad_x**2 + grad_y**2, axis=2))
    #magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    #magnitude[magnitude < threshold] = 0
    
    binary = np.where(magnitude >= edge_threshold, 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed_edges = _close_border_touching_edges(cleaned)
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        all_external_contours = []
        external_contours = []
        internal_contours = []
        contour_groups = []
    else:
        hierarchy = hierarchy[0]
        all_external_indices = [
            i for i, h in enumerate(hierarchy)
            if h[3] == -1
        ]
        external_indices = [
            i for i, h in enumerate(hierarchy)
            if h[3] == -1 and cv2.contourArea(contours[i]) >= area_threshold
        ]
        external_index_set = set(external_indices)

        all_external_contours = [contours[i] for i in all_external_indices]
        external_contours = [contours[i] for i in external_indices]
        internal_contours_by_external = {i: [] for i in external_indices}
        internal_contours = [
            cnt for i, cnt in enumerate(contours)
            if hierarchy[i][3] != -1 and _get_outer_parent(i, hierarchy) in external_index_set
        ]

        for i, cnt in enumerate(contours):
            if hierarchy[i][3] == -1:
                continue

            outer_parent = _get_outer_parent(i, hierarchy)

            if outer_parent in external_index_set:
                internal_contours_by_external[outer_parent].append(cnt)

        contour_groups = [
            (contours[i], internal_contours_by_external[i])
            for i in external_indices
    ]
    
    area_filtered_contours = external_contours
    internal_mask, all_internal_mask, region_results = _find_internal_region_pixels(
        image_rgb,
        closed_edges,
        contour_groups,
        profile_classes,
    )
    classified_image, external_mask = _create_classified_image(
        image_rgb.shape,
        area_filtered_contours,
        internal_mask,
    )
    
    contour_img = image_rgb.copy()
    
    cv2.drawContours(contour_img, area_filtered_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_img, internal_contours, -1, (255, 0, 0), 2)
    
    if display:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].imshow(image_rgb)
        axs[0, 0].set_title("Original")
        
        axs[0, 1].imshow(closed_edges, cmap='gray')
        axs[0, 1].set_title("Edges")
        
        axs[1, 0].imshow(contour_img)
        axs[1, 0].set_title("Detected Flakes")
        
        axs[1, 1].imshow(classified_image)
        axs[1, 1].set_title("Pixel Classification")
        
        for ax in axs.ravel():
            ax.axis('off')
        
        #plt.suptitle(filename)
        plt.tight_layout()
        plt.show()
    
    if return_details:
        details = {
            "image_rgb": image_rgb,
            "closed_edges": closed_edges,
            "all_external_contours": all_external_contours,
            "internal_contours": internal_contours,
            "internal_contours_by_external": contour_groups,
            "external_mask": external_mask,
            "internal_mask": internal_mask,
            "all_internal_mask": all_internal_mask,
            "internal_region_results": region_results,
            "profile_path": str(profile_path) if profile_path else None,
            "classified_image": classified_image,
            "contour_img": contour_img,
        }

        return classified_image, area_filtered_contours, details
    
    return classified_image, area_filtered_contours

def floodfill_internal_contours(
    image_bgr,
    edge_threshold=10,
    area_threshold=500,
    display=False,
    profile_path=None,
):
    classified_image, _, details = find_flakes(
        image_bgr,
        edge_threshold=edge_threshold,
        area_threshold=area_threshold,
        display=False,
        return_details=True,
        profile_path=profile_path,
    )

    segmented_mask = details["internal_mask"]
    filled_regions, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        plt.figure(figsize=(10, 8))
        plt.imshow(classified_image)
        plt.title("Pixel Classification")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return classified_image, segmented_mask, filled_regions

if __name__ == "__main__":
    '''
    folder_path = filedialog.askdirectory(title="Select Image Folder")

    extensions = (".png", ".jpg", ".jpeg", ".bmp")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            image_path = os.path.join(folder_path, filename)

            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

            find_flakes(image_bgr, display=True)
    '''
            
    image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])

    if not image_path:
        raise SystemExit("No image selected.")

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    _, _, details = find_flakes(
        image_bgr,
        display=False,
        return_details=True,
        profile_path=AN_TEST_PROFILE_PATH,
    )

    # Build a raw label image with exactly one output pixel per input pixel.
    # OpenCV uses BGR: black = background, white = external, red = internal.
    pixel_labels_bgr = np.zeros_like(image_bgr)
    pixel_labels_bgr[details["external_mask"] > 0] = (255, 255, 255)
    pixel_labels_bgr[details["internal_mask"] > 0] = (0, 0, 255)
    output_path = f"{os.path.splitext(image_path)[0]}_An_Test_pixel_labels.png"

    if not cv2.imwrite(output_path, pixel_labels_bgr):
        raise OSError(f"Could not save pixel-label image: {output_path}")

    height, width = pixel_labels_bgr.shape[:2]
    matched_regions = sum(
        result["matched_class"] is not None
        for result in details["internal_region_results"]
    )
    tested_regions = len(details["internal_region_results"])
    print(f"Saved {width}x{height} pixel-label image to: {output_path}")
    print(f"Matched {matched_regions} of {tested_regions} internal regions to An_Test.")

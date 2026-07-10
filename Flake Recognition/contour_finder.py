import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

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

def find_flakes(image_bgr, edge_threshold=10, area_threshold=500, display=False, return_details=False):

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
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
    
    area_filtered_contours = external_contours
    
    contour_img = image_rgb.copy()
    background_img = image_rgb.copy()
    
    cv2.drawContours(contour_img, area_filtered_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_img, internal_contours, -1, (255, 0, 0), 2)
    cv2.drawContours(background_img, all_external_contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    
    if display:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].imshow(image_rgb)
        axs[0, 0].set_title("Original")
        
        axs[0, 1].imshow(closed_edges, cmap='gray')
        axs[0, 1].set_title("Edges")
        
        axs[1, 0].imshow(contour_img)
        axs[1, 0].set_title("Detected Flakes")
        
        axs[1, 1].imshow(background_img)
        axs[1, 1].set_title("Masked Background")
        
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
            "internal_contours_by_external": [
                (contours[i], internal_contours_by_external[i])
                for i in external_indices
            ] if hierarchy is not None else [],
            "contour_img": contour_img,
        }

        return background_img, area_filtered_contours, details
    
    return background_img, area_filtered_contours

def floodfill_internal_contours(image_bgr, edge_threshold=10, area_threshold=500, display=False):
    _, external_contours, details = find_flakes(
        image_bgr,
        edge_threshold=edge_threshold,
        area_threshold=area_threshold,
        display=False,
        return_details=True,
    )

    image_rgb = details["image_rgb"]
    closed_edges = details["closed_edges"]
    height, width = closed_edges.shape
    segmented_mask = np.zeros((height, width), dtype=np.uint8)

    for external_contour, internal_contours in details["internal_contours_by_external"]:
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
        local_segmented_mask = segmented_mask[y1:y2, x1:x2]
        allowed_mask = (external_mask > 0) & (local_closed_edges == 0) & (local_segmented_mask == 0)

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

            local_segmented_mask[region_mask] = 255
            allowed_mask[region_mask] = False

    floodfill_overlay = image_rgb.copy()
    colored_regions = image_rgb.copy()
    colored_regions[segmented_mask > 0] = (255, 0, 0)
    floodfill_overlay = cv2.addWeighted(floodfill_overlay, 0.7, colored_regions, 0.3, 0)

    filled_regions, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(floodfill_overlay, external_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(floodfill_overlay, details["internal_contours"], -1, (255, 0, 0), 1)
    cv2.drawContours(floodfill_overlay, filled_regions, -1, (255, 255, 0), 2)

    if display:
        plt.figure(figsize=(10, 8))
        plt.imshow(floodfill_overlay)
        plt.title("Flood Fill Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return floodfill_overlay, segmented_mask, filled_regions

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
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    find_flakes(image_bgr, display=True)
    floodfill_internal_contours(image_bgr, display=True)

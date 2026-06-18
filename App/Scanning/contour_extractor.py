from tkinter import filedialog
import cv2
import numpy as np

def pick_point(image_bgr):

    state = {"seed": None}
    display = image_bgr.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["seed"] = (x, y)

            clicked_display = display.copy()
            cv2.circle(clicked_display, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Click seed point", clicked_display)

            print(f"Selected seed point: x={x}, y={y}")

    cv2.namedWindow("Click seed point", cv2.WINDOW_NORMAL)
    cv2.imshow("Click seed point", display)
    cv2.setMouseCallback("Click seed point", mouse_callback)

    print("Click a point on the image. Press q or ESC to cancel.")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if state["seed"] is not None:
            break

        if key == ord("q") or key == 27:
            cv2.destroyWindow("Click seed point")
            raise RuntimeError("Point selection cancelled.")

    cv2.destroyWindow("Click seed point")
    return state["seed"]

def get_region_from_point(
    image_bgr,
    seed_point,
    threshold=15,
    connectivity=8,
    min_area=5,
):

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    flags = (
        connectivity
        | cv2.FLOODFILL_FIXED_RANGE
        | cv2.FLOODFILL_MASK_ONLY
        | (255 << 8)
    )

    cv2.floodFill(
        image=gray.copy(),
        mask=mask,
        seedPoint=seed_point,
        newVal=0,
        loDiff=threshold,
        upDiff=threshold,
        flags=flags,
    )

    region_mask = mask[1:-1, 1:-1]

    contours, _ = cv2.findContours(
        region_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) >= min_area
    ]

    if not contours:
        return [], [], region_mask, None

    contour = max(contours, key=cv2.contourArea)

    contour_points = contour.reshape(-1, 2).astype(int).tolist()

    ys, xs = np.where(region_mask > 0)
    region_pixels = np.column_stack([xs, ys]).astype(int).tolist()

    return contour_points, region_pixels, region_mask, contour

def get_contour(image_bgr):
    seed_point = pick_point(image_bgr)

    contour_points, _, _, _ = get_region_from_point(
        image_bgr=image_bgr,
        seed_point=seed_point,
        threshold=10,
        connectivity=8,
    )

    return contour_points

if __name__ == "__main__":

    path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    seed_point = pick_point(image_bgr)

    contour_points, region_pixels, region_mask, contour = get_region_from_point(
        image_bgr=image_bgr,
        seed_point=seed_point,
        threshold=10,
        connectivity=8,
    )

    print(f"Seed point: {seed_point}")
    print(f"Region pixels found: {len(region_pixels)}")
    print(f"Contour points found: {len(contour_points)}")

    '''
    print("\nContour points:")
    print("[")
    for point in contour_points:
        print(f"    {point},")
    print("]")
    '''
    
    overlay = image_bgr.copy()

    if contour is not None:
        colored_region = image_bgr.copy()
        colored_region[region_mask > 0] = (0, 255, 0)

        overlay = cv2.addWeighted(image_bgr, 0.7, colored_region, 0.3, 0)

        # Draw contour in red
        cv2.drawContours(overlay, [contour], -1, (255, 255, 255), 2)

        # Draw selected seed point in blue
        cv2.circle(overlay, seed_point, 5, (0, 0, 255), -1)

    cv2.namedWindow("Selected Region", cv2.WINDOW_NORMAL)
    cv2.imshow("Selected Region", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
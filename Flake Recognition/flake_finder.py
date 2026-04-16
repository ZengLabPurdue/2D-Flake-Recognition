import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog


def _is_likely_chip_boundary_contour(cnt, img_w: int, img_h: int) -> bool:
    """
    Reject Sobel contours that trace a full die edge / large material boundary
    (long thin band spanning most of the frame), which are not flakes.
    """
    W, H = float(img_w), float(img_h)
    if W <= 1 or H <= 1:
        return False
    x, y, w, h = cv2.boundingRect(cnt)
    area = float(cv2.contourArea(cnt))

    # Half-frame or larger filled region (single huge blob)
    if area >= 0.30 * W * H:
        return True
    if w * h >= 0.38 * W * H:
        return True

    # Vertical seam: spans most of the height, bbox much taller than wide (chip / die edge)
    if h >= 0.55 * H and w <= max(0.22 * W, 36.0):
        return True
    # Horizontal seam
    if w >= 0.55 * W and h <= max(0.22 * H, 36.0):
        return True

    # Bbox spans nearly an entire axis (frame-filling edge trace)
    if w >= 0.86 * W or h >= 0.86 * H:
        return True

    # Very elongated contour whose long side is most of the image (jagged boundary line)
    short_side = max(float(min(w, h)), 1.0)
    long_side = float(max(w, h))
    aspect = long_side / short_side
    if aspect >= 15.0 and long_side >= 0.60 * max(W, H):
        return True

    return False


def _filter_chip_boundary_contours(contours, img_w: int, img_h: int):
    return [c for c in contours if not _is_likely_chip_boundary_contour(c, img_w, img_h)]


def _build_yellow_mask(image_bgr, hue_lo=15, hue_hi=38, sat_min=60, val_min=80):
    """Boolean (H,W) mask — True where pixel is in the yellow HSV band."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (h >= hue_lo) & (h <= hue_hi) & (s >= sat_min) & (v >= val_min)


def find_flakes(
    image_bgr,
    edge_threshold=10,
    area_threshold=500,
    display=False,
    filter_chip_boundary=True,
    suppress_yellow=False,
    yellow_hue_lo=15,
    yellow_hue_hi=38,
    yellow_sat_min=60,
    yellow_val_min=80,
):

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_bgr.shape[:2]
    
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

    if suppress_yellow:
        ymask = _build_yellow_mask(image_bgr,
                                   hue_lo=yellow_hue_lo, hue_hi=yellow_hue_hi,
                                   sat_min=yellow_sat_min, val_min=yellow_val_min)
        binary[ymask] = 0

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area_filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_threshold]
    if filter_chip_boundary:
        area_filtered_contours = _filter_chip_boundary_contours(area_filtered_contours, img_w, img_h)

    contour_img = image_rgb.copy()
    background_img = image_rgb.copy()
    
    cv2.drawContours(contour_img, area_filtered_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(background_img, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    
    if display:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].imshow(image_rgb)
        axs[0, 0].set_title("Original")
        
        axs[0, 1].imshow(cleaned, cmap='gray')
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
    
    return background_img, area_filtered_contours

if __name__ == "__main__":
    folder_path = filedialog.askdirectory(title="Select Image Folder")

    extensions = (".png", ".jpg", ".jpeg", ".bmp")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            image_path = os.path.join(folder_path, filename)

            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

            find_flakes(image_bgr, display=True)
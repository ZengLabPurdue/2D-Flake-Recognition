from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

def chip_filter(image, threshold=None, sample=30, display=False):

    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("chip_filter requires a BGR color image.")

    non_black_mask = np.any(image[:, :, :3] != 0, axis=2)

    if (display):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    if not np.any(non_black_mask):
        return np.zeros(image.shape[:2], dtype=np.uint8)

    if threshold is None:
        sample = max(1, int(sample))
        sampled_image = image[::sample, ::sample]
        sampled_mask = non_black_mask[::sample, ::sample]
        values = sampled_image[:, :, 0][sampled_mask]

        if values.size == 0:
            values = image[:, :, 0][non_black_mask]

        hist = np.bincount(values, minlength=256)
        threshold = threshold_after_highest_peak(hist, display=display)
        #print("Threshold:", threshold)

    blue = image[:, :, 0] 

    binary = ((blue >= threshold) & non_black_mask).astype(np.uint8) * 255
    h, w = binary.shape[:2]

    if (display):
        plt.imshow(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
        plt.axis("off")
        plt.show()

    #binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    return binary

def threshold_after_highest_peak(hist, smoothing=30, min_prominence=0.05, display=True):

    if smoothing > 1:
        hist = np.convolve(hist, np.ones(smoothing)/smoothing, mode='same')

    abs_prominence = min_prominence * np.max(hist)

    peaks, _ = find_peaks(hist, prominence=abs_prominence)

    if display:
        plt.figure(figsize=(10,5))
        plt.bar(np.arange(256), hist, width=1, color='lightgray', label='Raw histogram')
        plt.plot(hist, color='blue', linewidth=2, label='Smoothed histogram')
        plt.scatter(peaks, hist[peaks], color='red', s=50, label='Peaks')
        plt.xlabel('Blue Intensity')
        plt.ylabel('Pixel Count')
        plt.title('Histogram of Blue Channel (Smoothed)')
        plt.legend()
        plt.show()

    if len(peaks) == 1:
        if peaks[0] < 75: return 256
        else: return 0
    elif len(peaks) == 0: return 0

    top_two_peaks = np.sort(peaks)[-2:] if len(peaks) > 1 else peaks

    threshold = (top_two_peaks[0] + top_two_peaks[1]) // 2

    if display:
        x = np.arange(256)
        plt.figure()
        plt.bar(
            x[:threshold],
            hist[:threshold],
        )
        plt.bar(
            x[threshold:],
            hist[threshold:],
        )
        plt.axvline(threshold)
        plt.xlabel("Blue Intensity")
        plt.ylabel("Pixel Count")
        plt.title(f"Histogram with Threshold = {threshold}")
        plt.show()

    return threshold

def find_chips(filter_map, true_map):
    binary_map = (filter_map > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= 1000]
    
    chips = []
    
    for _, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        chips.append((x,y,w,h))
        cv2.rectangle(true_map, (x, y), (x+w, y+h), (255,255,0), 5)
    return chips, true_map

def select_and_filter_map(map_image=None, save_path=None, display=None):
    map_path = None

    if map_image is None:
        selected_path = filedialog.askopenfilename(
            title="Select Map Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not selected_path:
            return None

        map_path = Path(selected_path)
        map_image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if map_image is None:
            raise ValueError(f"Could not read map image: {map_path}")

        if display is None:
            display = True
    elif display is None:
        display = False

    if map_image.ndim != 3 or map_image.shape[2] < 3:
        raise ValueError("select_and_filter_map requires a BGR color map.")

    filtered_map = chip_filter(map_image, display=False)

    if map_path is not None and save_path is None:
        save_path = filedialog.asksaveasfilename(
            title="Save Filtered Map",
            initialdir=str(map_path.parent),
            initialfile=f"{map_path.stem}_filtered.png",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("Bitmap image", "*.bmp"),
                ("TIFF image", "*.tif *.tiff"),
            ],
        )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), filtered_map):
            raise OSError(f"Could not save filtered map to: {save_path}")
        print(f"Filtered map saved to: {save_path}")

    if not display:
        return filtered_map

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(map_path.name if map_path is not None else "2x Scan Map")

    axes[0].imshow(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Map")
    axes[0].axis("off")

    axes[1].imshow(filtered_map, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Filtered Map")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return filtered_map

if __name__ == "__main__":
    select_and_filter_map()

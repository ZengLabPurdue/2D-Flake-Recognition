import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import find_peaks

def wafer_filter(image, threshold=None, sample=30, display=False):

    if (display):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    if threshold:
        pass
    else:
        sample = 30
        values = image[::sample, ::sample, 0].ravel()

        hist = np.bincount(values, minlength=256)
        threshold = threshold_after_highest_peak(hist, display=display)
        print("Threshold:", threshold)

    blue = image[:, :, 0] 

    binary = (blue >= threshold).astype(np.uint8) * 255
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

def find_wafers(filter_map, true_map):
    binary_map = (filter_map > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= 1000]
    
    wafers = []
    
    for _, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        wafers.append((x,y,w,h))
        cv2.rectangle(true_map, (x, y), (x+w, y+h), (255,255,0), 5)
    return wafers, true_map

if __name__ == "__main__":
    image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.imshow(cv2.cvtColor(wafer_filter(image, display=True), cv2.COLOR_GRAY2RGB))
    plt.axis("off")
    plt.show()
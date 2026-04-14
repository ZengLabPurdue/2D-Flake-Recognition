import cv2
from tkinter import filedialog
import Util
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#----------------------------
# Load Image
#----------------------------

'''
image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
'''

def find_nearest_colors(image_rgb, reference_colors, output_colors=None, output_values=None, use_lab=False):

    start_time = time.time()

    if (output_colors is None) == (output_values is None):
        raise ValueError("Provide exactly one of output_colors or output_values")

    h, w = image_rgb.shape[:2]

    ref = np.asarray(reference_colors, dtype=np.uint8)

    if use_lab:
        image_cs = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_cs = cv2.cvtColor(ref.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    else:
        #image_cs = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_cs = image_rgb
        ref_cs = ref.astype(np.float32)

    pixels = image_cs.reshape(-1, 3)

    distances = np.linalg.norm(pixels[:, None, :] - ref_cs[None, :, :], axis=2)

    nearest_idx = np.argmin(distances, axis=1)

    print(f"Nearest color scan finished in {time.time() - start_time:.2f}s")

    counts = np.bincount(nearest_idx, minlength=len(reference_colors))
    color_counts = {tuple(ref[i]): int(counts[i]) for i in range(len(reference_colors))}

    if output_colors is not None:
        out = np.asarray(output_colors, dtype=np.uint8)
        labeled_pixels = out[nearest_idx]
        return labeled_pixels.reshape(h, w, 3), color_counts
    else:
        out = np.asarray(output_values)
        numeric_pixels = out[nearest_idx]
        return numeric_pixels.reshape(h, w), color_counts

def scan_windows(filtered_image, window_zoom=20):

    h, w = filtered_image.shape

    nx = w // window_zoom
    ny = h // window_zoom

    window_sums = np.zeros((ny, nx), dtype=np.int32)

    for y in range(ny):
        for x in range(nx):
            y0 = y * window_zoom
            y1 = y0 + window_zoom
            x0 = x * window_zoom
            x1 = x0 + window_zoom

            window_sums[y, x] = filtered_image[y0:y1, x0:x1].sum()

    return window_sums

# RGB
reference_colors = [
    [146, 127, 111],   # Background
    #[152, 143, 115],  # Thin flake
    [0,   64, 110],    # Usable blue
    [231, 221, 124],   # Thick flake
    [157, 183, 131]    # Glue residue
]

# RGB
output_colors = [
    [0, 0, 0],        # Background → black
    [0, 255, 0],      # Thin flake → bright green
    [255, 255, 0],    # Usable blue → yellow
    [255, 128, 0],    # Thick flake → orange
    #[255, 0, 0]      # Glue residue → red
]

output_values = [
    0,   # Background
    #2,  # Thin flake
    1,   # Useabe blue
    0,   # Thick flake
    0,   # Glue residue
]

'''
result = find_nearest_colors(image, reference_colors, output_values=output_values)
window_sums = scan_windows(result)

ny, nx = window_sums.shape
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, window_sums, linewidth=0, antialiased=False)

ax.set_xlabel('Window X')
ax.set_ylabel('Window Y')
ax.set_zlabel('Sum of Pixels')
ax.set_title('Windowed Pixel Sum Surface')

plt.show()
'''
# Performance Benchmarking

'''
total_time = 0

totalProgress = 100
start_time = time.time()

Util.progress_bar(0, totalProgress, start_time)
for i in range(100):
    start = time.perf_counter_ns()
    result = find_nearest_colors(image, reference_colors, output_values=output_values)
    #window_sums = scan_windows(result)
    end = time.perf_counter_ns()
    total_time += end-start
    Util.progress_bar(i+1, totalProgress, start_time)
print(f"Average Time: {int(total_time/100)}ns")
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from sklearn.mixture import GaussianMixture
import scipy.ndimage as ndi
import hdbscan
# from scipy.spatial.distance import cdist

import Util
from data_visualizer import DataVisualizer
import init_pass_filter
import time

#----------------------------
# Function
#----------------------------

def gaussian_clustering(colors_bgr, n_components=10, random_state=42):

    gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=random_state)
    gmm.fit(colors_bgr)

    centers = gmm.means_

    return centers

def hdbscan_clustering_colors(colors_bgr, process_color_type="lab", min_cluster_size=20, min_samples=5):

    if (process_color_type == "lab"):
        colors = cv2.cvtColor(colors_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    else:
        colors = colors_bgr

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")

    start_time = time.time()
    labels = clusterer.fit_predict(colors)
    print(f"HDBSCAN finished in {time.time() - start_time:.2f}s")

    unique_labels = [l for l in np.unique(labels) if l != -1]

    centers = np.array([colors[labels == label].mean(axis=0) for label in unique_labels])

    if (process_color_type == "lab"):
        centers = centers.astype(np.uint8)
        centers = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)

    return centers # BGR

def hdbscan_clustering_image(image_bgr, process_color_type="lab", subsample=2, min_cluster_size=200, min_samples=10):

    start_time = time.time()

    if (process_color_type == "lab"):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    else:
        image = image_bgr

    if subsample > 1:
        image_sub = image[::subsample, ::subsample, :]
    else:
        image_sub = image

    h_sub, w_sub, _ = image_sub.shape

    colors = image_sub.reshape(-1, 3)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")

    labels = clusterer.fit_predict(colors)
    
    print(f"HDBSCAN finished in {time.time() - start_time:.2f}s")

    unique_labels = [l for l in np.unique(labels) if l != -1]
    centers = np.array([
        colors[labels == label].mean(axis=0)
        for label in unique_labels
    ])

    labels_2d = labels.reshape(h_sub, w_sub)

    plt.imshow(labels_2d)
    plt.axis("off")
    plt.show()

    if (process_color_type == "lab"):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        centers = centers.astype(np.uint8)
        centers = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)

    return centers # BGR

def bin_image(image_bgr, process_color_type, bin_size):

    if not (type(bin_size) == int or (type(bin_size) == list and len(bin_size) != 3)):
            ValueError("Incorrect bin size type. Must be integer or np.array of length 3")

    if (process_color_type == "bgr"):
        
        binned_colors = (image_bgr // bin_size) * bin_size
        
        return binned_colors
    elif (process_color_type == "lab"):
        
        image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

        L, A, B = cv2.split(image_lab)

        L = (L // bin_size[0]) * bin_size[0]
        A = ((A - 128) // bin_size[1]) * bin_size[1] + 128
        B = ((B - 128) // bin_size[2]) * bin_size[2] + 128

        binned_lab = cv2.merge([L, A, B])

        binned_lab = binned_lab.astype(np.uint8)

        binned_bgr = cv2.cvtColor(binned_lab, cv2.COLOR_LAB2BGR)

        return binned_bgr
    elif (process_color_type == "hsv"):
        
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        bin_size = np.array(bin_size).reshape(1, 1, 3)

        binned_hsv = (image_hsv // bin_size) * bin_size

        binned_hsv = binned_hsv.astype(np.uint8)

        binned_bgr = cv2.cvtColor(binned_hsv, cv2.COLOR_HSV2BGR)
        
        return binned_bgr

def remove_small_regions(mask, min_size=500, connectivity=1):

    labels, num = ndi.label(mask, structure=ndi.generate_binary_structure(2, connectivity))

    sizes = ndi.sum(np.ones_like(labels), labels, index=range(1, num + 1))

    cleaned_mask = np.zeros_like(mask, dtype=mask.dtype)

    for label_id, size in enumerate(sizes, start=1):
        if size >= min_size:
            cleaned_mask[labels == label_id] = 1

    return cleaned_mask

#----------------------------
# Load Image
#----------------------------

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#clusters = 10
#cluster_colors, segmented_image = gaussian_clustering(image)

original_unique_colors = np.unique(image_bgr.reshape(-1, 3), axis=0)
print("Original Unique Colors: " + str(len(original_unique_colors)))

#filtered_image = cv2.GaussianBlur(image_rgb, (5, 5), 1.4)
'''
filtered_image_bgr = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=150, sigmaSpace=75)

filtered_unique_colors = np.unique(filtered_image_bgr.reshape(-1, 3), axis=0) 
print(f"Filtered Unique Colors: {len(filtered_unique_colors)}")
print(f"Percent Decrease: {round(1-len(filtered_unique_colors)/len(original_unique_colors), 2) * 100}%")
'''

'''
binned_image_bgr = bin_image(image_bgr, process_color_type="hsv", bin_size=np.array([4, 8, 8]))
binned_unique_colors = np.unique(binned_image_bgr.reshape(-1, 3), axis=0) 

print(f"Binned Unique Colors: {len(binned_unique_colors)}")
print(f"Percent Decrease: {round(1-len(binned_unique_colors)/len(original_unique_colors), 2) * 100}%")

subsample = 1
image_bgr_sub = binned_image_bgr[::subsample, ::subsample, :]
colors_bgr = image_bgr_sub.reshape(-1, 3)
'''

#cluster_colors_bgr = binned_unique_colors

#DataVisualizer.display_colors_many(original_unique_colors, sorting="hsv")
#cluster_colors_bgr = hdbscan_clustering_colors(original_unique_colors, process_color_type="bgr", min_cluster_size=20, min_samples=5)

cluster_colors_bgr = hdbscan_clustering_image(image_bgr, process_color_type="bgr", subsample=5)

print(f"Cluster Colors: {len(cluster_colors_bgr)}")
cluster_colors_rgb = cv2.cvtColor(cluster_colors_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2RGB).reshape(-1, 3)
DataVisualizer.display_colors_many(cluster_colors_rgb, sorting="hsv")

cluster_colors_bgr = hdbscan_clustering_colors(cluster_colors_bgr, process_color_type="bgr", min_cluster_size=2, min_samples=1)

print(f"Cluster Colors: {len(cluster_colors_bgr)}")
cluster_colors_rgb = cv2.cvtColor(cluster_colors_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2RGB).reshape(-1, 3)
DataVisualizer.display_colors_many(cluster_colors_rgb, sorting="hsv")

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

reference_colors = []
for color in cluster_colors_rgb:
    if np.array_equal(color, np.array([160, 128, 96])):
        reference_colors.append([0,255,0])
    if np.array_equal(color, np.array([128, 128, 96])):
        reference_colors.append([0,128,0])
    else:
        reference_colors.append([0,0,0])
        
cleaned_image_rgb, color_counts = init_pass_filter.find_nearest_colors(image_rgb, cluster_colors_rgb, cluster_colors_rgb, use_lab=False)
plt.imshow(cleaned_image_rgb)
plt.title("Cleaned Image")
plt.axis("off")
plt.show()

colors_list = list(color_counts.keys())
counts_list = np.array(list(color_counts.values()))

top_idx = np.argsort(counts_list)[::-1][:5]
top_colors = [colors_list[i] for i in top_idx]
top_counts = [counts_list[i] for i in top_idx]

labels = [f"{count}" for count in top_counts] 

DataVisualizer.display_colors_small(top_colors, grid_shape=(1, 5), labels=labels)

DataVisualizer.surface_graphing(cv2.cvtColor(cleaned_image_rgb, cv2.COLOR_BGR2GRAY))

#InitialPassFilter.scan_windows()

'''
cluster_centers = hdbscan_clustering_image(image)
print(len(cluster_centers))
'''
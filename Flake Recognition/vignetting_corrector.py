import os

import cv2
import numpy as np
from tkinter import filedialog
from data_visualizer import DataVisualizer
import Util
import matplotlib.pyplot as plt

def vignetting_correction_direct_single_channel(image, flatfield, reference_point=None, reference_radius=5):

    if image.shape != flatfield.shape:
        raise ValueError("Image and flat-field must have the same dimensions")

    image_float = image.astype(np.float32)
    flatfield_float = flatfield.astype(np.float32)

    flatfield_gray = cv2.cvtColor(flatfield_float.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    epsilon = 1e-6

    corrected = image_float / (flatfield_gray[:, :, np.newaxis] + epsilon)

    if reference_point is not None:
        x_ref, y_ref = reference_point
        h, w = image.shape[:2]

        x1 = max(0, x_ref - reference_radius)
        x2 = min(w, x_ref + reference_radius + 1)
        y1 = max(0, y_ref - reference_radius)
        y2 = min(h, y_ref + reference_radius + 1)

        image_ref = image_float[y1:y2, x1:x2]
        corrected_ref = corrected[y1:y2, x1:x2]

        ref_orig = np.mean(image_ref)
        ref_corr = np.mean(corrected_ref)

        scale = ref_orig / (ref_corr + epsilon) # * 0.75

    else:
        mean_orig = np.mean(image_float)
        mean_corr = np.mean(corrected)

        scale = mean_orig / (mean_corr + epsilon)

    corrected *= scale

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    #corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    return corrected

def vignetting_correction_direct_multi_channel(image, flatfield, reference_point=None, reference_radius=5):

    if image.shape != flatfield.shape:
        raise ValueError("Image and flat-field must have the same dimensions")

    image_float = image.astype(np.float32)
    flatfield_float = flatfield.astype(np.float32)

    epsilon = 1e-6

    corrected = image_float / (flatfield_float + epsilon)

    if reference_point is not None:
        x_ref, y_ref = reference_point
        h, w = image.shape[:2]

        x1 = max(0, x_ref - reference_radius)
        x2 = min(w, x_ref + reference_radius + 1)
        y1 = max(0, y_ref - reference_radius)
        y2 = min(h, y_ref + reference_radius + 1)

        image_ref = image_float[y1:y2, x1:x2]
        corrected_ref = corrected[y1:y2, x1:x2]

        ref_orig = np.mean(image_ref, axis=(0, 1))
        ref_corr = np.mean(corrected_ref, axis=(0, 1))

        scale = ref_orig / (ref_corr + epsilon)
    else:
        mean_orig = np.mean(image, axis=(0, 1))
        mean_corr = np.mean(corrected, axis=(0, 1))

        scale = mean_orig / (mean_corr + epsilon)

    corrected *= scale

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    return corrected_rgb

def fit_polynomial_surface(flatfield_single_channel, degree=2):
    h, w = flatfield_single_channel.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    x = X.flatten()
    y = Y.flatten()
    z = flatfield_single_channel.flatten()

    terms = [np.ones_like(x)]
    if degree >= 1:
        terms += [x, y]
    if degree >= 2:
        terms += [x**2, x*y, y**2]
    if degree >= 3:
        terms += [x**3, x**2*y, x*y**2, y**3]

    A = np.column_stack(terms)
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    V = (A @ coeffs).reshape(h, w)
    return V

def vignetting_correction_poly_all_channels(image, flatfield, degree=2):

    if image.shape != flatfield.shape:
        raise ValueError("Image and flat-field must have the same dimensions and channels")

    flatfield_blur = cv2.GaussianBlur(flatfield, (0, 0), sigmaX=50, sigmaY=50)

    V0 = fit_polynomial_surface(flatfield_blur[:, :, 0], degree=degree)
    V1 = fit_polynomial_surface(flatfield_blur[:, :, 1], degree=degree)
    V2 = fit_polynomial_surface(flatfield_blur[:, :, 2], degree=degree)

    DataVisualizer.surface_graphing(V0, image[:, :, 0])
    DataVisualizer.surface_graphing(V1, image[:, :, 1])
    DataVisualizer.surface_graphing(V2, image[:, :, 2])

    epsilon = 1e-6

    V = np.stack((V0, V1, V2), axis=2)

    corrected = image / (V + epsilon)

    mean_orig = np.mean(image)
    corrected *= mean_orig / np.mean(corrected)

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    return corrected_rgb

def vignetting_correction_poly_max(image_path, flatfield_path, degree=2):
    image = cv2.imread(image_path).astype(np.float32)
    flatfield = cv2.imread(flatfield_path).astype(np.float32)

    if image.shape != flatfield.shape:
        raise ValueError("Image and flat-field must have the same dimensions")

    flatfield_blur = cv2.GaussianBlur(flatfield, (0, 0), sigmaX=50, sigmaY=50)

    flat_gray = cv2.cvtColor(flatfield_blur, cv2.COLOR_BGR2GRAY).astype(np.float32)

    V = fit_polynomial_surface(flat_gray, degree=degree)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    DataVisualizer.surface_graphing(V, image_gray)

    epsilon = 1e-6

    V_max = np.max(V)
    gain = V_max / (V + epsilon)

    corrected = image * gain[:, :, np.newaxis]

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":

    image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    flatfield_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])

    image = cv2.imread(image_path)
    flatfield= cv2.imread(flatfield_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    corrected_image = vignetting_correction_direct_single_channel(image, flatfield, reference_point=(image.shape[1]//2, image.shape[0]//2), reference_radius=10)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    #plt.imshow(corrected_image)
    plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    plt.title("Corrected")
    plt.axis("off")

    plt.tight_layout()

    corrected_image_gray = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2GRAY)
    DataVisualizer.surface_graphing(corrected_image_gray)
    Util.save_image(cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR), "corrected_image.png")
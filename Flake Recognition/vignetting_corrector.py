import cv2
import numpy as np
from tkinter import filedialog
from data_visualizer import DataVisualizer
import Util
import matplotlib.pyplot as plt

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
flatfield_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])

def vignetting_correction_direct_single_channel(image_path, flatfield_path):
    image = cv2.imread(image_path).astype(np.float32)
    flatfield = cv2.imread(flatfield_path).astype(np.float32)

    if image.shape != flatfield.shape:
        raise ValueError("Image and flat-field must have the same dimensions")
    
    flatfield_blur = cv2.GaussianBlur(flatfield, (5, 5), 1.4)

    epsilon = 1e-6
    
    flatfield_gray = cv2.cvtColor(flatfield_blur.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    corrected = image / (flatfield_gray[:, :, np.newaxis] + epsilon)

    corrected = corrected / np.mean(corrected) * np.mean(image)

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

def vignetting_correction_direct_multi_channel(image_path, flatfield_path):
    import cv2
    import numpy as np

    image = cv2.imread(image_path).astype(np.float32)
    flatfield = cv2.imread(flatfield_path).astype(np.float32)

    if image.shape != flatfield.shape:
        raise ValueError("Image and flat-field must have the same dimensions")

    flatfield_blur = cv2.GaussianBlur(flatfield, (5, 5), 1.4)

    epsilon = 1e-6

    corrected = image / (flatfield_blur + epsilon)

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

def vignetting_correction_poly_all_channels(image_path, flatfield_path, degree=2):

    image = cv2.imread(image_path).astype(np.float32)
    flatfield = cv2.imread(flatfield_path).astype(np.float32)

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

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
corrected_image = vignetting_correction_poly_all_channels(image_path, flatfield_path)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(corrected_image)
plt.title("Corrected")
plt.axis("off")

plt.tight_layout()

corrected_image_gray = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2GRAY)
DataVisualizer.surface_graphing(corrected_image_gray)
Util.save_image(corrected_image)
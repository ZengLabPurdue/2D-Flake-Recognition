import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')

home_dir = os.path.dirname(os.path.abspath(__file__))
flake_reg_path = Path(home_dir) / "Flake Recognition"
sys.path.insert(0, str(flake_reg_path))

import contour_finder
import flake_classifier

root = tk.Tk()
root.withdraw()

'''
model_path = filedialog.askopenfilename(
    title="Select trained model",
    filetypes=[("Keras Model", "*.keras *.h5")]
)
'''
model_path = "flake_classifier_tf.keras"

if not model_path:
    print("No model selected.")
    sys.exit()

model = load_model(model_path)
print("Loaded model:", model_path)

image_path = filedialog.askopenfilename(
    title="Select image",
    filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
)

if not image_path:
    print("No image selected.")
    sys.exit()

image = cv2.imread(image_path)
if image is None:
    print("Failed to load image.")
    sys.exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

start_time = time.time()

intensities = []
red_values = []
green_values = []
blue_values = []
red_green_values = []

labels = []

masked_image, contours = contour_finder.find_flakes(image, display=False)

valid_contours = []
for c in contours:
    try:
        c_fixed = np.array(c, dtype=np.int32).reshape(-1, 1, 2)
        valid_contours.append(c_fixed)
    except:
        continue

scanned_image = image.copy()

flakes = []

class_to_color = {
    0: (255, 255, 0),    # Bad flake - yellow
    1: (0, 255, 0),      # Good flake - green
    2: (200, 200, 200),  # Not a flake - light gray
    3: (0, 255, 200),    # Unclear flake - teal
}

for c in valid_contours:

    x, y, w, h = cv2.boundingRect(c)

    cx, cy = x + w / 2, y + h / 2

    scale = 1.2
    new_w, new_h = w * scale, h * scale

    new_x = int(cx - new_w / 2)
    new_y = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)

    h_img, w_img = masked_image.shape[:2]

    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_x2 = min(w_img, new_x2)
    new_y2 = min(h_img, new_y2)

    if new_x2 <= new_x or new_y2 <= new_y:
        continue

    contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [c], -1, 255, -1)

    crop = image[new_y:new_y2, new_x:new_x2]

    h_crop, w_crop = crop.shape[:2]

    if crop.size == 0:
        continue

    c_local = c.copy()
    c_local[:, 0, 0] -= new_x
    c_local[:, 0, 1] -= new_y

    mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
    cv2.drawContours(mask, [c_local], -1, 255, -1)

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    masked_pixels = gray_crop[mask == 255]

    if len(masked_pixels) == 0:
        continue

    avg_intensity = float(masked_pixels.mean())
    avg_red = float(crop[:, :, 0][mask == 255].mean())
    avg_green = float(crop[:, :, 1][mask == 255].mean())
    avg_blue = float(crop[:, :, 2][mask == 255].mean())
    avg_red_green = (avg_red + avg_green) / 2

    if (avg_red_green > 150):
        continue

    crop = cv2.resize(crop, (flake_classifier.IMG_SIZE, flake_classifier.IMG_SIZE))
    crop = crop.astype(np.float32) / 255.0

    input_img = np.expand_dims(crop, axis=0)

    pred = model.predict(input_img, verbose=0)
    class_id = int(np.argmax(pred))

    color = class_to_color.get(class_id, (255, 255, 255))

    cv2.rectangle(scanned_image,
                  (new_x, new_y),
                  (new_x2, new_y2),
                  color,
                  2)

    flakes.append((class_id, (new_x, new_y, new_w, new_h)))

    intensities.append(avg_intensity)
    red_values.append(avg_red)
    green_values.append(avg_green)
    blue_values.append(avg_blue)
    red_green_values.append(avg_red_green)
    labels.append(class_id)

print(f"Time: {time.time() - start_time:.2f}s")

plt.figure(figsize=(10, 8))
plt.imshow(scanned_image)
plt.title("Flake Detection + Classification")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 5))

names = ["Intensity", "Red Values", "Green Values", "Blue Values", "Red + Green Values"]

for data, name in zip([intensities, red_values, green_values, blue_values, red_green_values], names):
    for i in range(len(data)):
        class_id = labels[i]
        intensity = data[i]

        color = np.array(class_to_color[class_id]) / 255.0

        plt.scatter(i, intensity, color=color)

    plt.xlabel("Flake Index")
    plt.ylabel(f"Average {name}")
    plt.title(f"Flake {name} by Predicted Class")

    plt.show()
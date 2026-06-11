import cv2
import os
import sys
from collections import Counter
from tkinter import filedialog, messagebox

folder_path = filedialog.askdirectory()

if not folder_path:
    messagebox.showerror("Error", "No folder selected. Exiting.")
    sys.exit()

valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

file_list = [
    f for f in os.listdir(folder_path)
    if os.path.splitext(f)[1].lower() in valid_ext
]

if not file_list:
    print("No valid image files found.")
    sys.exit()

num_files = 0
scales = [1.25, 1.5, 1.75, 2.0]

for filename in file_list:
    full_path = os.path.join(folder_path, filename)
    name, ext = os.path.splitext(filename)

    img = cv2.imread(full_path)
    if img is None:
        print(f"Skipping unreadable file: {filename}")
        continue

    img_h = cv2.flip(img, 1)
    img_v = cv2.flip(img, 0)
    img_b = cv2.flip(img, -1)

    #cv2.imwrite(os.path.join(folder_path, f"{name}_FH{ext}"), img_h)
    #cv2.imwrite(os.path.join(folder_path, f"{name}_FV{ext}"), img_v)
    #cv2.imwrite(os.path.join(folder_path, f"{name}_FB{ext}"), img_b)

    h, w = img.shape[:2]

    new_w = int(w * 1.25)
    stretched_h = cv2.resize(img, (new_w, h))
    cv2.imwrite(os.path.join(folder_path, f"{name}_SX{ext}"), stretched_h)

    new_h = int(h * 1.25)
    stretched_v = cv2.resize(img, (w, new_h))
    cv2.imwrite(os.path.join(folder_path, f"{name}_SY{ext}"), stretched_v)

    num_files += 1

print(f"Total original files: {num_files}")
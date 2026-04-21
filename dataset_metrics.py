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

labels = []

for filename in file_list:
    name = os.path.splitext(filename)[0]

    parts = name.split("_")

    label = parts[0].lower() if len(parts) > 1 else "unknown"

    labels.append(label)

label_counts = Counter(labels)
total = len(labels)

print(f"Total files: {total}")

for label, count in label_counts.items():
    percentage = (count / total) * 100
    print(f"{label.upper()} FLAKES: {count} images ({percentage:.2f}%)")
import cv2
import numpy as np
import os
import random
import re
from tkinter import filedialog
from pathlib import Path
from collections import Counter

def format_data_from_sample(output_file="output.txt", value=1, fallback_background=(0, 0, 0)):

    image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    if not image_path:
        print("No image selected.")
        return

    filename = Path(image_path).name
    match = re.search(r"\((\d+),\s*(\d+),\s*(\d+)\)", filename)

    if match:
        background_color = np.array(list(map(int, match.groups())))
    else:
        background_color = np.array(fallback_background)

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    height, width, _ = image_rgb.shape

    with open(output_file, "w") as f:
        f.write(" FR  FG  FB  BR  BG  BB   V.\n")

        for y in range(height):
            for x in range(width):
                fr, fg, fb = image_rgb[y, x]

                if fr == 255 and fg == 255 and fb == 255:
                    continue

                br = np.clip(background_color[0] + random.randint(-4, 4), 0, 255)
                bg = np.clip(background_color[1] + random.randint(-4, 4), 0, 255)
                bb = np.clip(background_color[2] + random.randint(-4, 4), 0, 255)

                f.write(
                    f"{fr:3d}{fg:4d}{fb:4d}"
                    f"{br:4d}{bg:4d}{bb:4d}"
                    f"{value:5d}\n"
                )

    print(f"Pixel data written to {output_file}")

def combine_text_files_ignore_headers(output_file="combined_output.txt"):

    folder_path = filedialog.askdirectory()
    if not folder_path:
        print("No folder selected.")
        return

    folder = Path(folder_path)
    txt_files = sorted(folder.glob("*.txt"))

    if not txt_files:
        print("No text files found in the selected folder.")
        return

    line_count = 0
    first_file = True

    with open(folder / output_file, "w", encoding="utf-8") as outfile:
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as infile:
                for i, line in enumerate(infile):
                    if not first_file and i == 0:
                        continue
                    outfile.write(line)
                    line_count += 1

            first_file = False

    print(f"Combined {len(txt_files)} files into {folder / output_file}")
    print(f"Total number of lines written: {line_count}")

def sample_file(output_file="sampled_output.txt", step=10):

    file_path = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt")]
    )
    if not file_path:
        print("No file selected.")
        return

    file_path = Path(file_path)

    line_count = 0

    with open(file_path, "r", encoding="utf-8") as infile, \
         open(file_path.parent / output_file, "w", encoding="utf-8") as outfile:

        for i, line in enumerate(infile):
            if i == 0:
                outfile.write(line)
                line_count += 1
                continue

            if (i - 1) % step == 0:
                outfile.write(line)
                line_count += 1

    print(f"Sampled file written to: {file_path.parent / output_file}")
    print(f"Total number of lines written: {line_count}")

def analyze_v_makeup():

    file_path = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt")]
    )
    if not file_path:
        print("No file selected.")
        return

    v_counts = Counter()
    total_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            v = int(parts[-1])
            v_counts[v] += 1
            total_count += 1

    print(f"Total data points: {total_count}")

    for v_value, count in sorted(v_counts.items()):
        percentage = (count / total_count) * 100
        print(f"V = {v_value}: {count} ({percentage:.2f}%)")

#format_data_from_sample(output_file="glue_2.txt", value=1)
#combine_text_files_ignore_headers(output_file="all_data.txt")
#sample_file(output_file="combined_sampled_background_data.txt")
analyze_v_makeup()

# Background = 0
# Glue = 1
# Thin Flake = 2
# Med Flake = 3
# Thick Flake = 4
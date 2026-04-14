import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import re
from pathlib import Path

import time
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import math

from tensorflow.keras.models import load_model
from tkinter import messagebox

import cv2
import csv

home_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = home_dir.parent
flake_reg_path = parent_dir / "Flake Recognition"
sys.path.insert(0, str(flake_reg_path))

import flake_finder

folder_path = filedialog.askdirectory()

if not folder_path:
    messagebox.showerror("Error", "No folder selected. Exiting.")
    sys.exit()

valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

file_list = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith(valid_ext)
]

print(len(file_list), "files found in the folder.")
num_contours = 0

for file_path in file_list:
    image_bgr = cv2.imread(file_path)
    num_contours += len(flake_finder.find_flakes(image_bgr, display=False)[1])

print(f"Total number of contours found: {num_contours}")
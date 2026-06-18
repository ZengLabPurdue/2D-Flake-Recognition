from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.insert(0, project_root)

from Scanning.contour_extractor import get_contour

IMAGE_SAVE_DIR = Path(r"C:\Users\Zengl\Box\Zenglab_fabrication\DATABASE\Images")
JSON_SAVE_DIR = Path(r"C:\Users\Zengl\Box\Zenglab_fabrication\DATABASE\to_read")

def save_flake(
    flake_id : str,
    image_20X: np.ndarray,
    image_100X: np.ndarray,
    metadata,
    image_format="png",
):
    IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if flake_id is None:
        flake_id = datetime.now().strftime("flake_%Y%m%d_%H%M%S_%f")

    image_20X_path = IMAGE_SAVE_DIR / f"{flake_id}_20X.{image_format}"
    image_100X_path = IMAGE_SAVE_DIR / f"{flake_id}_100X.{image_format}"
    json_path = JSON_SAVE_DIR / f"{flake_id}.json"

    cv2.imwrite(str(image_20X_path), image_20X)
    cv2.imwrite(str(image_100X_path), image_100X)

    data = {
        "flake_id": flake_id,
        "date_found": datetime.now().strftime("%m-%d-%Y"),
        "images": {
            "20X": str(image_20X_path),
            "100X": str(image_100X_path),
        },
        "metadata": metadata or {},
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Flake saved successfully!")
    print(f"  20X image: {image_20X_path}")
    print(f"  100X image: {image_100X_path}")
    print(f"  JSON: {json_path}")

    return image_20X_path, image_100X_path, json_path

if __name__ == "__main__":
    from tkinter import filedialog

    path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    img1 = cv2.imread(path, cv2.IMREAD_COLOR)

    path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    img2 = cv2.imread(path, cv2.IMREAD_COLOR)

    points = get_contour(img2)

    save_flake(
        flake_id="Test_00001_C",
        image_20X=img1,
        image_100X=img2,
        metadata={
            "substrate": "285nm",
            "material" : "Graphene",
            "classification": "thin",
            "contour": points
        }
    )
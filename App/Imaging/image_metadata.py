from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


METADATA_PREFIX = "flake_search_"


def save_png(path, image_bgr, metadata=None):
    path = Path(path)
    image_bgr = np.asarray(image_bgr)
    if image_bgr.ndim == 2:
        image = Image.fromarray(image_bgr)
    elif image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("PNG images must be grayscale or three-channel BGR images.")

    png_info = PngInfo()
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        png_info.add_text(f"{METADATA_PREFIX}{key}", str(value))
    image.save(path, format="PNG", pnginfo=png_info)


def read_png_metadata(path):
    path = Path(path)
    if path.suffix.lower() != ".png" or not path.is_file():
        return {}
    try:
        with Image.open(path) as image:
            return {
                key[len(METADATA_PREFIX):]: str(value)
                for key, value in image.info.items()
                if key.startswith(METADATA_PREFIX)
            }
    except (OSError, ValueError):
        return {}


def is_vignette_corrected(path):
    value = read_png_metadata(path).get("vignette_applied", "false")
    return value.strip().casefold() == "true"

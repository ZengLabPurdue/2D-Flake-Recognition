import os
from pathlib import Path

import cv2

HOME_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

FLATFIELD_IMG = cv2.imread(str(HOME_DIR / "Flatfields" / "flatfield_2x_med_smoothed.png"))

DEFAULT_EXPOSURE = 60

CROP_RATIO = {
    "2X": {
        "x": 0.7,
        "y": 0.7,
    },
    "10X": {
        "x": 0.7,
        "y": 0.7,
    },
    "20X": {
        "x": 1.0,
        "y": 1.0,
    },
    "100X": {
        "x": 0.7,
        "y": 0.7,
    },
}

RELATIVE_Z = {
    "2X": 0,
    "10X": 1250,
    "20X": 4300,
    "100X": 4300
}

CROP_RATIO = {
    "2X": {
        "x": 0.7,
        "y": 0.7,
    },
    "10X": {
        "x": 0.9,
        "y": 0.9,
    },
    "20X": {
        "x": 1.0,
        "y": 1.0,
    },
    "100X": {
        "x": 1.0,
        "y": 1.0,
    },
}

RESOLUTION = {
    "HIGH": 0,
    "MED" : 1, 
    "LOW": 2,
}

RESOLUTION_DIM = {
    "LOW": {
        "x": 1824,
        "y": 1216,
    },
    "MED": {
        "x": 2736,
        "y": 1824,
    },
    "HIGH": {
        "x": 5440,
        "y": 3648,
    }
}

PIXEL_SIZE = {
    "2X": {
        "LOW": 1.162453,
        "MED": 0.768964,
        "HIGH": 0.385936,
    },
    "10X": {
        "LOW": 0.230062,
        "MED": 0.152501,
        "HIGH": 0.076746,
    },
    "20X": {
        "LOW": 0.117185,
        "MED": 0.078098,
        "HIGH": 0.039048,
    },
    "100X": {
        "LOW": 0.0231928,
        "MED": 0.0154422,
        "HIGH": 0.00773423,
    },
}


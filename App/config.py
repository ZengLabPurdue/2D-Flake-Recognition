import os
from pathlib import Path

import cv2

HOME_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

PROCESS_FRAME_RATE = 30

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
        "x": 0.8,
        "y": 0.8,
    },
    "100X": {
        "x": 0.7,
        "y": 0.7,
    },
}

RELATIVE_Z = {
    "2X": 0,
    "10X": 1000,
    "20X": 1000,
    "100X": 1000,
}

RELATIVE_XY = {
    "2X": {
        "X": 0,
        "Y": 0,
    },
    "10X": {
        "X": 350,
        "Y": 75,
    },
    "20X": {
        "X": 400,
        "Y": 60,
    },
    "100X": {
        "X": 580,
        "Y": 80,
    },
}

RESOLUTION = {
    "HIGH": 0,
    "MED" : 1, 
    "LOW": 2,
}

RESOLUTION_DISPLAY = {
    "HIGH": "5440 × 3648",
    "MED": "2736 × 1824",
    "LOW": "1824 × 1216",
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

# um / pixel

PIXEL_SIZE = {
    "2X": {
        "LOW": 5.78001272,
        "MED": 3.84585801,
        "HIGH": 1.91938580,
    },
    "10X": {
        "LOW": 1.15068178,
        "MED": 0.76651847,
        "HIGH": 0.38282934,
    },
    "20X": {
        "LOW": 0.58614242,
        "MED": 0.39061127,
        "HIGH": 0.19530068,
    },
    "100X": {
        "LOW": 0.11634198,
        "MED": 0.07763704,
        "HIGH": 0.03885661,
    },
}


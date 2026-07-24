import cv2
import numpy as np


def create_vignette_gain(flatfield, reference_point=None, reference_radius=5):
    if flatfield is None or flatfield.ndim != 3 or flatfield.shape[2] != 3:
        raise ValueError("The vignette filter must be a three-channel image")

    if reference_point is None:
        reference_values = np.mean(
            flatfield,
            axis=(0, 1),
            dtype=np.float64,
        )
    else:
        x_ref, y_ref = reference_point
        height, width = flatfield.shape[:2]
        if not (0 <= x_ref < width and 0 <= y_ref < height):
            raise ValueError(
                "The vignette reference point must be inside the image"
            )
        x1 = max(0, x_ref - reference_radius)
        x2 = min(width, x_ref + reference_radius + 1)
        y1 = max(0, y_ref - reference_radius)
        y2 = min(height, y_ref + reference_radius + 1)
        reference_values = np.mean(
            flatfield[y1:y2, x1:x2],
            axis=(0, 1),
            dtype=np.float64,
        )

    reference_values = np.maximum(
        reference_values,
        1.0,
    ).astype(np.float32)
    gain = flatfield.astype(np.float32)
    np.maximum(gain, 1.0, out=gain)
    np.divide(
        reference_values.reshape(1, 1, 3),
        gain,
        out=gain,
    )
    return gain


def apply_vignette_gain(image, gain):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The camera image must be a three-channel image")
    scalar_gain = gain.ndim == 2 and gain.shape == image.shape[:2]
    color_gain = gain.ndim == 3 and gain.shape == image.shape
    if not scalar_gain and not color_gain:
        raise ValueError("Image and vignette filter must have the same dimensions")

    if color_gain:
        return cv2.multiply(
            image,
            gain,
            dtype=cv2.CV_8U,
        )

    corrected = np.empty_like(image)
    for channel in range(3):
        corrected[:, :, channel] = cv2.multiply(
            image[:, :, channel],
            gain,
            dtype=cv2.CV_8U,
        )
    return corrected


def correct_vignetting_effect(
    image,
    flatfield,
    reference_point=None,
    reference_radius=5,
):
    if image.shape != flatfield.shape:
        raise ValueError("Image and flat-field must have the same dimensions")
    gain = create_vignette_gain(
        flatfield,
        reference_point=reference_point,
        reference_radius=reference_radius,
    )
    return apply_vignette_gain(image, gain)

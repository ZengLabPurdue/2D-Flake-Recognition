import cv2
import numpy as np


def create_vignette_gain(flatfield, reference_point=None, reference_radius=5):
    if flatfield is None or flatfield.ndim != 3 or flatfield.shape[2] != 3:
        raise ValueError("The vignette filter must be a three-channel image")

    flatfield_gray = cv2.cvtColor(flatfield, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if reference_point is None:
        reference_value = float(np.mean(flatfield_gray))
    else:
        x_ref, y_ref = reference_point
        height, width = flatfield_gray.shape
        x1 = max(0, x_ref - reference_radius)
        x2 = min(width, x_ref + reference_radius + 1)
        y1 = max(0, y_ref - reference_radius)
        y2 = min(height, y_ref + reference_radius + 1)
        reference_value = float(np.mean(flatfield_gray[y1:y2, x1:x2]))

    np.maximum(flatfield_gray, 1.0, out=flatfield_gray)
    return reference_value / flatfield_gray


def apply_vignette_gain(image, gain):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The camera image must be a three-channel image")
    if gain.shape != image.shape[:2]:
        raise ValueError("Image and vignette filter must have the same dimensions")

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

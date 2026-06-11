import cv2
import numpy as np

def correct_vignetting_effect(image, flatfield, reference_point=None, reference_radius=5):

    if image.shape != flatfield.shape:
        raise ValueError("Image and flat-field must have the same dimensions")

    image_float = image.astype(np.float32)
    flatfield_float = flatfield.astype(np.float32)

    flatfield_gray = cv2.cvtColor(flatfield_float.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    epsilon = 1e-6

    corrected = image_float / (flatfield_gray[:, :, np.newaxis] + epsilon)

    if reference_point is not None:
        x_ref, y_ref = reference_point
        h, w = image.shape[:2]

        x1 = max(0, x_ref - reference_radius)
        x2 = min(w, x_ref + reference_radius + 1)
        y1 = max(0, y_ref - reference_radius)
        y2 = min(h, y_ref + reference_radius + 1)

        image_ref = image_float[y1:y2, x1:x2]
        corrected_ref = corrected[y1:y2, x1:x2]

        ref_orig = np.mean(image_ref)
        ref_corr = np.mean(corrected_ref)

        scale = ref_orig / (ref_corr + epsilon) # * 0.75

    else:
        mean_orig = np.mean(image_float)
        mean_corr = np.mean(corrected)

        scale = mean_orig / (mean_corr + epsilon)

    corrected *= scale

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    #corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    return corrected
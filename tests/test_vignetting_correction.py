from pathlib import Path
import sys
import unittest

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from Imaging import vignetting_corrector  # noqa: E402
from Imaging.frame_processing import FrameProcessor  # noqa: E402


class _FakeApp:
    @staticmethod
    def get_magnification():
        return "2X"


class VignettingCorrectionTests(unittest.TestCase):
    def test_per_channel_gain_removes_spatial_color_cast(self):
        flatfield = np.array(
            [
                [[40, 90, 140], [70, 100, 130], [60, 60, 80]],
                [[50, 80, 120], [80, 120, 160], [70, 90, 100]],
                [[30, 70, 100], [60, 100, 150], [50, 80, 90]],
            ],
            dtype=np.uint8,
        )

        gain = vignetting_corrector.create_vignette_gain(
            flatfield,
            reference_point=(1, 1),
            reference_radius=0,
        )
        corrected = vignetting_corrector.apply_vignette_gain(
            flatfield,
            gain,
        )

        expected = np.broadcast_to(
            flatfield[1, 1],
            flatfield.shape,
        )
        self.assertEqual(gain.shape, flatfield.shape)
        np.testing.assert_allclose(
            corrected,
            expected,
            atol=1,
        )

    def test_uniform_flatfield_preserves_existing_color_balance(self):
        flatfield = np.empty((4, 5, 3), dtype=np.uint8)
        flatfield[:] = (80, 120, 160)
        image = np.array(
            [
                [[10, 30, 90], [100, 80, 20]],
                [[25, 50, 75], [200, 150, 100]],
            ],
            dtype=np.uint8,
        )
        image = np.tile(image, (2, 3, 1))[:4, :5]

        gain = vignetting_corrector.create_vignette_gain(
            flatfield,
            reference_point=(2, 2),
        )
        corrected = vignetting_corrector.apply_vignette_gain(
            image,
            gain,
        )

        np.testing.assert_array_equal(corrected, image)

    def test_legacy_scalar_gain_remains_supported(self):
        image = np.array(
            [
                [[10, 20, 30], [20, 40, 60]],
                [[30, 60, 90], [40, 80, 120]],
            ],
            dtype=np.uint8,
        )
        gain = np.array(
            [
                [2.0, 0.5],
                [1.0, 1.5],
            ],
            dtype=np.float32,
        )

        corrected = vignetting_corrector.apply_vignette_gain(
            image,
            gain,
        )

        expected = np.clip(
            np.rint(image.astype(np.float32) * gain[:, :, None]),
            0,
            255,
        ).astype(np.uint8)
        np.testing.assert_array_equal(corrected, expected)

    def test_frame_processor_crops_a_color_gain_to_a_cropped_frame(self):
        processor = FrameProcessor.__new__(FrameProcessor)
        processor.app = _FakeApp()
        processor.vignette_gain = np.ones(
            (10, 10, 3),
            dtype=np.float32,
        )
        processor.load_vignette_filter = lambda _shape=None: None
        image = np.full((6, 6, 3), (20, 40, 80), dtype=np.uint8)

        corrected = processor.apply_vignette_filter(image)

        np.testing.assert_array_equal(corrected, image)


if __name__ == "__main__":
    unittest.main()

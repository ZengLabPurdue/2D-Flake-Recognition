from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from Imaging.focus import FocusController  # noqa: E402


class _FakeApp:
    def __init__(self, magnification="10X"):
        self.magnification = magnification

    def get_magnification(self):
        return self.magnification


class _FakeStage:
    POSITION_TOLERANCE_UM = 1.0

    def __init__(self, z=0.0):
        self.z = float(z)
        self.z_velocity = 500
        self.z_acceleration = 10000
        self.moves = []
        self.velocity_changes = []
        self.acceleration_changes = []
        self.stop_calls = 0

    def get_z_velocity(self):
        return self.z_velocity

    def get_z_acceleration(self):
        return self.z_acceleration

    def set_z_velocity(self, value):
        self.z_velocity = int(value)
        self.velocity_changes.append(self.z_velocity)
        return self.z_velocity

    def set_z_acceleration(self, value):
        self.z_acceleration = int(value)
        self.acceleration_changes.append(self.z_acceleration)

    def get_z_position(self, strict=False):
        return self.z

    def wait_until_not_busy(self, **_kwargs):
        return 0.0

    def move_to_z(self, z, wait=True, **_kwargs):
        self.z = round(float(z), 1)
        self.moves.append((self.z, wait))
        return True

    def stop_z(self):
        self.stop_calls += 1


class _FakeFrameProcessor:
    def __init__(self, stage):
        self.stage = stage
        self.frame_buffer = []
        self._frame_condition = threading.Condition()
        self.capture_calls = []

    def capture_frame(self, **kwargs):
        self.capture_calls.append(kwargs)
        return np.full((20, 20, 3), self.stage.z, dtype=np.float32)


class AutofocusTests(unittest.TestCase):
    @staticmethod
    def _controller(magnification="10X", z=0.0):
        stage = _FakeStage(z)
        frame_processor = _FakeFrameProcessor(stage)
        controller = FocusController(
            app=_FakeApp(magnification),
            stage=stage,
            frame_processor=frame_processor,
        )
        return controller, stage, frame_processor

    def test_higher_magnifications_use_finer_target_precision(self):
        precisions = []
        for magnification in ("2X", "10X", "20X", "100X"):
            controller, _stage, _processor = self._controller(magnification)
            _name, profile = controller._focus_profile(100)
            precisions.append(profile["target_precision"])

        self.assertEqual(precisions, [5.0, 1.0, 0.5, 0.1])

    def test_coarse_sweep_stops_only_after_a_prominent_peak_and_drop(self):
        rising = [
            (float(z), score)
            for z, score in enumerate((10, 15, 30, 60, 100, 140))
        ]
        passed = rising + [
            (6.0, 100),
            (7.0, 70),
            (8.0, 50),
        ]

        self.assertFalse(
            FocusController._coarse_peak_has_passed(
                rising,
                peak_found_threshold=20,
                coarse_drop_ratio=0.1,
                minimum_travel=2,
            )
        )
        self.assertTrue(
            FocusController._coarse_peak_has_passed(
                passed,
                peak_found_threshold=20,
                coarse_drop_ratio=0.1,
                minimum_travel=2,
            )
        )

    def test_golden_section_refinement_converges_with_few_settled_samples(self):
        controller, stage, _processor = self._controller("100X")
        peak_z = 3.7
        coarse_samples = [
            (z, 2000.0 - 40.0 * (z - peak_z) ** 2)
            for z in np.arange(-20.0, 20.1, 0.5)
        ]
        _name, profile = controller._focus_profile(20)
        measured_positions = []

        def measure(target_z, frame_count=1):
            actual_z = round(float(target_z), 1)
            stage.z = actual_z
            measured_positions.append((actual_z, frame_count))
            score = 2000.0 - 40.0 * (actual_z - peak_z) ** 2
            return score, actual_z

        with patch.object(
            controller,
            "_measure_focus_at",
            side_effect=measure,
        ):
            best_z, _best_score = controller._refine_focus_peak(
                coarse_samples,
                -20.0,
                20.0,
                profile,
            )

        self.assertLessEqual(abs(best_z - peak_z), 0.1)
        self.assertLessEqual(len(measured_positions), 18)
        self.assertTrue(
            all(
                frame_count == profile["fine_frame_count"]
                for _z, frame_count in measured_positions
            )
        )

    def test_local_probe_search_brackets_peak_without_full_range_sweep(self):
        controller, stage, _processor = self._controller("10X")
        peak_z = 8.0
        _name, profile = controller._focus_profile(20)
        measured_positions = []

        def measure(target_z, frame_count=1):
            actual_z = round(float(target_z), 1)
            stage.z = actual_z
            measured_positions.append(actual_z)
            score = 1000.0 - 10.0 * (actual_z - peak_z) ** 2
            return score, actual_z

        with patch.object(
            controller,
            "_measure_focus_at",
            side_effect=measure,
        ):
            result = controller._adaptive_focus_bracket(
                current_z=0.0,
                z_start=-20.0,
                z_end=20.0,
                peak_found_threshold=20,
                profile=profile,
            )

        self.assertIsNotNone(result)
        _samples, bounds = result
        self.assertLessEqual(bounds[0], peak_z)
        self.assertGreaterEqual(bounds[1], peak_z)
        self.assertLessEqual(len(measured_positions), 5)

    def test_auto_focus_uses_local_bracket_and_restores_settings(self):
        controller, stage, _processor = self._controller("20X", z=50.0)
        peak_z = 57.3
        settled_samples = [
            (50.0, 1000.0 - 8.0 * (50.0 - peak_z) ** 2),
            (57.0, 1000.0 - 8.0 * (57.0 - peak_z) ** 2),
            (65.0, 1000.0 - 8.0 * (65.0 - peak_z) ** 2),
        ]

        def measure(target_z, frame_count=1):
            actual_z = round(float(target_z), 1)
            stage.z = actual_z
            score = 1000.0 - 8.0 * (actual_z - peak_z) ** 2
            return score, actual_z

        with (
            patch.object(
                controller,
                "_adaptive_focus_bracket",
                return_value=(settled_samples, (50.0, 65.0)),
            ) as local_search,
            patch.object(
                controller,
                "_coarse_focus_sweep",
            ) as coarse_sweep,
            patch.object(
                controller,
                "_measure_focus_at",
                side_effect=measure,
            ),
        ):
            best_z = controller.auto_focus(
                focus_range=20,
                z_velo=25,
                z_accel=9000,
                peak_found_threshold=20,
            )

        self.assertLessEqual(abs(best_z - peak_z), 0.5)
        self.assertEqual(stage.z, best_z)
        self.assertEqual(stage.z_velocity, 500)
        self.assertEqual(stage.z_acceleration, 10000)
        local_search.assert_called_once()
        coarse_sweep.assert_not_called()

    def test_cancel_restores_controller_settings(self):
        controller, stage, _processor = self._controller("10X", z=25.0)
        controller.stop_focus_event.set()

        result = controller.auto_focus(
            focus_range=20,
            z_velo=50,
            z_accel=9000,
            peak_found_threshold=20,
        )

        self.assertEqual(result, 25.0)
        self.assertEqual(stage.z_velocity, 500)
        self.assertEqual(stage.z_acceleration, 10000)
        self.assertEqual(stage.stop_calls, 1)

    def test_settled_measurement_uses_fresh_averaged_capture(self):
        controller, stage, frame_processor = self._controller("10X")
        controller.find_sharpness = (
            lambda frame, **_kwargs: (float(np.mean(frame)), 0.0)
        )

        score, actual_z = controller._measure_focus_at(
            12.4,
            frame_count=2,
        )

        self.assertEqual(actual_z, 12.4)
        self.assertAlmostEqual(score, 12.4, places=4)
        self.assertEqual(stage.moves[-1], (12.4, True))
        self.assertEqual(
            frame_processor.capture_calls[-1]["num_images"],
            2,
        )
        self.assertEqual(
            frame_processor.capture_calls[-1]["timeout_seconds"],
            controller.FOCUS_FRAME_TIMEOUT_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()

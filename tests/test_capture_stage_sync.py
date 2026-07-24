from collections import deque
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from Hardware.stage_api import stage as Stage  # noqa: E402
from Imaging import amcam  # noqa: E402
from Imaging.frame_processing import FrameProcessor  # noqa: E402


class _FakeApp:
    @staticmethod
    def get_magnification():
        return "2X"


class _FakeStage:
    POSITION_TOLERANCE_UM = 1.0

    def __init__(self):
        self.motion_sequence = 0

    @staticmethod
    def is_busy():
        return False

    @staticmethod
    def wait_until_not_busy(cancel_check=None):
        if cancel_check is not None:
            cancel_check()


class _FakeCamera:
    def __init__(self):
        self.options = []
        self.flushed = threading.Event()

    def put_Option(self, option, value):
        self.options.append((option, value))
        self.flushed.set()


class CaptureAndStageSynchronizationTests(unittest.TestCase):
    def test_prior_busy_bitmask_accepts_xy_and_z_motion_values(self):
        candidate = Stage.__new__(Stage)
        responses = iter(
            ("0", "3", "4", "not-a-status", "not-a-status")
        )
        candidate.cmd = lambda _command: (0, next(responses))

        self.assertFalse(candidate.is_xy_busy(strict=True))
        self.assertTrue(candidate.is_xy_busy(strict=True))
        self.assertTrue(candidate.is_z_busy(strict=True))
        with self.assertRaisesRegex(
            RuntimeError,
            "Could not read controller busy state",
        ):
            candidate.is_z_busy(strict=True)
        self.assertTrue(candidate.is_z_busy())

    def test_xy_wait_ignores_initial_idle_until_target_is_stable(self):
        candidate = Stage.__new__(Stage)
        candidate.last_confirmed_xy = None
        candidate.MOTION_POLL_INTERVAL_SECONDS = 0
        candidate.POSITION_TOLERANCE_UM = 1.0
        candidate.POSITION_STABLE_SAMPLES = 3
        busy_values = iter((False, True, False, False, False))
        position_values = iter(
            ((0, 0), (5, 0), (10, 0), (10, 0), (10, 0))
        )
        candidate.is_xy_busy = (
            lambda strict=False: next(busy_values)
        )
        candidate.get_xy_position = (
            lambda strict=False: next(position_values)
        )
        candidate.stop_all = lambda: None

        position = candidate.wait_for_xy_target(
            10,
            0,
            timeout=1,
        )

        self.assertEqual(position, (10, 0))
        self.assertEqual(candidate.last_confirmed_xy, (10, 0))

    def test_strict_position_read_does_not_reuse_stale_coordinates(self):
        candidate = Stage.__new__(Stage)
        candidate.x = 12
        candidate.y = 34
        candidate.cmd = lambda _command: (0, "not-a-position")

        with self.assertRaisesRegex(RuntimeError, "Could not parse XY"):
            candidate.get_xy_position(strict=True)

        self.assertEqual(candidate.get_xy_position(), (12, 34))

    def test_raw_capture_discards_first_frame_and_uses_cv_accumulate(self):
        processor = self._processor()
        producer = self._start_producer(
            processor,
            [
                (200, (0, 0)),
                (10, (0, 0)),
                (20, (0, 0)),
            ],
        )

        with patch(
            "Imaging.frame_processing.cv2.accumulate",
            wraps=cv2.accumulate,
        ) as accumulate:
            captured = processor.capture_frame_raw(
                num_images=2,
                timeout_seconds=2,
            )

        producer.join(timeout=1)
        self.assertFalse(producer.is_alive())
        self.assertEqual(accumulate.call_count, 2)
        self.assertTrue(np.all(captured == 15))
        self.assertIn(
            (amcam.AMCAM_OPTION_FLUSH, 3),
            processor.hcam.options,
        )

    def test_single_cropped_capture_bypasses_accumulator(self):
        processor = self._processor()
        producer = self._start_producer(
            processor,
            [
                (200, (0, 0)),
                (33, (0, 0)),
            ],
        )

        with patch(
            "Imaging.frame_processing.cv2.accumulate",
            wraps=cv2.accumulate,
        ) as accumulate:
            captured = processor.capture_frame(
                num_images=1,
                timeout_seconds=2,
            )

        producer.join(timeout=1)
        self.assertEqual(accumulate.call_count, 0)
        self.assertEqual(captured.shape, (6, 6, 3))
        self.assertTrue(np.all(captured == 33))

    def test_capture_after_motion_requires_the_verified_xy_position(self):
        processor = self._processor()
        processor.arm_capture_after_motion((50, 75))
        producer = self._start_producer(
            processor,
            [
                (200, (50, 75)),
                (20, (49, 70)),
                (40, (50, 75)),
            ],
        )

        captured = processor.capture_frame_raw(
            num_images=1,
            timeout_seconds=2,
        )

        producer.join(timeout=1)
        self.assertFalse(producer.is_alive())
        self.assertTrue(np.all(captured == 40))

    def test_motion_invalidates_an_armed_position_barrier(self):
        processor = self._processor()
        processor.arm_capture_after_motion((50, 75))
        processor.stage.motion_sequence += 1

        with patch.object(
            processor,
            "_flush_camera_queue",
            return_value=123,
        ) as flush:
            capture_state = processor._begin_capture_sequence()

        flush.assert_called_once_with(discard_frames=1)
        self.assertEqual(capture_state, (123, None, 1))

    def test_motion_during_average_discards_frames_from_both_positions(self):
        processor = self._processor()

        def append_frame(value, stage_busy=False):
            with processor._frame_condition:
                processor.frame_buffer.append({
                    "frame": np.full(
                        (10, 10, 3),
                        value,
                        dtype=np.uint8,
                    ),
                    "frame_id": processor.frame_id,
                    "stage_busy": stage_busy,
                    "x": 0,
                    "y": 0,
                })
                processor.frame_id += 1
                processor._frame_condition.notify_all()

        def produce():
            if not processor.hcam.flushed.wait(timeout=1):
                return
            append_frame(200)
            time.sleep(0.02)
            append_frame(10)
            time.sleep(0.05)

            processor.stage.motion_sequence += 1
            append_frame(250, stage_busy=True)
            deadline = time.monotonic() + 1
            while (
                len(processor.hcam.options) < 2
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)

            append_frame(201)
            time.sleep(0.02)
            append_frame(30)
            time.sleep(0.02)
            append_frame(50)

        producer = threading.Thread(target=produce)
        producer.start()
        captured = processor.capture_frame_raw(
            num_images=2,
            timeout_seconds=2,
        )

        producer.join(timeout=1)
        self.assertFalse(producer.is_alive())
        self.assertEqual(len(processor.hcam.options), 2)
        self.assertTrue(np.all(captured == 40))

    @staticmethod
    def _processor():
        processor = FrameProcessor.__new__(FrameProcessor)
        processor.app = _FakeApp()
        processor.stage = _FakeStage()
        processor.hcam = _FakeCamera()
        processor.frame_buffer = deque(maxlen=5)
        processor.frame_id = 0
        processor.last_used_capture_frame_id = -1
        processor._frame_condition = threading.Condition()
        processor._capture_barrier = None
        processor._live_capture_min_frame_id = None
        processor._live_capture_expected_xy = None
        processor._camera_flush_warning_shown = False
        return processor

    @staticmethod
    def _start_producer(processor, frames):
        def produce():
            if not processor.hcam.flushed.wait(timeout=1):
                return
            for value, position in frames:
                time.sleep(0.02)
                with processor._frame_condition:
                    processor.frame_buffer.append({
                        "frame": np.full(
                            (10, 10, 3),
                            value,
                            dtype=np.uint8,
                        ),
                        "frame_id": processor.frame_id,
                        "stage_busy": False,
                        "x": position[0],
                        "y": position[1],
                    })
                    processor.frame_id += 1
                    processor._frame_condition.notify_all()

        thread = threading.Thread(target=produce)
        thread.start()
        return thread


if __name__ == "__main__":
    unittest.main()

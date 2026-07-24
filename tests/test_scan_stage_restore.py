from pathlib import Path
import sys
import unittest


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from Scanning.scan_manager import ScanManager  # noqa: E402


class _FakeApp:
    focus_controller = None

    @staticmethod
    def get_resolution():
        return "HIGH"


class _FakeStage:
    def __init__(self, position=(120, 340)):
        self.position = position
        self.last_confirmed_xy = None
        self.position_reads = []
        self.moves = []
        self.waits = 0

    def get_xy_position(self, strict=False):
        self.position_reads.append(strict)
        return self.position

    def wait_until_not_busy(self, **_kwargs):
        self.waits += 1

    def move_to_xy(self, x, y, wait=True, **_kwargs):
        self.moves.append((x, y, wait))
        self.position = (x, y)
        self.last_confirmed_xy = (x, y)
        return True


class _FakeFrameProcessor:
    def __init__(self):
        self.motion_targets = []

    def arm_capture_after_motion(self, expected_xy):
        self.motion_targets.append(expected_xy)


class ScanStageRestoreTests(unittest.TestCase):
    @staticmethod
    def _manager():
        stage = _FakeStage()
        frame_processor = _FakeFrameProcessor()
        statuses = []
        manager = ScanManager(
            root=None,
            app=_FakeApp(),
            stage=stage,
            turret_controller=None,
            camera=None,
            frame_processor=frame_processor,
            mapper=None,
            update_scan_status=lambda **status: statuses.append(status),
        )
        return manager, stage, frame_processor, statuses

    def test_requested_scan_types_return_to_their_pre_scan_xy(self):
        scan_types = (
            "2x Scan",
            "10x Scan",
            "20x Scan",
            "Complete Scan (1 Chip)",
            "Full Stage Scan",
        )

        for scan_type in scan_types:
            with self.subTest(scan_type=scan_type):
                manager, stage, frame_processor, statuses = self._manager()

                def leave_stage_elsewhere(**_kwargs):
                    stage.position = (900, 800)

                manager.run_2x_scan = leave_stage_elsewhere
                manager.run_10x_scan = leave_stage_elsewhere
                manager.run_20x_scan = leave_stage_elsewhere
                manager.run_complete_scan = leave_stage_elsewhere

                completed = manager.run_scan(
                    scan_type,
                    window=(1, 1),
                    detection_model="Flake Detection",
                )

                self.assertTrue(completed)
                self.assertEqual(stage.position_reads, [True])
                self.assertEqual(stage.position, (120, 340))
                self.assertEqual(stage.moves, [(120, 340, True)])
                self.assertEqual(
                    frame_processor.motion_targets,
                    [(120, 340)],
                )
                self.assertIn(
                    {"processing_state": "Returning to start"},
                    statuses,
                )
                self.assertEqual(statuses[-1]["stage"], "Complete")

    def test_scan_error_still_returns_to_the_pre_scan_xy(self):
        manager, stage, _frame_processor, statuses = self._manager()

        def fail_after_move(**_kwargs):
            stage.position = (900, 800)
            raise RuntimeError("camera failed")

        manager.run_10x_scan = fail_after_move

        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            manager.run_scan("10x Scan", window=(1, 1))

        self.assertEqual(stage.position, (120, 340))
        self.assertEqual(stage.moves, [(120, 340, True)])
        self.assertEqual(statuses[-1]["stage"], "Error")

    def test_cancelled_scan_still_returns_to_the_pre_scan_xy(self):
        manager, stage, _frame_processor, statuses = self._manager()

        def cancel_after_move(**_kwargs):
            stage.position = (900, 800)
            manager._stop_event.set()

        manager.run_20x_scan = cancel_after_move

        completed = manager.run_scan("20x Scan", window=(1, 1))

        self.assertFalse(completed)
        self.assertEqual(stage.position, (120, 340))
        self.assertEqual(stage.moves, [(120, 340, True)])
        self.assertEqual(statuses[-1]["stage"], "Stopped")


if __name__ == "__main__":
    unittest.main()

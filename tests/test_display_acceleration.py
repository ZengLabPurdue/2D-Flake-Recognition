from collections import OrderedDict
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from UI.display_acceleration import (  # noqa: E402
    CPU_BACKEND,
    OPENCL_BACKEND,
    DisplayResizer,
    opencl_available,
    requested_backend,
)
from UI.sparse_tile_viewer import SparseTileViewer  # noqa: E402
from UI.panels.view_scans_panel import ViewScansPanel  # noqa: E402


class _FakeCanvas:
    @staticmethod
    def winfo_width():
        return 800

    @staticmethod
    def winfo_height():
        return 600


class _FakeVariable:
    def __init__(self, value=False):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DisplayAccelerationTests(unittest.TestCase):
    def test_camera_auto_uses_cpu_and_returns_requested_size(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        with patch.dict(
            os.environ,
            {"FLAKE_SEARCH_CAMERA_RENDERER": "auto"},
        ):
            resizer = DisplayResizer()

        resized, elapsed = resizer.resize(
            image,
            (50, 30),
            cv2.INTER_AREA,
        )

        self.assertEqual(resizer.backend, CPU_BACKEND)
        self.assertEqual(resized.shape, (30, 50, 3))
        self.assertGreaterEqual(elapsed, 0.0)

    def test_invalid_renderer_setting_uses_auto(self):
        with patch.dict(os.environ, {"TEST_RENDERER": "not-a-backend"}):
            self.assertEqual(requested_backend("TEST_RENDERER"), "auto")

    def test_interactive_map_buffer_is_smaller_than_quality_buffer(self):
        quality_size = SparseTileViewer._render_buffer_dimensions(1600, 900)
        interactive_size = SparseTileViewer._render_buffer_dimensions(
            1600,
            900,
            ratio=SparseTileViewer.INTERACTIVE_RENDER_BUFFER_RATIO,
        )

        self.assertEqual(quality_size, (3200, 1800))
        self.assertEqual(interactive_size, (1920, 1080))

    def test_stale_tile_request_does_not_open_the_source_image(self):
        viewer = self._viewer_without_tk()
        viewer._generation = 2
        missing_record = {"path": Path("this-file-does-not-exist.png")}

        sampled = viewer._sampled_tile(
            missing_record,
            factor=1,
            nearest=False,
            generation=1,
        )

        self.assertIsNone(sampled)

    def test_map_view_state_preserves_relative_position_and_zoom(self):
        source = self._viewer_without_tk()
        source.records = [object()]
        source.map_width = 1000.0
        source.map_height = 1000.0
        source.center_x = 400.0
        source.center_y = 600.0
        source.minimum_scale = 0.2
        source.scale = 0.4
        state = source.capture_view_state()

        target = self._viewer_without_tk()
        target.records = [object()]
        target.canvas = _FakeCanvas()
        target.map_width = 2000.0
        target.map_height = 2000.0
        target.maximum_scale = 1.0
        scheduled = []
        target._schedule_render = lambda **kwargs: scheduled.append(kwargs)

        target._apply_view_state(state)

        self.assertAlmostEqual(target.center_x, 800.0)
        self.assertAlmostEqual(target.center_y, 1200.0)
        self.assertAlmostEqual(
            target.scale / target.minimum_scale,
            2.0,
        )
        self.assertTrue(scheduled)

    def test_filtered_map_toggle_maps_2x_and_10x_layers(self):
        for current_view, checked, expected_view in (
            ("Raw 2x", True, "Filtered 2x"),
            ("Filtered 2x", False, "Raw 2x"),
            ("Raw 10x", True, "Processed 10x"),
            ("Processed 10x", False, "Raw 10x"),
        ):
            with self.subTest(current_view=current_view):
                panel = ViewScansPanel.__new__(ViewScansPanel)
                panel.selected_view = current_view
                pair = panel._map_layer_pair(current_view)
                panel.available_result_views = set(pair)
                panel.filtered_map_var = _FakeVariable(checked)
                expected_state = {
                    "center_x_ratio": 0.4,
                    "center_y_ratio": 0.6,
                    "zoom_ratio": 3.0,
                }
                panel.tile_viewer = type(
                    "_Viewer",
                    (),
                    {
                        "capture_view_state": (
                            lambda _self: expected_state
                        )
                    },
                )()
                calls = []
                panel.set_view_folder = (
                    lambda view, **kwargs: calls.append((view, kwargs))
                )

                panel._toggle_filtered_map()

                self.assertEqual(calls[0][0], expected_view)
                self.assertEqual(
                    calls[0][1]["view_state"],
                    expected_state,
                )
                self.assertTrue(calls[0][1]["preserve_context"])

    def test_map_opencl_failure_falls_back_to_cpu(self):
        viewer = self._viewer_without_tk()
        viewer.render_backend = OPENCL_BACKEND
        viewer._select_render_records = lambda _snapshot: ((object(),), False)

        def fail_opencl(*_args):
            raise RuntimeError("simulated OpenCL failure")

        viewer._compose_viewport_opencl = fail_opencl
        viewer._compose_viewport_cpu = lambda *_args: (
            Image.new("RGB", (4, 3)),
            [],
            {1},
        )
        snapshot = {
            "generation": 1,
            "width": 4,
            "height": 3,
            "title": "test",
            "scale": 1.0,
            "view_left": 0.0,
            "view_top": 0.0,
            "view_right": 4.0,
            "view_bottom": 3.0,
        }

        result = viewer._compose_viewport(snapshot)

        self.assertEqual(result["render_backend"], CPU_BACKEND)
        self.assertEqual(viewer.render_backend, CPU_BACKEND)
        self.assertIn("simulated OpenCL failure", viewer.render_fallback_reason)

    @unittest.skipUnless(opencl_available(), "OpenCL is unavailable")
    def test_opencl_map_composition_matches_nearest_cpu_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            tile_path = Path(directory) / "tile.png"
            y_values, x_values = np.mgrid[:80, :128]
            pixels = np.dstack(
                (
                    x_values.astype(np.uint8),
                    y_values.astype(np.uint8),
                    ((x_values + y_values) % 256).astype(np.uint8),
                )
            )
            Image.fromarray(pixels).save(tile_path)
            record = {
                "path": tile_path,
                "x": 0.0,
                "y": 0.0,
                "width": 128.0,
                "height": 80.0,
                "pixel_width": 128,
                "pixel_height": 80,
                "map_zoom": 1.0,
                "map_width": 128.0,
                "map_height": 80.0,
            }
            viewer = self._viewer_without_tk()
            snapshot = {
                "generation": 1,
                "width": 96,
                "height": 60,
                "viewport_width": 96,
                "viewport_height": 60,
                "view_left": 16.0,
                "view_top": 10.0,
                "view_right": 112.0,
                "view_bottom": 70.0,
                "scale": 1.0,
                "all_records": (record,),
                "buckets": {(0, 0): [0]},
                "overview_record": None,
                "sparse_tile_limit": 64,
                "title": "test",
                "nearest": True,
            }

            viewer.render_backend = CPU_BACKEND
            cpu_result = viewer._compose_viewport(snapshot)
            viewer.render_backend = OPENCL_BACKEND
            gpu_result = viewer._compose_viewport(snapshot)

            self.assertIsNotNone(cpu_result)
            self.assertIsNotNone(gpu_result)
            self.assertEqual(gpu_result["render_backend"], OPENCL_BACKEND)
            np.testing.assert_array_equal(
                np.asarray(cpu_result["viewport"]),
                np.asarray(gpu_result["viewport"]),
            )
            self.assertGreater(viewer._gpu_cache_bytes, 0)

    @staticmethod
    def _viewer_without_tk():
        viewer = SparseTileViewer.__new__(SparseTileViewer)
        viewer._generation = 1
        viewer._shutdown = False
        viewer._sampled_cache = OrderedDict()
        viewer._cache_bytes = 0
        viewer._gpu_cache = OrderedDict()
        viewer._gpu_cache_bytes = 0
        viewer._cache_lock = threading.Lock()
        viewer.render_fallback_reason = None
        viewer.render_device = cv2.ocl.Device_getDefault().name()
        return viewer


if __name__ == "__main__":
    unittest.main()

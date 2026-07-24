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

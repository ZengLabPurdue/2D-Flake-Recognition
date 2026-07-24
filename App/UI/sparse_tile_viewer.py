"""Responsive viewport renderer for metadata-positioned scan tiles."""

from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
import math
import os
from pathlib import Path
import re
import threading
from time import perf_counter
from tkinter import TclError

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from Imaging import image_metadata
from UI.display_acceleration import (
    CPU_BACKEND,
    OPENCL_BACKEND,
    opencl_available,
    opencl_device_name,
    requested_backend,
)


class SparseTileViewer:
    """Render just the scan tiles intersecting a pannable, zoomable viewport.

    A background render worker coordinates a bounded image-decoding pool and
    viewport composition. The Tk thread only handles input, lightweight
    loading placeholders, and publishing the completed ``PhotoImage``.
    """

    OUTSIDE_MARGIN_RATIO = 0.04
    ZOOM_STEP = 1.25
    CACHE_LIMIT_BYTES = 128 * 1024 * 1024
    GPU_CACHE_LIMIT_BYTES = 256 * 1024 * 1024
    BUCKET_SIZE = 512
    RENDER_DEBOUNCE_MS = 35
    QUALITY_RENDER_DELAY_MS = 120
    POLL_INTERVAL_MS = 16
    PLACEHOLDER_ZOOM_RATIO = 1.5
    PLACEHOLDER_COLOR = "#eeeeee"
    RENDER_BUFFER_RATIO = 2.0
    INTERACTIVE_RENDER_BUFFER_RATIO = 1.2
    RENDER_BUFFER_MAX_PIXELS = 12_000_000
    BUFFER_PREFETCH_GUARD_RATIO = 0.12
    MAX_SPARSE_RENDER_TILES = 64
    OVERVIEW_MIN_SAMPLE_FACTOR = 2
    MAX_SAMPLE_FACTOR = 128
    DECODE_WORKERS = max(2, min(8, os.cpu_count() or 4))

    def __init__(self, canvas):
        self.canvas = canvas
        self.records = []
        self._overview_record = None
        self._buckets = {}
        self.map_width = 1.0
        self.map_height = 1.0
        self.center_x = 0.5
        self.center_y = 0.5
        self.scale = 1.0
        self.minimum_scale = 1.0
        self.maximum_scale = 1.0
        self.title = ""
        self.nearest = False

        self._drag_position = None
        self._render_job = None
        self._poll_job = None
        self._photo = None
        self._buffer_view = None
        self._last_render_metrics = None
        self._generation = 0
        self._shutdown = False

        requested_renderer = requested_backend("FLAKE_SEARCH_MAP_RENDERER")
        self.render_backend = (
            OPENCL_BACKEND
            if requested_renderer in ("auto", OPENCL_BACKEND) and opencl_available()
            else CPU_BACKEND
        )
        self.render_device = (
            opencl_device_name() if self.render_backend == OPENCL_BACKEND else None
        )
        self.render_fallback_reason = (
            "OpenCL is unavailable"
            if requested_renderer == OPENCL_BACKEND
            and self.render_backend != OPENCL_BACKEND
            else None
        )

        # There is never more than one running task and one replaceable pending
        # task.  Rapid dragging therefore cannot build an unbounded executor
        # backlog; the pending slot always contains the newest viewport.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="sparse-map-render",
        )
        # PNG decompression dominates a cold detailed zoom. Independent image
        # files can be decoded safely in parallel while the render worker
        # preserves deterministic composition order.
        self._decode_executor = ThreadPoolExecutor(
            max_workers=self.DECODE_WORKERS,
            thread_name_prefix="sparse-map-decode",
        )
        self._future = None
        self._future_generation = None
        self._pending_task = None

        self._sampled_cache = OrderedDict()
        self._cache_bytes = 0
        self._gpu_cache = OrderedDict()
        self._gpu_cache_bytes = 0
        self._cache_lock = threading.Lock()

        canvas.configure(bg="black", highlightthickness=0, cursor="fleur")
        canvas.bind("<Configure>", self._on_configure)
        canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1))
        canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, -1))
        canvas.bind("<ButtonPress-1>", self._start_pan)
        canvas.bind("<B1-Motion>", self._pan)
        canvas.bind("<ButtonRelease-1>", self._end_pan)
        canvas.bind("<Double-Button-1>", lambda _event: self.fit_to_view())

    @staticmethod
    def _number(metadata, key, default=None):
        value = metadata.get(key, default)
        if value is None:
            raise ValueError(f"Missing sparse-map metadata: {key}")
        return float(value)

    def clear(self, message="No scan map selected"):
        """Cancel obsolete work and replace the viewer with a message."""
        if self._shutdown:
            return
        self._invalidate_work()
        self.records = []
        self._overview_record = None
        self._buckets = {}
        self._clear_render_buffer()
        self._clear_cache()
        self.canvas.delete("all")
        self.canvas.create_text(
            max(1, self.canvas.winfo_width()) // 2,
            max(1, self.canvas.winfo_height()) // 2,
            text=message,
            fill="#b0b0b0",
            font=("TkDefaultFont", 14),
        )

    def load_tiles(
        self,
        tile_paths,
        title="",
        nearest=False,
        on_empty=None,
        overview_path=None,
    ):
        """Index positioned PNG tiles without blocking the Tk event loop.

        ``True`` means an asynchronous index operation was accepted.  If no
        usable positioned tiles are found, ``on_empty`` is called on the Tk
        thread so the caller can load a legacy flattened-map fallback.
        """
        folder = None
        paths = None
        if isinstance(tile_paths, (str, Path)):
            folder = Path(tile_paths)
        else:
            paths = tuple(Path(path) for path in tile_paths)
        if paths == ():
            self.clear("No positioned scan tiles were found")
            return False

        generation = self._invalidate_work()
        self.records = []
        self._overview_record = None
        self._buckets = {}
        self.title = title
        self.nearest = nearest
        self._clear_render_buffer()
        self._clear_cache()
        self._show_message(f"Indexing {title or 'scan map'}...")

        self._pending_task = {
            "kind": "index",
            "generation": generation,
            "folder": folder,
            "paths": paths,
            "overview_path": Path(overview_path) if overview_path else None,
            "title": title,
            "nearest": nearest,
            "on_empty": on_empty,
        }
        self._start_pending_task()
        return True

    def load_image(self, image_path, title="", nearest=False):
        """Load a single image through the same asynchronous viewport renderer."""
        image_path = Path(image_path)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError:
            self.clear(f"Could not open {image_path.name}")
            return False

        self._invalidate_work()
        self.records = [{
            "path": image_path,
            "x": 0.0,
            "y": 0.0,
            "width": float(width),
            "height": float(height),
            "pixel_width": width,
            "pixel_height": height,
            "map_zoom": 1.0,
            "map_width": float(width),
            "map_height": float(height),
        }]
        self._overview_record = None
        self._index_records()
        self.map_width = float(width)
        self.map_height = float(height)
        # Do not allow a monolithic 10k map into an uncached factor-1 band.
        # Its highest useful level is the finest RGB pyramid level that fits
        # the bounded cache; sparse scans still retain their higher native zoom.
        cache_factor = self._minimum_cache_factor(width, height)
        self.maximum_scale = 1.0 / cache_factor
        self.title = title or image_path.name
        self.nearest = nearest
        self._clear_render_buffer()
        self._clear_cache()
        self.fit_to_view()
        return True

    def fit_to_view(self):
        if self._shutdown or not self.records:
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        margin = max(24, int(min(width, height) * self.OUTSIDE_MARGIN_RATIO))
        available_width = max(1, width - margin * 2)
        available_height = max(1, height - margin * 2)
        self.minimum_scale = min(
            available_width / max(1.0, self.map_width),
            available_height / max(1.0, self.map_height),
        )
        self.maximum_scale = max(self.minimum_scale, self.maximum_scale)
        self.scale = self.minimum_scale
        self.center_x = self.map_width / 2
        self.center_y = self.map_height / 2
        self._schedule_render(delay_ms=0, force_placeholder=True)

    def pause(self):
        """Stop obsolete viewer work while another application view is active."""
        if not self._shutdown:
            self._invalidate_work()

    def shutdown(self):
        """Cancel callbacks and prevent worker results from touching destroyed Tk."""
        if self._shutdown:
            return
        self._shutdown = True
        self._generation += 1
        self._pending_task = None
        self._cancel_after("_render_job")
        self._cancel_after("_poll_job")
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._decode_executor.shutdown(wait=False, cancel_futures=True)
        self._clear_cache()

    def _on_configure(self, _event):
        if self._shutdown or not self.records:
            return
        if self.scale <= self.minimum_scale * 1.001:
            self.fit_to_view()
        else:
            self._clamp_center()
            if self._buffer_covers_current_view():
                self._display_render_buffer(loading=False)
                self._maybe_prefetch()
            else:
                self._schedule_render(force_placeholder=True)

    def _on_mouse_wheel(self, event):
        self._zoom_at(event.x, event.y, 1 if event.delta > 0 else -1)

    def _zoom_at(self, screen_x, screen_y, direction):
        if self._shutdown or not self.records:
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        map_x = self.center_x + (screen_x - width / 2) / self.scale
        map_y = self.center_y + (screen_y - height / 2) / self.scale
        factor = self.ZOOM_STEP if direction > 0 else 1 / self.ZOOM_STEP
        new_scale = min(self.maximum_scale, max(self.minimum_scale, self.scale * factor))
        if math.isclose(new_scale, self.scale, rel_tol=1e-12):
            return
        self.center_x = map_x - (screen_x - width / 2) / new_scale
        self.center_y = map_y - (screen_y - height / 2) / new_scale
        self.scale = new_scale
        self._clamp_center()
        self._schedule_render(
            force_placeholder=True,
            interactive=True,
        )

    def _start_pan(self, event):
        self._drag_position = (event.x, event.y)

    def _pan(self, event):
        if self._drag_position is None or not self.records or self._shutdown:
            return
        previous_x, previous_y = self._drag_position
        old_center = (self.center_x, self.center_y)
        self.center_x -= (event.x - previous_x) / self.scale
        self.center_y -= (event.y - previous_y) / self.scale
        self._drag_position = (event.x, event.y)
        self._clamp_center()
        if (
            math.isclose(self.center_x, old_center[0], rel_tol=1e-12)
            and math.isclose(self.center_y, old_center[1], rel_tol=1e-12)
        ):
            return
        if self._buffer_covers_current_view():
            self._display_render_buffer(loading=False)
            self._maybe_prefetch()
        else:
            self._schedule_render(force_placeholder=True)

    def _end_pan(self, _event):
        was_dragging = self._drag_position is not None
        self._drag_position = None
        if was_dragging and self.records:
            if self._buffer_covers_current_view():
                self._display_render_buffer(loading=False)
                self._maybe_prefetch(delay_ms=0)
            else:
                self._schedule_render(delay_ms=0, force_placeholder=True)

    def _clamp_center(self):
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        margin = max(24, int(min(width, height) * self.OUTSIDE_MARGIN_RATIO))
        half_width = width / (2 * self.scale)
        half_height = height / (2 * self.scale)
        margin_map = margin / self.scale

        minimum_x = half_width - margin_map
        maximum_x = self.map_width - half_width + margin_map
        minimum_y = half_height - margin_map
        maximum_y = self.map_height - half_height + margin_map
        self.center_x = (
            self.map_width / 2
            if minimum_x > maximum_x
            else min(maximum_x, max(minimum_x, self.center_x))
        )
        self.center_y = (
            self.map_height / 2
            if minimum_y > maximum_y
            else min(maximum_y, max(minimum_y, self.center_y))
        )

    def _schedule_render(
        self,
        delay_ms=None,
        force_placeholder=False,
        interactive=False,
    ):
        if self._shutdown or not self.records:
            return
        if delay_ms is None:
            delay_ms = self.RENDER_DEBOUNCE_MS

        self._generation += 1
        generation = self._generation
        self._pending_task = None
        self._cancel_after("_render_job")

        high_zoom = self.scale >= self.minimum_scale * self.PLACEHOLDER_ZOOM_RATIO
        if force_placeholder or high_zoom:
            self._show_placeholder()

        try:
            self._render_job = self.canvas.after(
                max(0, int(delay_ms)),
                lambda: self._queue_current_view(generation, interactive),
            )
        except TclError:
            self._render_job = None

    def _queue_current_view(self, generation, interactive=False):
        self._render_job = None
        if self._shutdown or generation != self._generation or not self.records:
            return
        snapshot = self._make_snapshot(
            generation,
            interactive=interactive,
        )
        self._pending_task = {
            "kind": "render",
            "generation": generation,
            "snapshot": snapshot,
        }
        self._start_pending_task()

    def _start_pending_task(self):
        if (
            self._shutdown
            or self._future is not None
            or self._pending_task is None
        ):
            return
        task = self._pending_task
        self._pending_task = None
        try:
            self._future = self._executor.submit(self._run_task, task)
            self._future_generation = task["generation"]
        except RuntimeError:
            self._future = None
            self._future_generation = None
            return
        self._ensure_polling()

    def _ensure_polling(self):
        if self._shutdown or self._poll_job is not None or self._future is None:
            return
        try:
            self._poll_job = self.canvas.after(
                self.POLL_INTERVAL_MS,
                self._poll_worker,
            )
        except TclError:
            self._poll_job = None

    def _poll_worker(self):
        self._poll_job = None
        if self._shutdown:
            return
        if self._future is None:
            self._start_pending_task()
            return
        if not self._future.done():
            self._ensure_polling()
            return

        future = self._future
        future_generation = self._future_generation
        self._future = None
        self._future_generation = None
        try:
            result = future.result()
        except CancelledError:
            result = None
        except Exception as error:  # A bad tile must not kill future rendering.
            result = {
                "kind": "error",
                "generation": future_generation,
                "message": str(error),
            }

        if result is not None and result.get("generation") == self._generation:
            kind = result.get("kind")
            if kind == "render":
                self._publish_viewport(result)
            elif kind == "index":
                self._apply_index_result(result)
            elif kind == "error":
                self._show_message("Could not render this scan map")

        self._start_pending_task()
        self._ensure_polling()

    def _run_task(self, task):
        if task["kind"] == "index":
            return self._index_tile_paths(task)
        return self._compose_viewport(task["snapshot"])

    def _index_tile_paths(self, task):
        generation = task["generation"]
        records = []
        prefix = image_metadata.METADATA_PREFIX
        if task["folder"] is not None:
            try:
                paths = []
                for path in task["folder"].iterdir():
                    if self._is_stale(generation):
                        return None
                    if path.suffix.lower() == ".png":
                        paths.append(path)
            except OSError:
                paths = []
        else:
            paths = list(task["paths"])
        if self._is_stale(generation):
            return None
        paths = sorted(paths, key=self._natural_path_key)

        for tile_path in paths:
            if self._is_stale(generation):
                return None
            try:
                # Read the PNG dimensions and positioning fields with one file
                # open.  Accessing this header does not decode the image pixels.
                with Image.open(tile_path) as image:
                    metadata = {
                        key[len(prefix):]: str(value)
                        for key, value in image.info.items()
                        if key.startswith(prefix)
                    }
                    image_width, image_height = image.size
                map_x = self._number(metadata, "map_x")
                map_y = self._number(metadata, "map_y")
                map_zoom = max(1.0, self._number(metadata, "map_zoom"))
                map_width = self._number(metadata, "map_width")
                map_height = self._number(metadata, "map_height")
            except (OSError, TypeError, ValueError):
                continue
            records.append({
                "path": tile_path,
                "x": map_x,
                "y": map_y,
                "width": float(math.ceil(image_width / map_zoom)),
                "height": float(math.ceil(image_height / map_zoom)),
                "pixel_width": image_width,
                "pixel_height": image_height,
                "map_zoom": map_zoom,
                "map_width": map_width,
                "map_height": map_height,
            })

        if self._is_stale(generation):
            return None
        overview_record = self._read_overview_record(
            task["overview_path"],
            records,
            generation,
        )
        if self._is_stale(generation):
            return None
        buckets = self._build_bucket_index(records, generation=generation)
        if buckets is None:
            return None
        return {
            "kind": "index",
            "generation": generation,
            "records": records,
            "overview_record": overview_record,
            "buckets": buckets,
            "title": task["title"],
            "nearest": task["nearest"],
            "on_empty": task["on_empty"],
        }

    def _apply_index_result(self, result):
        records = result["records"]
        if not records:
            callback = result.get("on_empty")
            if callback is not None:
                callback()
            else:
                self.clear("No positioned scan tiles were found")
            return

        self.records = records
        self._overview_record = result["overview_record"]
        self._buckets = result["buckets"]
        self.map_width = max(record["map_width"] for record in records)
        self.map_height = max(record["map_height"] for record in records)
        self.maximum_scale = max(
            1.0,
            min(record["map_zoom"] for record in records),
        )
        self.title = result["title"]
        self.nearest = result["nearest"]
        self.fit_to_view()

    def _make_snapshot(self, generation, interactive=False):
        viewport_width = max(1, self.canvas.winfo_width())
        viewport_height = max(1, self.canvas.winfo_height())
        width, height = self._render_buffer_dimensions(
            viewport_width,
            viewport_height,
            ratio=(
                self.INTERACTIVE_RENDER_BUFFER_RATIO
                if interactive
                else self.RENDER_BUFFER_RATIO
            ),
        )
        # Do not allocate overscan along an axis where the whole map already
        # fits; there is nowhere useful to pan in that direction.
        if self.map_width * self.scale <= viewport_width:
            width = viewport_width
        if self.map_height * self.scale <= viewport_height:
            height = viewport_height
        view_left = self.center_x - width / (2 * self.scale)
        view_top = self.center_y - height / (2 * self.scale)
        view_right = self.center_x + width / (2 * self.scale)
        view_bottom = self.center_y + height / (2 * self.scale)

        return {
            "generation": generation,
            "width": width,
            "height": height,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "view_left": view_left,
            "view_top": view_top,
            "view_right": view_right,
            "view_bottom": view_bottom,
            "scale": self.scale,
            # These containers are replaced, never mutated, when another scan
            # is loaded.  The worker can therefore safely use this snapshot.
            "all_records": self.records,
            "buckets": self._buckets,
            "overview_record": self._overview_record,
            "sparse_tile_limit": max(
                self.MAX_SPARSE_RENDER_TILES,
                int(
                    math.ceil(
                        self.MAX_SPARSE_RENDER_TILES
                        * width
                        * height
                        / (viewport_width * viewport_height)
                    )
                ),
            ),
            "title": self.title,
            "nearest": self.nearest,
            "interactive": interactive,
        }

    def _compose_viewport(self, snapshot):
        started_at = perf_counter()
        generation = snapshot["generation"]
        if self._is_stale(generation):
            return None

        render_records, using_overview = self._select_render_records(snapshot)
        if render_records is None:
            return None

        if self.render_backend == OPENCL_BACKEND:
            try:
                viewport, failed_bounds, sample_factors = (
                    self._compose_viewport_opencl(
                        snapshot,
                        render_records,
                        using_overview,
                    )
                )
            except (AttributeError, TypeError, cv2.error, RuntimeError) as exc:
                # OpenCL availability can change with drivers, displays, and
                # remote sessions. Fall back without losing the requested view.
                self.render_backend = CPU_BACKEND
                self.render_device = None
                self.render_fallback_reason = str(exc)
                self._clear_gpu_cache()
                viewport, failed_bounds, sample_factors = self._compose_viewport_cpu(
                    snapshot,
                    render_records,
                    using_overview,
                )
        else:
            viewport, failed_bounds, sample_factors = self._compose_viewport_cpu(
                snapshot,
                render_records,
                using_overview,
            )

        if viewport is None:
            return None
        if failed_bounds:
            draw = ImageDraw.Draw(viewport)
            for left, top, right, bottom in failed_bounds:
                draw.rectangle(
                    (left, top, max(left, right - 1), max(top, bottom - 1)),
                    fill=self.PLACEHOLDER_COLOR,
                )

        if self._is_stale(generation):
            return None
        return {
            "kind": "render",
            "generation": generation,
            "viewport": viewport,
            "width": snapshot["width"],
            "height": snapshot["height"],
            "title": snapshot["title"],
            "scale": snapshot["scale"],
            "sample_factors": tuple(sorted(sample_factors)),
            "using_overview": using_overview,
            "view_left": snapshot["view_left"],
            "view_top": snapshot["view_top"],
            "view_right": snapshot["view_right"],
            "view_bottom": snapshot["view_bottom"],
            "render_backend": self.render_backend,
            "render_device": self.render_device,
            "render_ms": (perf_counter() - started_at) * 1000,
            "render_tile_count": len(render_records),
            "interactive": snapshot.get("interactive", False),
        }

    def _prepare_render_tiles(self, snapshot, render_records, using_overview):
        """Decode visible tiles concurrently, preserving their draw order."""
        generation = snapshot["generation"]
        specifications = []
        failed_bounds = []
        sample_factors = set()
        for record in render_records:
            if self._is_stale(generation):
                return None, failed_bounds, sample_factors
            bounds = self._record_screen_bounds(record, snapshot)
            if bounds is None:
                continue

            sample_factor = self._sample_factor(record, snapshot["scale"])
            if using_overview:
                sample_factor = max(
                    self.OVERVIEW_MIN_SAMPLE_FACTOR,
                    sample_factor,
                )
            sample_factors.add(sample_factor)
            specifications.append((record, bounds, sample_factor))

        prepared = []
        decode_executor = getattr(self, "_decode_executor", None)
        futures = None
        if decode_executor is not None and len(specifications) > 1:
            try:
                futures = [
                    decode_executor.submit(
                        self._sampled_tile,
                        record,
                        sample_factor,
                        snapshot["nearest"],
                        generation,
                    )
                    for record, _bounds, sample_factor in specifications
                ]
            except RuntimeError:
                futures = None

        for index, (record, bounds, sample_factor) in enumerate(specifications):
            if self._is_stale(generation):
                if futures is not None:
                    for future in futures[index:]:
                        future.cancel()
                return None, failed_bounds, sample_factors
            try:
                sampled = (
                    futures[index].result()
                    if futures is not None
                    else self._sampled_tile(
                        record,
                        sample_factor,
                        snapshot["nearest"],
                        generation,
                    )
                )
                if sampled is None:
                    if futures is not None:
                        for future in futures[index + 1:]:
                            future.cancel()
                    return None, failed_bounds, sample_factors
                prepared.append((record, bounds, sample_factor, sampled))
            except (OSError, ValueError):
                failed_bounds.append(bounds)

        return prepared, failed_bounds, sample_factors

    def _compose_viewport_cpu(self, snapshot, render_records, using_overview):
        generation = snapshot["generation"]
        viewport = Image.new(
            "RGB",
            (snapshot["width"], snapshot["height"]),
            "black",
        )
        prepared, failed_bounds, sample_factors = self._prepare_render_tiles(
            snapshot,
            render_records,
            using_overview,
        )
        if prepared is None:
            return None, failed_bounds, sample_factors

        for record, bounds, _sample_factor, sampled in prepared:
            tile = self._visible_tile_image(
                sampled,
                record,
                snapshot,
                bounds,
            )
            if self._is_stale(generation):
                return None, failed_bounds, sample_factors
            viewport.paste(tile, (bounds[0], bounds[1]))

        if self._is_stale(generation):
            return None, failed_bounds, sample_factors
        return viewport, failed_bounds, sample_factors

    def _compose_viewport_opencl(self, snapshot, render_records, using_overview):
        generation = snapshot["generation"]
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.useOpenCL():
            raise RuntimeError("OpenCV could not activate the OpenCL device")
        viewport_gpu = cv2.UMat(
            snapshot["height"],
            snapshot["width"],
            cv2.CV_8UC3,
        )
        cv2.rectangle(
            viewport_gpu,
            (0, 0),
            (snapshot["width"], snapshot["height"]),
            (0, 0, 0),
            thickness=-1,
        )
        prepared, failed_bounds, sample_factors = self._prepare_render_tiles(
            snapshot,
            render_records,
            using_overview,
        )
        if prepared is None:
            return None, failed_bounds, sample_factors

        for record, bounds, sample_factor, sampled in prepared:
            if self._is_stale(generation):
                return None, failed_bounds, sample_factors
            try:
                sampled_gpu = self._gpu_sampled_tile(
                    record,
                    sample_factor,
                    snapshot["nearest"],
                    sampled,
                    generation,
                )
                if sampled_gpu is None:
                    return None, failed_bounds, sample_factors
                source_box = self._visible_source_box(
                    sampled,
                    record,
                    snapshot,
                    bounds,
                )
                source_left = max(0, int(math.floor(source_box[0])))
                source_top = max(0, int(math.floor(source_box[1])))
                source_right = min(sampled.width, int(math.ceil(source_box[2])))
                source_bottom = min(sampled.height, int(math.ceil(source_box[3])))
                source_width = source_right - source_left
                source_height = source_bottom - source_top
                if source_width <= 0 or source_height <= 0:
                    continue

                destination_width = bounds[2] - bounds[0]
                destination_height = bounds[3] - bounds[1]
                source_roi = cv2.UMat(
                    sampled_gpu,
                    (source_left, source_top, source_width, source_height),
                )
                interpolation = (
                    cv2.INTER_NEAREST
                    if snapshot["nearest"]
                    else (
                        cv2.INTER_LINEAR
                        if snapshot.get("interactive", False)
                        else cv2.INTER_LANCZOS4
                    )
                )
                resized = cv2.resize(
                    source_roi,
                    (destination_width, destination_height),
                    interpolation=interpolation,
                )
                destination_roi = cv2.UMat(
                    viewport_gpu,
                    (
                        bounds[0],
                        bounds[1],
                        destination_width,
                        destination_height,
                    ),
                )
                cv2.copyTo(resized, None, destination_roi)
            except (OSError, ValueError):
                failed_bounds.append(bounds)

        if self._is_stale(generation):
            return None, failed_bounds, sample_factors
        return Image.fromarray(viewport_gpu.get()), failed_bounds, sample_factors

    def _select_render_records(self, snapshot):
        """Select a bounded sparse set, or one low-resolution overview."""
        overview = snapshot["overview_record"]
        if (
            overview is not None
            and snapshot["scale"]
            <= overview["map_zoom"] / self.OVERVIEW_MIN_SAMPLE_FACTOR
        ):
            return (overview,), True

        limit = snapshot["sparse_tile_limit"] if overview is not None else None
        visible, overflow = self._collect_visible_records(
            snapshot["all_records"],
            snapshot["buckets"],
            snapshot["view_left"],
            snapshot["view_top"],
            snapshot["view_right"],
            snapshot["view_bottom"],
            limit=limit,
            generation=snapshot["generation"],
        )
        if visible is None:
            return None, False
        if overflow and overview is not None:
            return (overview,), True
        return visible, False

    @classmethod
    def _sample_factor(cls, record, scale):
        """Choose one cached power-of-two level for an entire zoom range."""
        source_pixels_per_screen_pixel = max(1.0, record["map_zoom"] / scale)
        factor = 2 ** int(math.floor(math.log2(source_pixels_per_screen_pixel)))
        largest_source_factor = 2 ** int(
            math.floor(
                math.log2(max(1, min(record["pixel_width"], record["pixel_height"])))
            )
        )
        return max(1, min(cls.MAX_SAMPLE_FACTOR, largest_source_factor, factor))

    @classmethod
    def _render_buffer_dimensions(
        cls,
        viewport_width,
        viewport_height,
        ratio=None,
    ):
        viewport_pixels = max(1, viewport_width * viewport_height)
        memory_ratio = math.sqrt(cls.RENDER_BUFFER_MAX_PIXELS / viewport_pixels)
        requested_ratio = cls.RENDER_BUFFER_RATIO if ratio is None else ratio
        ratio = max(1.0, min(requested_ratio, memory_ratio))
        return (
            max(viewport_width, int(round(viewport_width * ratio))),
            max(viewport_height, int(round(viewport_height * ratio))),
        )

    @classmethod
    def _minimum_cache_factor(cls, width, height):
        factor = 1
        while (
            math.ceil(width / factor)
            * math.ceil(height / factor)
            * 3
            > cls.CACHE_LIMIT_BYTES
        ):
            factor *= 2
        return factor

    def _sampled_tile(self, record, factor, nearest, generation):
        if self._is_stale(generation):
            return None
        key = (str(record["path"].resolve()), factor, nearest)
        with self._cache_lock:
            cached = self._sampled_cache.pop(key, None)
            if cached is not None:
                self._sampled_cache[key] = cached
                if self._is_stale(generation):
                    return None
                return cached

        with Image.open(record["path"]) as source:
            if factor == 1:
                sampled = source.convert("RGB")
            elif nearest:
                sample_size = (
                    max(1, math.ceil(source.width / factor)),
                    max(1, math.ceil(source.height / factor)),
                )
                sampled = source.resize(sample_size, Image.Resampling.NEAREST).convert("RGB")
            else:
                sampled = source.reduce(factor).convert("RGB")

        if self._is_stale(generation):
            return None

        cache_size = sampled.width * sampled.height * 3
        if cache_size <= self.CACHE_LIMIT_BYTES:
            with self._cache_lock:
                if self._is_stale(generation):
                    return None
                existing = self._sampled_cache.pop(key, None)
                if existing is not None:
                    self._cache_bytes -= (
                        existing.width * existing.height * 3
                    )
                while (
                    self._sampled_cache
                    and self._cache_bytes + cache_size > self.CACHE_LIMIT_BYTES
                ):
                    _, removed = self._sampled_cache.popitem(last=False)
                    self._cache_bytes -= removed.width * removed.height * 3
                self._sampled_cache[key] = sampled
                self._cache_bytes += cache_size
        return sampled

    def _gpu_sampled_tile(
        self,
        record,
        factor,
        nearest,
        sampled,
        generation,
    ):
        """Upload one pyramid tile once and retain it across view renders."""
        key = (str(record["path"].resolve()), factor, nearest)
        with self._cache_lock:
            cached = self._gpu_cache.pop(key, None)
            if cached is not None:
                self._gpu_cache[key] = cached
                if self._is_stale(generation):
                    return None
                return cached[0]

        pixels = np.ascontiguousarray(np.asarray(sampled, dtype=np.uint8))
        uploaded = cv2.UMat(pixels)
        cache_size = int(pixels.nbytes)
        if self._is_stale(generation):
            return None

        if cache_size <= self.GPU_CACHE_LIMIT_BYTES:
            with self._cache_lock:
                if self._is_stale(generation):
                    return None
                while (
                    self._gpu_cache
                    and self._gpu_cache_bytes + cache_size
                    > self.GPU_CACHE_LIMIT_BYTES
                ):
                    _, removed = self._gpu_cache.popitem(last=False)
                    self._gpu_cache_bytes -= removed[1]
                self._gpu_cache[key] = (uploaded, cache_size)
                self._gpu_cache_bytes += cache_size
        return uploaded

    @staticmethod
    def _visible_source_box(sampled, record, snapshot, bounds):
        left, top, right, bottom = bounds
        scale = snapshot["scale"]

        map_left = snapshot["view_left"] + left / scale
        map_top = snapshot["view_top"] + top / scale
        map_right = snapshot["view_left"] + right / scale
        map_bottom = snapshot["view_top"] + bottom / scale

        return (
            max(0.0, (map_left - record["x"]) / record["width"] * sampled.width),
            max(0.0, (map_top - record["y"]) / record["height"] * sampled.height),
            min(
                float(sampled.width),
                (map_right - record["x"]) / record["width"] * sampled.width,
            ),
            min(
                float(sampled.height),
                (map_bottom - record["y"]) / record["height"] * sampled.height,
            ),
        )

    @staticmethod
    def _visible_tile_image(sampled, record, snapshot, bounds):
        left, top, right, bottom = bounds
        destination_size = (right - left, bottom - top)
        source_box = SparseTileViewer._visible_source_box(
            sampled,
            record,
            snapshot,
            bounds,
        )
        resampling = (
            Image.Resampling.NEAREST
            if snapshot["nearest"]
            else (
                Image.Resampling.BILINEAR
                if snapshot.get("interactive", False)
                else Image.Resampling.LANCZOS
            )
        )
        return sampled.resize(destination_size, resampling, box=source_box)

    def _publish_viewport(self, result):
        if self._shutdown:
            return
        candidate_view = {
            "left": result["view_left"],
            "top": result["view_top"],
            "right": result["view_right"],
            "bottom": result["view_bottom"],
            "scale": result["scale"],
            "sample_factors": result["sample_factors"],
            "using_overview": result["using_overview"],
        }
        candidate_covers = self._view_covers_current(candidate_view)
        current_covers = self._buffer_covers_current_view()
        if current_covers and not candidate_covers:
            # The user panned back into the old buffer while this request was
            # rendering.  Keep the useful bitmap instead of replacing it with
            # a now-obsolete viewport.
            self._display_render_buffer(loading=False)
            self._maybe_prefetch()
            return

        publish_started_at = perf_counter()
        viewport = result["viewport"]
        if (
            self._photo is not None
            and self._photo.width() == viewport.width
            and self._photo.height() == viewport.height
        ):
            self._photo.paste(viewport)
        else:
            self._photo = ImageTk.PhotoImage(viewport)
        result["publish_ms"] = (perf_counter() - publish_started_at) * 1000
        self._last_render_metrics = result
        self._buffer_view = candidate_view
        covered = self._buffer_covers_current_view()
        self._display_render_buffer(loading=not covered)
        if covered:
            if result.get("interactive", False):
                # Refine the small, fast zoom preview with the normal
                # overscanned, high-quality buffer after wheel input settles.
                self._schedule_render(
                    delay_ms=self.QUALITY_RENDER_DELAY_MS,
                )
            else:
                self._maybe_prefetch()
        else:
            self._schedule_render(delay_ms=0)

    def _show_placeholder(self):
        if self._shutdown or not self.records:
            return
        self._display_render_buffer(loading=True)

    def _display_render_buffer(self, loading=False):
        """Move the reusable bitmap over the viewport without recomposing it."""
        if self._shutdown or not self.records:
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        view_left, view_top, _view_right, _view_bottom = (
            self._current_view_bounds(width, height)
        )
        self.canvas.delete("all")

        map_left = max(0, int(round((0.0 - view_left) * self.scale)))
        map_top = max(0, int(round((0.0 - view_top) * self.scale)))
        map_right = min(
            width,
            int(round((self.map_width - view_left) * self.scale)),
        )
        map_bottom = min(
            height,
            int(round((self.map_height - view_top) * self.scale)),
        )
        if map_right > map_left and map_bottom > map_top:
            self.canvas.create_rectangle(
                map_left,
                map_top,
                map_right,
                map_bottom,
                fill=self.PLACEHOLDER_COLOR,
                outline="",
            )

        buffer_view = self._buffer_view
        if (
            self._photo is not None
            and buffer_view is not None
            and math.isclose(
                buffer_view["scale"],
                self.scale,
                rel_tol=1e-12,
            )
        ):
            image_x = int(round((buffer_view["left"] - view_left) * self.scale))
            image_y = int(round((buffer_view["top"] - view_top) * self.scale))
            self.canvas.create_image(
                image_x,
                image_y,
                image=self._photo,
                anchor="nw",
            )

        self._draw_overlay(
            width,
            height,
            self.title,
            self.scale,
            sample_factors=(
                buffer_view.get("sample_factors", ())
                if buffer_view is not None
                else ()
            ),
            using_overview=(
                buffer_view.get("using_overview", False)
                if buffer_view is not None
                else False
            ),
            loading=loading,
        )

    def _current_view_bounds(self, width=None, height=None):
        width = max(1, self.canvas.winfo_width()) if width is None else width
        height = max(1, self.canvas.winfo_height()) if height is None else height
        return (
            self.center_x - width / (2 * self.scale),
            self.center_y - height / (2 * self.scale),
            self.center_x + width / (2 * self.scale),
            self.center_y + height / (2 * self.scale),
        )

    def _view_covers_current(self, buffer_view, guard_ratio=0.0):
        if buffer_view is None or not math.isclose(
            buffer_view["scale"],
            self.scale,
            rel_tol=1e-12,
        ):
            return False
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        left, top, right, bottom = self._current_view_bounds(width, height)
        guard_x = (
            0.0
            if self.map_width * self.scale <= width
            else width * guard_ratio / self.scale
        )
        guard_y = (
            0.0
            if self.map_height * self.scale <= height
            else height * guard_ratio / self.scale
        )
        return (
            left >= buffer_view["left"] + guard_x
            and top >= buffer_view["top"] + guard_y
            and right <= buffer_view["right"] - guard_x
            and bottom <= buffer_view["bottom"] - guard_y
        )

    def _buffer_covers_current_view(self, guard_ratio=0.0):
        return self._view_covers_current(self._buffer_view, guard_ratio)

    def _maybe_prefetch(self, delay_ms=None):
        if self._buffer_covers_current_view(self.BUFFER_PREFETCH_GUARD_RATIO):
            return
        if (
            self._render_job is not None
            or self._future is not None
            or self._pending_task is not None
        ):
            return
        self._schedule_render(delay_ms=delay_ms)

    def _clear_render_buffer(self):
        self._photo = None
        self._buffer_view = None
        self._last_render_metrics = None

    def _draw_overlay(
        self,
        width,
        height,
        title,
        scale,
        sample_factors=(),
        using_overview=False,
        loading=False,
    ):
        if not title:
            return
        status = "Loading visible tiles..." if loading else ""
        if sample_factors and not loading:
            levels = ", ".join(f"1/{factor}" for factor in sample_factors)
            source = "Overview" if using_overview else "Tile"
            status = f"{source} sample: {levels}"
        metrics = self._last_render_metrics
        if metrics is not None and not loading:
            backend = metrics["render_backend"].upper()
            if metrics.get("render_device"):
                backend = f"{backend}: {metrics['render_device']}"
            quality = (
                "fast preview"
                if metrics.get("interactive", False)
                else "high quality"
            )
            timing = (
                f"Render: {metrics['render_ms']:.1f} ms + "
                f"Tk upload: {metrics.get('publish_ms', 0.0):.1f} ms "
                f"({backend}, {metrics['render_tile_count']} tiles, {quality})"
            )
            status = f"{status}\n{timing}" if status else timing
        if self.render_fallback_reason and self.render_backend == CPU_BACKEND:
            fallback = "OpenCL unavailable; using CPU"
            status = f"{status}\n{fallback}" if status else fallback
        status_line = f"{status}\n" if status else ""
        self.canvas.create_text(
            width / 2,
            height - 14,
            text=(
                f"{status_line}{title}  |  Zoom: {scale:.3g}x\n"
                "Mouse wheel: zoom  |  Drag: pan  |  Double-click: fit"
            ),
            fill="white",
            anchor="s",
            justify="center",
            width=max(1, width - 40),
            font=("TkDefaultFont", 10),
        )

    def _show_message(self, message):
        self._clear_render_buffer()
        self.canvas.delete("all")
        self.canvas.create_text(
            max(1, self.canvas.winfo_width()) // 2,
            max(1, self.canvas.winfo_height()) // 2,
            text=message,
            fill="#b0b0b0",
            font=("TkDefaultFont", 14),
        )

    def _clear_cache(self):
        with self._cache_lock:
            self._sampled_cache.clear()
            self._cache_bytes = 0
            self._gpu_cache.clear()
            self._gpu_cache_bytes = 0

    def _clear_gpu_cache(self):
        with self._cache_lock:
            self._gpu_cache.clear()
            self._gpu_cache_bytes = 0

    def _index_records(self):
        self._buckets = self._build_bucket_index(self.records)

    def _build_bucket_index(self, records, generation=None):
        buckets = {}
        for index, record in enumerate(records):
            if generation is not None and self._is_stale(generation):
                return None
            x1 = math.floor(record["x"] / self.BUCKET_SIZE)
            y1 = math.floor(record["y"] / self.BUCKET_SIZE)
            x2 = math.floor(
                (record["x"] + max(0.0, record["width"] - 1e-9))
                / self.BUCKET_SIZE
            )
            y2 = math.floor(
                (record["y"] + max(0.0, record["height"] - 1e-9))
                / self.BUCKET_SIZE
            )
            for bucket_y in range(y1, y2 + 1):
                for bucket_x in range(x1, x2 + 1):
                    buckets.setdefault((bucket_x, bucket_y), []).append(index)
        return buckets

    @staticmethod
    def _natural_path_key(path):
        return tuple(
            (1, int(part)) if part.isdigit() else (0, part.casefold())
            for part in re.split(r"(\d+)", str(path))
        )

    def _read_overview_record(self, overview_path, records, generation):
        if overview_path is None or not records or self._is_stale(generation):
            return None
        try:
            with Image.open(overview_path) as overview:
                pixel_width, pixel_height = overview.size
        except OSError:
            return None

        map_width = max(record["map_width"] for record in records)
        map_height = max(record["map_height"] for record in records)
        map_zoom = min(pixel_width / map_width, pixel_height / map_height)
        if map_zoom <= 0:
            return None
        return {
            "path": overview_path,
            "x": 0.0,
            "y": 0.0,
            "width": map_width,
            "height": map_height,
            "pixel_width": pixel_width,
            "pixel_height": pixel_height,
            "map_zoom": map_zoom,
            "map_width": map_width,
            "map_height": map_height,
        }

    def _visible_records(self, left, top, right, bottom):
        visible, _overflow = self._collect_visible_records(
            self.records,
            self._buckets,
            left,
            top,
            right,
            bottom,
        )
        return iter(visible or ())

    def _collect_visible_records(
        self,
        records,
        buckets,
        left,
        top,
        right,
        bottom,
        limit=None,
        generation=None,
    ):
        indices = set()
        seen = set()
        for bucket_y in range(
            math.floor(top / self.BUCKET_SIZE),
            math.floor(bottom / self.BUCKET_SIZE) + 1,
        ):
            for bucket_x in range(
                math.floor(left / self.BUCKET_SIZE),
                math.floor(right / self.BUCKET_SIZE) + 1,
            ):
                if generation is not None and self._is_stale(generation):
                    return None, False
                for index in buckets.get((bucket_x, bucket_y), ()):
                    if index in seen:
                        continue
                    seen.add(index)
                    record = records[index]
                    if not self._record_intersects(
                        record,
                        left,
                        top,
                        right,
                        bottom,
                    ):
                        continue
                    indices.add(index)
                    if limit is not None and len(indices) > limit:
                        return (), True
        return tuple(records[index] for index in sorted(indices)), False

    @staticmethod
    def _record_intersects(record, left, top, right, bottom):
        return not (
            record["x"] + record["width"] <= left
            or record["x"] >= right
            or record["y"] + record["height"] <= top
            or record["y"] >= bottom
        )

    @staticmethod
    def _record_screen_bounds(record, snapshot):
        left = max(
            0,
            int(round((record["x"] - snapshot["view_left"]) * snapshot["scale"])),
        )
        top = max(
            0,
            int(round((record["y"] - snapshot["view_top"]) * snapshot["scale"])),
        )
        right = min(
            snapshot["width"],
            int(
                round(
                    (record["x"] + record["width"] - snapshot["view_left"])
                    * snapshot["scale"]
                )
            ),
        )
        bottom = min(
            snapshot["height"],
            int(
                round(
                    (record["y"] + record["height"] - snapshot["view_top"])
                    * snapshot["scale"]
                )
            ),
        )
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    def _invalidate_work(self):
        self._generation += 1
        self._pending_task = None
        self._cancel_after("_render_job")
        return self._generation

    def _cancel_after(self, attribute):
        job = getattr(self, attribute)
        if job is None:
            return
        setattr(self, attribute, None)
        try:
            self.canvas.after_cancel(job)
        except (TclError, RuntimeError):
            pass

    def _is_stale(self, generation):
        return self._shutdown or generation != self._generation

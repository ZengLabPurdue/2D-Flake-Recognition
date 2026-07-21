import time
import threading
import json
import secrets
from queue import Full, Queue
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import HOME_DIR, PIXEL_SIZE, RESOLUTION_DIM, CROP_RATIO, RELATIVE_XY
from Imaging import image_metadata

from . import chip_detection
from . import flake_detection
from . import coordinate_generator


class ScanCancelled(RuntimeError):
    pass


class ScanManager:
    HUNDRED_X_GROUP_FIELD_RATIO = 0.75
    DEFAULT_WINDOWS = {
        "2x Scan": (5, 5),
        "10x Scan": (10, 10),
        "20x Scan": (22, 30),
        "Vignette Filter": (10, 10),
    }
    COMPLETE_SCAN_WINDOWS = {
        "Complete Scan (1 Chip)": (5, 5),
        "Full Stage Scan": (50, 25),
    }

    def __init__(
        self,
        root,
        app,
        stage,
        turret_controller,
        camera,
        frame_processor,
        mapper,
        update_scan_status,
        set_scan_running=None,
    ):
        self.root = root
        self.app = app
        self.stage = stage
        self.turret_controller = turret_controller
        self.camera = camera
        self.frame_processor = frame_processor
        self.mapper = mapper
        self.update_scan_status = update_scan_status
        self.set_scan_running = set_scan_running

        self.resolution = self.app.get_resolution()
        self.scan_running = False
        self.scan_metadata = None
        self._stop_event = threading.Event()

    def is_scan_running(self):
        return self.scan_running

    def stop_scan(self):
        if not self.scan_running:
            return
        self._stop_event.set()
        self.update_scan_status(stage="Stopping...")

    def run_scan(
        self,
        scan_type,
        window=None,
        material=None,
        substrate_thickness=None,
        full_scan_magnification="10x",
        detection_model="Region Detection",
        scan_profile=None,
    ):
        if self.scan_running:
            raise RuntimeError("A scan is already running.")

        self.scan_running = True
        self._stop_event.clear()
        self.resolution = self.app.get_resolution()
        if self.set_scan_running is not None:
            self.set_scan_running(True)

        profile_path = getattr(scan_profile, "path", None)
        if profile_path is not None:
            profile_path = Path(profile_path)
        selected_window = self.COMPLETE_SCAN_WINDOWS.get(
            scan_type,
            window or self.DEFAULT_WINDOWS.get(scan_type),
        )
        self.scan_metadata = {
            "scan_type": scan_type,
            "window": tuple(selected_window) if selected_window else None,
            "material": material,
            "substrate_thickness": substrate_thickness,
            "full_scan_magnification": full_scan_magnification,
            "detection_model": detection_model,
            "scan_profile_name": getattr(scan_profile, "name", None),
            "scan_profile_path": str(profile_path) if profile_path else None,
        }
        if hasattr(self.app, "set_region_map_available"):
            self.app.set_region_map_available(False)

        try:
            if scan_type in self.COMPLETE_SCAN_WINDOWS:
                if detection_model not in ("Flake Detection", "Region Detection"):
                    raise ValueError(f"Unknown detection model: {detection_model}")
                if detection_model == "Region Detection" and profile_path is None:
                    raise ValueError(
                        "A scan profile is required for region detection."
                    )
                if detection_model == "Region Detection":
                    profile_file = (
                        profile_path / "profile.json"
                        if profile_path.is_dir()
                        else profile_path
                    )
                    if not profile_file.is_file():
                        raise ValueError(
                            f"The selected scan profile was not found: {profile_file}"
                        )
                self.run_complete_scan(
                    window=self.COMPLETE_SCAN_WINDOWS[scan_type],
                    scan_magnification=full_scan_magnification,
                    detection_model=detection_model,
                    profile_path=profile_path,
                )
            elif scan_type == "2x Scan":
                self.run_2x_scan(
                    window=self._validate_window(
                        window or self.DEFAULT_WINDOWS[scan_type]
                    ),
                    full_zoom=True,
                )
            elif scan_type == "10x Scan":
                self.run_10x_scan(window=self._validate_window(
                    window or self.DEFAULT_WINDOWS[scan_type]
                ))
            elif scan_type == "20x Scan":
                self.run_20x_scan(window=self._validate_window(
                    window or self.DEFAULT_WINDOWS[scan_type]
                ))
            elif scan_type == "100x Scan":
                if detection_model == "Region Detection":
                    profile_file = (
                        profile_path / "profile.json"
                        if profile_path is not None and profile_path.is_dir()
                        else profile_path
                    )
                    if profile_file is None or not profile_file.is_file():
                        raise ValueError(
                            f"The selected scan profile was not found: {profile_file}"
                        )
                self.run_100x_scan(
                    detection_model=detection_model,
                    profile_path=profile_path,
                )
            elif scan_type == "Vignette Filter":
                self.create_vignette_filter(
                    window=self._validate_window(
                        window or self.DEFAULT_WINDOWS[scan_type]
                    )
                )
            else:
                raise ValueError(f"Unknown scan type: {scan_type}")

            self._check_cancelled()
            self.update_scan_status(stage="Complete", progress="100%")
            return True
        except ScanCancelled:
            self.update_scan_status(stage="Stopped", progress="Stopped")
            return False
        except Exception:
            self.update_scan_status(stage="Error", progress="Stopped")
            raise
        finally:
            self.scan_running = False
            if self.set_scan_running is not None:
                self.set_scan_running(False)

    def _check_cancelled(self):
        if self._stop_event.is_set():
            raise ScanCancelled("The scan was stopped.")

    def _process_ui_events(self):
        self.root.update()
        self._check_cancelled()

    def _queue_image(self, image_queue, image_data):
        while True:
            try:
                image_queue.put(image_data, timeout=0.1)
                return
            except Full:
                self.root.update()

    @staticmethod
    def _validate_window(window):
        if (
            not isinstance(window, (tuple, list))
            or len(window) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in window)
            or any(value < 1 for value in window)
        ):
            raise ValueError("Window width and height must be positive whole numbers.")
        return tuple(window)

    def run_complete_scan(
        self,
        window=(5, 5),
        scan_magnification="10x",
        detection_model="Region Detection",
        profile_path=None,
    ):
        self.app.set_live_mapping(False)
        self.app.open_panel("Scan Info Panel")
        region_mapping = detection_model == "Region Detection"
        if hasattr(self.app, "set_region_map_available"):
            self.app.set_region_map_available(region_mapping)
        if region_mapping and hasattr(self.app, "reset_region_map"):
            self.app.reset_region_map(
                "region-scan-pending",
                self.app.get_true_map().shape,
            )
        start_time = time.time()
        scan_path = HOME_DIR / "Scans" / datetime.now().strftime("Full Scan (%Y-%m-%d) (%H-%M-%S)")
        scan_magnification = scan_magnification.lower()
        if scan_magnification not in ("10x", "20x"):
            raise ValueError("Full scan magnification must be 10x or 20x.")

        self.update_scan_status(
            scan_type=f"Full Scan ({scan_magnification})",
            stage="2x Scan",
            progress="0%",
            stage_elapsed_time="00:00:00",
            total_elapsed_time="00:00:00",
        )
        center_x, center_y, scale_2x = self.run_2x_scan(
            scan_path=scan_path,
            full_scan=True,
            full_scan_start_time=start_time,
            window=window,
            full_zoom=True,
        )
        self._check_cancelled()

        true_map_bgr = cv2.cvtColor(self.app.get_true_map(), cv2.COLOR_RGB2BGR)
        filter_map = chip_detection.select_and_filter_map(
            map_image=true_map_bgr,
            save_path=scan_path / "Maps" / "map_2x_filtered.png",
            display=False,
        )
        self.app.set_filter_map(filter_map)
        self._check_cancelled()
        wafers, true_map = chip_detection.find_chips(filter_map, self.app.get_true_map())
        print("Scan areas found")
        self.app.set_true_map(true_map)
        print("True map set")

        coordinate_generator_function = (
            coordinate_generator.generate_10x_scan_coordinates
            if scan_magnification == "10x"
            else coordinate_generator.generate_20x_scan_coordinates
        )
        scan_coordinates = coordinate_generator_function(
            self.app,
            wafers,
            center_x,
            center_y,
            scale_2x,
            self.app.get_true_map(),
            self.camera.get_Size(),
        )
        print("Scan coordinates created")
        self._check_cancelled()

        image_queue = Queue(maxsize=200)
        print("Queue made")
        flake_detector = flake_detection.Flake_Detector()
        flake_detection_thread = threading.Thread(
            target=flake_detector.flake_detection,
            kwargs={
                "image_queue": image_queue,
                "frame_processor": self.frame_processor,
                "stop_requested": self._stop_event.is_set,
                "detection_model": detection_model,
                "profile_path": profile_path,
                "scan_path": scan_path,
            },
            daemon=True,
        )
        flake_detection_thread.start()
        print("Started flake detection thread")

        try:
            print(f"Running {scan_magnification}")
            scan_function = (
                self.run_10x_scan
                if scan_magnification == "10x"
                else self.run_20x_scan
            )
            scan_function(
                scan_coordinates,
                scan_path=scan_path,
                full_scan=True,
                full_scan_start_time=start_time,
                image_queue=image_queue,
                region_mapping=region_mapping,
            )
        finally:
            image_queue.put(None)
            while flake_detection_thread.is_alive():
                flake_detection_thread.join(timeout=0.1)
                if region_mapping and self.app.get_view() == "Map":
                    self.app.display_map()
                self.root.update()
            self._build_saved_scan_maps(scan_path, scan_magnification)
            self.turret_controller.change_objective(1)

        print("Full scan finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")

    @staticmethod
    def _metadata_number(metadata, key, number_type=int):
        value = metadata.get(key)
        if value is None:
            raise ValueError(f"Missing map metadata field: {key}")
        return number_type(float(value))

    def _compose_map_from_tiles(self, tile_paths):
        tile_records = []
        map_width = None
        map_height = None
        for tile_path in tile_paths:
            metadata = image_metadata.read_png_metadata(tile_path)
            try:
                record = {
                    "path": tile_path,
                    "map_x": self._metadata_number(metadata, "map_x"),
                    "map_y": self._metadata_number(metadata, "map_y"),
                    "map_zoom": max(
                        1,
                        self._metadata_number(metadata, "map_zoom"),
                    ),
                    "map_index": self._metadata_number(metadata, "map_index"),
                }
                record["map_id"] = metadata["map_id"]
                record["map_width"] = self._metadata_number(metadata, "map_width")
                record["map_height"] = self._metadata_number(metadata, "map_height")
            except (KeyError, TypeError, ValueError):
                continue
            map_width = record["map_width"] if map_width is None else map_width
            map_height = record["map_height"] if map_height is None else map_height
            tile_records.append(record)

        if not tile_records or map_width is None or map_height is None:
            return None, None

        result = np.zeros((map_height, map_width, 3), dtype=np.uint8)
        for record in tile_records:
            image = cv2.imread(str(record["path"]))
            if image is None:
                continue
            image = image[::record["map_zoom"], ::record["map_zoom"]]
            map_x = record["map_x"]
            map_y = record["map_y"]
            destination_x = max(0, map_x)
            destination_y = max(0, map_y)
            source_x = max(0, -map_x)
            source_y = max(0, -map_y)
            width = min(image.shape[1] - source_x, map_width - destination_x)
            height = min(image.shape[0] - source_y, map_height - destination_y)
            if width <= 0 or height <= 0:
                continue
            result[
                destination_y:destination_y + height,
                destination_x:destination_x + width,
            ] = image[
                source_y:source_y + height,
                source_x:source_x + width,
            ]
        return result, tile_records[0]["map_index"]

    def _build_saved_scan_maps(self, scan_path, magnification):
        """Rebuild raw and processed maps solely from saved tile metadata."""
        magnification = magnification.lower()
        image_root = Path(scan_path) / "All Images" / magnification
        maps_dir = Path(scan_path) / "Maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        for source_name, output_kind in (("Raw", ""), ("Processed", "_processed")):
            grouped_paths = {}
            for tile_path in image_root.glob(f"wafer */{source_name}/*.png"):
                metadata = image_metadata.read_png_metadata(tile_path)
                map_id = metadata.get("map_id")
                if map_id:
                    grouped_paths.setdefault(map_id, []).append(tile_path)

            for tile_paths in grouped_paths.values():
                map_image, map_index = self._compose_map_from_tiles(tile_paths)
                if map_image is None:
                    continue
                self.frame_processor.save_image(
                    image=map_image,
                    save_dir=maps_dir,
                    filename=(
                        f"map_{magnification}{output_kind}_wafer_{map_index}.png"
                    ),
                    vignette_applied=source_name == "Raw",
                    metadata={
                        "map_source": source_name.lower(),
                        "map_index": map_index,
                        "partial": bool(self._stop_event.is_set()),
                    },
                )

    def run_2x_scan(
        self,
        window=(5, 5),
        scan_path=None,
        zoom=6,
        full_scan=False,
        full_scan_start_time=None,
        full_zoom=False,
    ):
        self._check_cancelled()
        self.app.set_live_mapping(False)
        self.app.open_panel("Scan Info Panel")
        print("2x scan running...")

        if scan_path is None:
            path = HOME_DIR / "Scans" / datetime.now().strftime("2x (%Y-%m-%d) (%H-%M-%S)")
            map_save_dir = path / "Maps"
        else:
            path = scan_path / "All Images" / "2x"
            map_save_dir = scan_path / "Maps"

        return self.mapper.auto_map_2x(
            window=window,
            zoom=zoom,
            full_zoom=full_zoom,
            save_dir=path,
            map_save_dir=map_save_dir,
            full_scan=full_scan,
            full_scan_start_time=full_scan_start_time,
            check_cancelled=self._check_cancelled,
        )

    def run_10x_scan(
        self,
        scan_coordinates_10x=None,
        scan_path=None,
        image_queue=None,
        window=(10, 10),
        zoom=4,
        full_scan=False,
        full_scan_start_time=None,
        region_mapping=False,
    ):
        self._check_cancelled()
        self.app.set_live_mapping(False)

        self.app.open_panel("Scan Info Panel")

        start_time = time.time()

        self.app.set_view("Map")
        self.turret_controller.change_objective(2)
        self.stage.wait_until_not_busy()

        if scan_path is None:
            path = HOME_DIR / "Scans" / datetime.now().strftime("10x (%Y-%m-%d) (%H-%M-%S)")
            map_save_dir = path / "Maps"
        else:
            path = scan_path / "All Images" / "10x"
            map_save_dir = scan_path / "Maps"

        if scan_coordinates_10x is None:
            x, y, _ = self.stage.get_position()
            scan_coordinates_10x = [[x, y, window[0], window[1]]]

        for i, coordinates in enumerate(scan_coordinates_10x, start=1):
            self._check_cancelled()
            wafer_time = time.time()

            self.app.set_true_map(np.zeros((6000, 6000, 3), dtype=np.uint8))
            region_map_id = f"10x-wafer-{i}"
            if region_mapping and hasattr(self.app, "reset_region_map"):
                self.app.reset_region_map(
                    region_map_id,
                    self.app.get_true_map().shape,
                )

            if full_scan:
                self.update_scan_status(stage=f"10x Scan - Wafer {i} / {len(scan_coordinates_10x)}", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")
            else:
                self.update_scan_status(scan_type="10x Scan", stage="10x Scan", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")

            center_x = coordinates[0]
            center_y = coordinates[1]

            coords, total_frames = coordinate_generator.generate_rect_coords(coordinates[2], coordinates[3])

            self.stage.move_to_xy(center_x, center_y)
            print(f"Moving to: ({center_x}, {center_y})")

            self.stage.wait_until_not_busy()

            camera_width, camera_height = self.camera.get_Size()

            max_zoom = max(zoom, int(camera_height / (self.app.get_true_map().shape[0] / coordinates[3])), int(camera_width / (self.app.get_true_map().shape[1] / coordinates[2])))

            for j, (offset_x, offset_y) in enumerate(coords):
                self._check_cancelled()
                target_x = center_x + offset_x * PIXEL_SIZE["10X"][self.resolution] * RESOLUTION_DIM[self.resolution]["x"] * CROP_RATIO["10X"]["x"]
                target_y = center_y - offset_y * PIXEL_SIZE["10X"][self.resolution] * RESOLUTION_DIM[self.resolution]["y"] * CROP_RATIO["10X"]["y"]

                self.stage.move_to_xy(target_x, target_y)
                self.stage.wait_until_not_busy()

                img = self.frame_processor.capture_frame()

                img = self.frame_processor.apply_vignette_filter(img)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                true_map = self.app.get_true_map()
                map_x = int(true_map.shape[1] / 2 - (offset_x + 0.5) * img_rgb.shape[1] / max_zoom)
                map_y = int(true_map.shape[0] / 2 + (offset_y - 0.5) * img_rgb.shape[0] / max_zoom)
                img_small = img_rgb[::max_zoom, ::max_zoom]

                image_path = path / f"wafer {i} ({center_x}, {center_y})" / "Raw" / f"img_10x_{j}.png"

                image_path.parent.mkdir(parents=True, exist_ok=True)

                self.frame_processor.save_image(
                    image=img,
                    filename=image_path,
                    vignette_applied=True,
                    metadata={
                        "map_id": region_map_id,
                        "map_index": i,
                        "map_x": map_x,
                        "map_y": map_y,
                        "map_center_x": map_x + img_small.shape[1] / 2,
                        "map_center_y": map_y + img_small.shape[0] / 2,
                        "map_zoom": max_zoom,
                        "map_width": int(true_map.shape[1]),
                        "map_height": int(true_map.shape[0]),
                        "stage_x_um": float(target_x),
                        "stage_y_um": float(target_y),
                    },
                )

                if self.app.get_view() == "Camera View":
                    if self.app.get_filter():
                        self.app.set_filter(False)

                    self.root.after(0, lambda img=img: self.app.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

                if image_queue is not None:
                    self._queue_image(image_queue, {
                        "path": image_path,
                        "stage_x": float(target_x),
                        "stage_y": float(target_y),
                        "magnification": "10X",
                        "pixel_size_um": float(PIXEL_SIZE["10X"][self.resolution]),
                        "region_map_id": region_map_id,
                        "map_x": map_x,
                        "map_y": map_y,
                        "map_zoom": max_zoom,
                    })

                x_start = max(0, map_x)
                y_start = max(0, map_y)
                x_end = min(true_map.shape[1], x_start + img_small.shape[1])
                y_end = min(true_map.shape[0], y_start + img_small.shape[0])

                true_map[y_start:y_end, x_start:x_end] = img_small[:y_end - y_start, :x_end - x_start,]

                self.app.display_map()

                stage_elapsed = time.time() - wafer_time

                if full_scan_start_time is not None:
                    total_elapsed = time.time() - full_scan_start_time
                    total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
                else:
                    total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stage_elapsed))

                stage_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stage_elapsed))
                progress_percent = f"{j + 1}/{total_frames} ({(j + 1) * 100 // total_frames}%)"

                if full_scan:
                    self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=total_elapsed_str)
                else:
                    self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=stage_elapsed_str)

                self._process_ui_events()

            map_bgr = cv2.cvtColor(self.app.get_true_map(), cv2.COLOR_RGB2BGR)
            self.frame_processor.save_image(
                image=map_bgr,
                save_dir=map_save_dir,
                filename=f"map_10x_wafer_{i}.png",
                vignette_applied=True,
            )

            print(f"wafer {i} imaging finished!")
            print(f"Time taken: {time.time() - wafer_time:.2f}s")

        print("10x scan imaging finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")

    def run_20x_scan(
        self,
        scan_coordinates_20x=None,
        scan_path=None,
        image_queue=None,
        window=(22, 30),
        zoom=4,
        full_scan=False,
        full_scan_start_time=None,
        region_mapping=False,
    ):
        self._check_cancelled()
        self.app.set_live_mapping(False)

        self.app.open_panel("Scan Info Panel")

        start_time = time.time()

        self.app.set_view("Map")
        self.turret_controller.change_objective(2)
        self.turret_controller.change_objective(3)

        if scan_path is None:
            path = HOME_DIR / "Scans" / datetime.now().strftime("20x (%Y-%m-%d) (%H-%M-%S)")
            map_save_dir = path / "Maps"
        else:
            path = scan_path / "All Images" / "20x"
            map_save_dir = scan_path / "Maps"

        if scan_coordinates_20x is None:
            x, y, _ = self.stage.get_position()
            scan_coordinates_20x = [[x, y, window[0], window[1]]]

        for i, coordinates in enumerate(scan_coordinates_20x, start=1):
            self._check_cancelled()
            wafer_time = time.time()

            self.app.set_true_map(np.zeros((6000, 6000, 3), dtype=np.uint8))
            region_map_id = f"20x-wafer-{i}"
            if region_mapping and hasattr(self.app, "reset_region_map"):
                self.app.reset_region_map(
                    region_map_id,
                    self.app.get_true_map().shape,
                )

            if full_scan:
                self.update_scan_status(stage=f"20x Scan - Wafer {i} / {len(scan_coordinates_20x)}", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")
            else:
                self.update_scan_status(scan_type="20x Scan", stage="20x Scan", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")

            center_x = coordinates[0]
            center_y = coordinates[1]

            coords, total_frames = coordinate_generator.generate_rect_coords(coordinates[2], coordinates[3])

            self.stage.move_to_xy(center_x, center_y)
            print(f"Moving to: ({center_x}, {center_y})")
            self.stage.wait_until_not_busy()

            camera_width, camera_height = self.camera.get_Size()

            max_zoom = max(zoom, int(camera_height / (self.app.get_true_map().shape[0] / coordinates[3])), int(camera_width / (self.app.get_true_map().shape[1] / coordinates[2])))

            for j, (offset_x, offset_y) in enumerate(coords):
                self._check_cancelled()
                target_x = center_x + offset_x * PIXEL_SIZE["20X"][self.resolution] * RESOLUTION_DIM[self.resolution]["x"] * CROP_RATIO["20X"]["x"]
                target_y = center_y - offset_y * PIXEL_SIZE["20X"][self.resolution] * RESOLUTION_DIM[self.resolution]["y"] * CROP_RATIO["20X"]["y"]

                self.stage.move_to_xy(target_x, target_y)
                self.stage.wait_until_not_busy()

                img = self.frame_processor.capture_frame()

                img = self.frame_processor.apply_vignette_filter(img)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                true_map = self.app.get_true_map()
                map_x = int(true_map.shape[1] / 2 - (offset_x + 0.5) * img_rgb.shape[1] / max_zoom)
                map_y = int(true_map.shape[0] / 2 + (offset_y - 0.5) * img_rgb.shape[0] / max_zoom)
                img_small = img_rgb[::max_zoom, ::max_zoom]

                image_path = path / f"wafer {i} ({center_x}, {center_y})" / "Raw" / f"img_20x_{j}.png"

                image_path.parent.mkdir(parents=True, exist_ok=True)

                self.frame_processor.save_image(
                    image=img,
                    filename=image_path,
                    vignette_applied=True,
                    metadata={
                        "map_id": region_map_id,
                        "map_index": i,
                        "map_x": map_x,
                        "map_y": map_y,
                        "map_center_x": map_x + img_small.shape[1] / 2,
                        "map_center_y": map_y + img_small.shape[0] / 2,
                        "map_zoom": max_zoom,
                        "map_width": int(true_map.shape[1]),
                        "map_height": int(true_map.shape[0]),
                        "stage_x_um": float(target_x),
                        "stage_y_um": float(target_y),
                    },
                )

                if self.app.get_view() == "Camera View":
                    if self.app.get_filter():
                        self.app.set_filter(False)

                    self.root.after(0, lambda img=img: self.app.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

                if image_queue is not None:
                    self._queue_image(image_queue, {
                        "path": image_path,
                        "stage_x": float(target_x),
                        "stage_y": float(target_y),
                        "magnification": "20X",
                        "pixel_size_um": float(PIXEL_SIZE["20X"][self.resolution]),
                        "region_map_id": region_map_id,
                        "map_x": map_x,
                        "map_y": map_y,
                        "map_zoom": max_zoom,
                    })

                x_start = max(0, map_x)
                y_start = max(0, map_y)
                x_end = min(true_map.shape[1], x_start + img_small.shape[1])
                y_end = min(true_map.shape[0], y_start + img_small.shape[0])

                true_map[y_start:y_end, x_start:x_end] = img_small[:y_end - y_start, :x_end - x_start,]

                self.app.display_map()

                stage_elapsed = time.time() - wafer_time

                if full_scan_start_time is not None:
                    total_elapsed = time.time() - full_scan_start_time
                    total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
                else:
                    total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stage_elapsed))

                stage_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stage_elapsed))
                progress_percent = f"{j + 1}/{total_frames} ({(j + 1) * 100 // total_frames}%)"

                if full_scan:
                    self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=total_elapsed_str)
                else:
                    self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=stage_elapsed_str)

                self._process_ui_events()

            map_bgr = cv2.cvtColor(self.app.get_true_map(), cv2.COLOR_RGB2BGR)
            self.frame_processor.save_image(
                image=map_bgr,
                save_dir=map_save_dir,
                filename=f"map_20x_wafer_{i}.png",
                vignette_applied=True,
            )

            print(f"Wafer {i} imaging finished!")
            print(f"Time taken: {time.time() - wafer_time:.2f}s")

        print("20x scan imaging finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")

    @staticmethod
    def _group_nearby_regions(regions, threshold_um):
        """Group nearby targets with an average O(n + k) spatial-hash pass."""
        if threshold_um <= 0:
            raise ValueError("The 100x grouping threshold must be positive.")
        if not regions:
            return []

        parents = list(range(len(regions)))
        ranks = [0] * len(regions)

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if ranks[left_root] < ranks[right_root]:
                left_root, right_root = right_root, left_root
            parents[right_root] = left_root
            if ranks[left_root] == ranks[right_root]:
                ranks[left_root] += 1

        threshold_squared = threshold_um * threshold_um
        grid = {}
        for index, region in enumerate(regions):
            target = region["target_position_10x_um"]
            x = float(target["x"])
            y = float(target["y"])
            cell = (
                int(np.floor(x / threshold_um)),
                int(np.floor(y / threshold_um)),
            )
            for cell_y in range(cell[1] - 1, cell[1] + 2):
                for cell_x in range(cell[0] - 1, cell[0] + 2):
                    for other_index in grid.get((cell_x, cell_y), ()):
                        other = regions[other_index]["target_position_10x_um"]
                        delta_x = x - float(other["x"])
                        delta_y = y - float(other["y"])
                        if delta_x * delta_x + delta_y * delta_y <= threshold_squared:
                            union(index, other_index)
            grid.setdefault(cell, []).append(index)

        components = {}
        for index in range(len(regions)):
            components.setdefault(find(index), []).append(index)

        groups = []
        for member_indices in sorted(components.values(), key=lambda items: items[0]):
            target_x_values = [
                float(regions[index]["target_position_10x_um"]["x"])
                for index in member_indices
            ]
            target_y_values = [
                float(regions[index]["target_position_10x_um"]["y"])
                for index in member_indices
            ]
            groups.append({
                "member_indices": member_indices,
                "target_position_10x_um": {
                    "x": (min(target_x_values) + max(target_x_values)) / 2,
                    "y": (min(target_y_values) + max(target_y_values)) / 2,
                },
            })
        return groups

    def run_100x_scan(
        self,
        detection_model="Region Detection",
        profile_path=None,
        group_threshold_um=None,
    ):
        """Find regions at 10x, navigate at 20x, and capture each one at 100x."""
        if detection_model not in ("Flake Detection", "Region Detection"):
            raise ValueError(f"Unknown detection model: {detection_model}")
        if detection_model == "Region Detection" and profile_path is None:
            raise ValueError("A scan profile is required for region detection.")

        self._check_cancelled()
        self.app.set_live_mapping(False)
        self.app.set_view("Camera View")
        self.app.open_panel("Scan Info Panel")
        start_time = time.time()
        scan_path = HOME_DIR / "Scans" / datetime.now().strftime(
            "100x Scan (%Y-%m-%d) (%H-%M-%S)"
        )
        source_dir = scan_path / "All Images" / "10x"
        capture_dir = scan_path / "All Images" / "100x"
        processed_dir = scan_path / "Processed" / "10x"

        self.update_scan_status(
            scan_type="100x Scan",
            stage="Finding regions at 10x",
            progress="0%",
            stage_elapsed_time="00:00:00",
            total_elapsed_time="00:00:00",
        )

        self.turret_controller.change_objective(2)
        self.stage.wait_until_not_busy()
        self._check_cancelled()
        source_x, source_y, source_z = self.stage.get_position()
        source_image = self.frame_processor.capture_frame()
        if source_image is None:
            raise RuntimeError("The camera did not return a 10x image.")
        source_image = self.frame_processor.apply_vignette_filter(source_image)
        source_path = self.frame_processor.save_image(
            image=source_image,
            save_dir=source_dir,
            filename="source_10x.png",
            vignette_applied=True,
        )

        detector = flake_detection.Flake_Detector()
        image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        color_seed = None
        if detection_model == "Region Detection":
            color_seed = secrets.randbits(32)
            annotated_rgb, detections, _ = (
                detector.flake_identifier.identify_flakes_region_model(
                    image_rgb,
                    profile_path,
                    color_seed=color_seed,
                )
            )
        else:
            annotated_rgb, all_detections, _ = (
                detector.flake_identifier.identify_flakes_flake_model(image_rgb)
            )
            detections = [
                detection for detection in all_detections if int(detection[0]) == 1
            ]

        self.frame_processor.save_image(
            image=cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR),
            save_dir=processed_dir,
            filename="source_10x.png",
            vignette_applied=True,
        )

        image_height, image_width = image_rgb.shape[:2]
        pixel_size = float(PIXEL_SIZE["10X"][self.resolution])
        regions = []
        for detection in detections:
            if detection_model == "Region Detection":
                classification = detection.get("matched_class")
                bounding_box = detection.get("bounding_box")
            else:
                classification = flake_detection.FLAKE_CLASSIFICATIONS[1]
                bounding_box = detection[1]
            bounded_box = detector._bounded_box(bounding_box, image_rgb.shape)
            if classification is None or bounded_box is None:
                continue
            x, y, width, height = bounded_box
            center_x = x + width / 2
            center_y = y + height / 2
            regions.append({
                "classification": classification,
                "bounding_box_px": {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "width": round(width, 3),
                    "height": round(height, 3),
                },
                "bounding_box_center_px": {
                    "x": round(center_x, 3),
                    "y": round(center_y, 3),
                },
                "size_um": {
                    "width": round(width * pixel_size, 6),
                    "height": round(height * pixel_size, 6),
                    "bounding_box_area": round(width * height * pixel_size ** 2, 6),
                },
                "target_position_10x_um": {
                    "x": round(source_x - (center_x - image_width / 2) * pixel_size, 6),
                    "y": round(source_y - (center_y - image_height / 2) * pixel_size, 6),
                },
            })

        if group_threshold_um is None:
            camera_width, camera_height = self.camera.get_Size()
            field_width_um = (
                PIXEL_SIZE["100X"][self.resolution]
                * camera_width
                * CROP_RATIO["100X"]["x"]
            )
            field_height_um = (
                PIXEL_SIZE["100X"][self.resolution]
                * camera_height
                * CROP_RATIO["100X"]["y"]
            )
            group_threshold_um = (
                min(field_width_um, field_height_um)
                * self.HUNDRED_X_GROUP_FIELD_RATIO
            )
        group_threshold_um = float(group_threshold_um)
        capture_groups = self._group_nearby_regions(regions, group_threshold_um)

        if capture_groups:
            self.update_scan_status(stage="Capturing regions at 100x", progress="0%")
            self.turret_controller.change_objective(3)
            self.stage.wait_until_not_busy()
            navigation_offset_x = RELATIVE_XY["20X"]["X"] - RELATIVE_XY["10X"]["X"]
            navigation_offset_y = RELATIVE_XY["20X"]["Y"] - RELATIVE_XY["10X"]["Y"]

            for index, capture_group in enumerate(capture_groups, start=1):
                self._check_cancelled()
                target = capture_group["target_position_10x_um"]
                navigation_x = target["x"] + navigation_offset_x
                navigation_y = target["y"] + navigation_offset_y
                self.stage.move_to_xy(navigation_x, navigation_y)
                self.stage.wait_until_not_busy()
                navigation_position = {
                    "x": round(navigation_x, 6),
                    "y": round(navigation_y, 6),
                }

                try:
                    self.turret_controller.change_objective(5)
                    self.stage.wait_until_not_busy()
                    capture_x, capture_y, capture_z = self.stage.get_position()
                    image_100x = self.frame_processor.capture_frame()
                    if image_100x is None:
                        raise RuntimeError(
                            f"The camera did not return 100x image {index}."
                        )
                    filename = f"flake_group_{index:04d}.png"
                    saved_path = self.frame_processor.save_image(
                        image=image_100x,
                        save_dir=capture_dir,
                        filename=filename,
                        vignette_applied=False,
                    )
                finally:
                    self.turret_controller.change_objective(3)
                    self.stage.wait_until_not_busy()

                relative_image_path = Path(saved_path).relative_to(scan_path).as_posix()
                capture_position = {
                    "x": round(float(capture_x), 6),
                    "y": round(float(capture_y), 6),
                    "z": round(float(capture_z), 6),
                }
                capture_group.update({
                    "group_id": index,
                    "navigation_position_20x_um": navigation_position,
                    "capture_position_100x_um": capture_position,
                    "image_100x": relative_image_path,
                    "region_indices": [
                        member_index + 1
                        for member_index in capture_group["member_indices"]
                    ],
                })
                for member_index in capture_group["member_indices"]:
                    regions[member_index].update({
                        "capture_group_id": index,
                        "navigation_position_20x_um": navigation_position,
                        "capture_position_100x_um": capture_position,
                        "image_100x": relative_image_path,
                    })
                elapsed = time.time() - start_time
                self.update_scan_status(
                    progress=(
                        f"{index}/{len(capture_groups)} "
                        f"({index * 100 // len(capture_groups)}%)"
                    ),
                    stage_elapsed_time=time.strftime("%H:%M:%S", time.gmtime(elapsed)),
                    total_elapsed_time=time.strftime("%H:%M:%S", time.gmtime(elapsed)),
                )
                self._process_ui_events()

        manifest = {
            "schema": "flake-search.100x-scan",
            "version": 1,
            "detection_model": detection_model,
            "profile_path": str(profile_path) if profile_path is not None else None,
            "region_color_seed": color_seed,
            "source_image": Path(source_path).relative_to(scan_path).as_posix(),
            "source_capture_position_10x_um": {
                "x": round(float(source_x), 6),
                "y": round(float(source_y), 6),
                "z": round(float(source_z), 6),
            },
            "resolution": self.resolution,
            "pixel_size_10x_um": pixel_size,
            "vignette_applied_100x": False,
            "group_threshold_um": round(group_threshold_um, 6),
            "flake_count": len(regions),
            "capture_count": len(capture_groups),
            "capture_groups": [
                {
                    key: value
                    for key, value in capture_group.items()
                    if key != "member_indices"
                }
                for capture_group in capture_groups
            ],
            "flakes": regions,
        }
        scan_path.mkdir(parents=True, exist_ok=True)
        (scan_path / "flakes_found.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(f"100x scan saved to: {scan_path}")

    def create_vignette_filter(self, window=(10, 10), sigma=40, output=False):
        self._check_cancelled()
        magnification = self.app.get_magnification()
        self.app.open_panel("Scan Info Panel")
        print("Creating vignette filter...")
        start_time = time.time()
        folder_path = HOME_DIR / "Saved Images" / "Vignette Filter"
        img_paths = []

        import shutil
        if folder_path.exists():
            shutil.rmtree(folder_path)

        center_x, center_y, _ = self.stage.get_position()
        coords, total_frames = coordinate_generator.generate_rect_coords(*window)
        self.update_scan_status(scan_type="Vignette Filter", stage="Vignette Filter", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")
        step_x = 300
        step_y = 300

        if magnification == "2X":
            pass
        elif magnification == "10X":
            step_x /= 5
            step_y /= 5
        elif magnification == "20X":
            step_x /= 2
            step_y /= 2
        elif magnification == "100X":
            pass

        try:
            for i, (offset_x, offset_y) in enumerate(coords, start=1):
                self._check_cancelled()
                target_x = center_x + offset_x * step_x
                target_y = center_y - offset_y * step_y

                self.stage.move_to_xy(target_x, target_y)
                self.stage.wait_until_not_busy()

                img = self.frame_processor.capture_frame_raw(num_images=10)
                if img is not None and output:
                    print(f"Captured Image {i}/{total_frames}")

                img_path = folder_path / f"img_{i}.png"
                img_paths.append(img_path)
                self.frame_processor.save_image(
                    image=img,
                    save_dir=folder_path,
                    filename=f"img_{i}.png",
                )

                stage_elapsed = time.time() - start_time
                stage_elapsed_str = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(stage_elapsed),
                )
                progress_percent = (
                    f"{i}/{total_frames} ({i * 100 // total_frames}%)"
                )
                self.update_scan_status(
                    progress=progress_percent,
                    stage_elapsed_time=stage_elapsed_str,
                    total_elapsed_time=stage_elapsed_str,
                )
                self._process_ui_events()
        finally:
            self.stage.move_to_xy(center_x, center_y)

        print("Vignette filter imaging finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        sum_img = None
        count = 0

        for path in img_paths:
            self._check_cancelled()
            img = cv2.imread(str(path))

            if img is None:
                print(f"Skipping unreadable image: {path}")
                continue
            
            img_float = img.astype(np.float64)

            if sum_img is None:
                sum_img = np.zeros_like(img_float)

            sum_img += img_float
            count += 1

            self._process_ui_events()

        if count == 0:
            raise RuntimeError(f"No valid images found in {folder_path}")

        avg_img = sum_img / count

        b, g, r = cv2.split(avg_img)

        b_blur = cv2.GaussianBlur(b, (0, 0), sigmaX=sigma, sigmaY=sigma)
        g_blur = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
        r_blur = cv2.GaussianBlur(r, (0, 0), sigmaX=sigma, sigmaY=sigma)

        vignette_filter_bgr = cv2.merge([b_blur, g_blur, r_blur])

        vignette_filter = np.clip(vignette_filter_bgr, 0, 255).astype(np.uint8)

        save_path = HOME_DIR / "Flatfields" / f"vignette_filter_{magnification.lower()}_{self.app.get_resolution().lower()}.png"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), vignette_filter):
            raise OSError(f"Could not save vignette filter to {save_path}")
        self.frame_processor.clear_vignette_filter_cache()

        print(f"Saved vignette image to: {save_path}")

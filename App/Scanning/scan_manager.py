import time
import threading
from queue import Full, Queue
from datetime import datetime

import cv2
import numpy as np

from config import HOME_DIR, PIXEL_SIZE, RESOLUTION_DIM, CROP_RATIO

from . import chip_detection
from . import flake_detection
from . import coordinate_generator


class ScanCancelled(RuntimeError):
    pass


class ScanManager:
    DEFAULT_WINDOWS = {
        "2x Scan": (3, 3),
        "10x Scan": (10, 10),
        "20x Scan": (22, 30),
        "Vignette Filter": (10, 10),
    }
    COMPLETE_SCAN_WINDOWS = {
        "Complete Scan (1 Chip)": (3, 3),
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
        detection_model="Flake Detection",
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

        try:
            if scan_type in self.COMPLETE_SCAN_WINDOWS:
                if detection_model != "Flake Detection":
                    raise ValueError("Region detection is not connected yet.")
                self.run_complete_scan(
                    window=self.COMPLETE_SCAN_WINDOWS[scan_type],
                    scan_magnification=full_scan_magnification,
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

    def _queue_image(self, image_queue, image_path):
        while True:
            self._check_cancelled()
            try:
                image_queue.put(image_path, timeout=0.1)
                return
            except Full:
                self._process_ui_events()

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

    def run_complete_scan(self, window=(3, 3), scan_magnification="10x"):
        self.app.set_live_mapping(False)
        self.app.open_panel("Scan Info Panel")
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
            save_path=scan_path / "All Images" / "2x" / "map_2x_filtered.png",
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
            )
        finally:
            image_queue.put(None)
            while flake_detection_thread.is_alive():
                flake_detection_thread.join(timeout=0.1)
                self.root.update()
            self.turret_controller.change_objective(1)

        print("Full scan finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")

    def run_2x_scan(
        self,
        window=(3, 3),
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
        else:
            path = scan_path / "All Images" / "2x"

        return self.mapper.auto_map_2x(
            window=window,
            zoom=zoom,
            full_zoom=full_zoom,
            save_dir=path,
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
        else:
            path = scan_path / "All Images" / "10x"

        if scan_coordinates_10x is None:
            x, y, _ = self.stage.get_position()
            scan_coordinates_10x = [[x, y, window[0], window[1]]]

        for i, coordinates in enumerate(scan_coordinates_10x, start=1):
            self._check_cancelled()
            wafer_time = time.time()

            self.app.set_true_map(np.zeros((6000, 6000, 3), dtype=np.uint8))

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

                image_path = path / f"wafer {i} ({center_x}, {center_y})" / "Raw" / f"img_10x_{j}.png"

                image_path.parent.mkdir(parents=True, exist_ok=True)

                self.frame_processor.save_image(
                    image=img,
                    filename=image_path,
                    vignette_applied=True,
                )

                if image_queue is not None:
                    self._queue_image(image_queue, image_path)

                if self.app.get_view() == "Camera View":
                    if self.app.get_filter():
                        self.app.set_filter(False)

                    self.root.after(0, lambda img=img: self.app.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

                true_map = self.app.get_true_map()

                map_x = int(true_map.shape[1] / 2 - (offset_x + 0.5) * img_rgb.shape[1] / max_zoom)
                map_y = int(true_map.shape[0] / 2 + (offset_y - 0.5) * img_rgb.shape[0] / max_zoom)

                img_small = img_rgb[::max_zoom, ::max_zoom]

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
        else:
            path = scan_path / "All Images" / "20x"

        if scan_coordinates_20x is None:
            x, y, _ = self.stage.get_position()
            scan_coordinates_20x = [[x, y, window[0], window[1]]]

        for i, coordinates in enumerate(scan_coordinates_20x, start=1):
            self._check_cancelled()
            wafer_time = time.time()

            self.app.set_true_map(np.zeros((6000, 6000, 3), dtype=np.uint8))

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

                image_path = path / f"wafer {i} ({center_x}, {center_y})" / "Raw" / f"img_20x_{j}.png"

                image_path.parent.mkdir(parents=True, exist_ok=True)

                self.frame_processor.save_image(
                    image=img,
                    filename=image_path,
                    vignette_applied=True,
                )

                if image_queue is not None:
                    self._queue_image(image_queue, image_path)

                if self.app.get_view() == "Camera View":
                    if self.app.get_filter():
                        self.app.set_filter(False)

                    self.root.after(0, lambda img=img: self.app.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

                true_map = self.app.get_true_map()

                map_x = int(true_map.shape[1] / 2 - (offset_x + 0.5) * img_rgb.shape[1] / max_zoom)
                map_y = int(true_map.shape[0] / 2 + (offset_y - 0.5) * img_rgb.shape[0] / max_zoom)

                img_small = img_rgb[::max_zoom, ::max_zoom]

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

            print(f"Wafer {i} imaging finished!")
            print(f"Time taken: {time.time() - wafer_time:.2f}s")

        print("20x scan imaging finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")

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

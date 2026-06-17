import time
import threading
from queue import Queue
from datetime import datetime

import cv2
import numpy as np

from config import HOME_DIR, PIXEL_SIZE, RESOLUTION_DIM, CROP_RATIO, FLATFIELD_IMG
from Imaging import vignetting_corrector

from . import wafer_detection
from . import flake_detection
from . import coordinate_generator

class ScanManager:
    def __init__(
        self,
        root,
        app,
        stage,
        turret_controller,
        camera,
        frame_processor,
        update_scan_status,
    ):
        self.root = root
        self.app = app
        self.stage = stage
        self.turret_controller = turret_controller
        self.camera = camera
        self.frame_processor = frame_processor
        self.update_scan_status = update_scan_status

        self.resolution = self.app.get_resolution()

    def run_complete_scan(self, window=(3, 3)):
        self.app.open_panel("Info Panel")

        start_time = time.time()

        scan_path = HOME_DIR / "Scans" / datetime.now().strftime("Full Scan (%Y-%m-%d) (%H-%M-%S)")

        self.update_scan_status(scan_type="Full Scan")

        center_x, center_y, scale_2x = self.run_2x_scan(scan_path=scan_path, full_scan=True, full_scan_start_time=start_time, window=window, full_zoom=True)
        wafers, true_map = wafer_detection.find_wafers(self.app.get_filter_map(), self.app.get_true_map())
        print("Wafers found")
        self.app.set_true_map(true_map)
        print("True map set")
        scan_coordinates = coordinate_generator.generate_10x_scan_coordinates(self.app, wafers, center_x, center_y, scale_2x, self.app.get_true_map(), self.camera.get_Size())
        print("Scan coordinates created")

        image_queue = Queue(maxsize=200)
        print("Queue made")

        flake_detection_thread = threading.Thread(
            target=flake_detection.flake_detection_10x,
            kwargs={
                "image_queue": image_queue,
                "frame_processor" : self.frame_processor
            },
            daemon=True,
        )

        flake_detection_thread.start()
        print("Started flake detection thread")

        print("Running 10x")
        self.run_10x_scan(scan_coordinates, scan_path=scan_path, full_scan=True, full_scan_start_time=start_time, image_queue=image_queue)

        image_queue.put(None)
        flake_detection_thread.join()

        print("Full scan finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")

        self.stage.move_to_xy(0, 0)
        self.turret_controller.change_objective(1)

    def run_2x_scan(
        self,
        window=(3, 3),
        scan_path=None,
        zoom=6,
        full_scan=False,
        full_scan_start_time=None,
        full_zoom=False,
    ):
        self.app.open_panel("Info Panel")

        print("2x scan running...")

        self.turret_controller.change_objective(1)
        self.app.set_view("Map", True)

        start_time = time.time()

        if scan_path is None:
            path = HOME_DIR / "Scans" / datetime.now().strftime("2x (%Y-%m-%d) (%H-%M-%S)")
        else:
            path = scan_path / "All Images" / "2x"

        self.app.set_true_map(np.zeros((3000, 3000, 3), dtype=np.uint8))
        self.app.set_filter_map(np.zeros((3000, 3000), dtype=np.uint8))

        center_x, center_y, _ = self.stage.get_position()

        coords, total_frames = coordinate_generator.generate_rect_coords(window[1], window[0])

        camera_width, camera_height = self.camera.get_Size()

        zoom = max(zoom, int(camera_height / (self.app.get_true_map().shape[0] / window[1])), int(camera_width / (self.app.get_true_map().shape[1] / window[0])))

        if full_zoom:
            zoom = max(int(camera_height / (self.app.get_true_map().shape[0] / window[1])), int(camera_width / (self.app.get_true_map().shape[1] / window[0])))

        if full_scan:
            self.update_scan_status(stage="2x Scan", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")
        else:
            self.update_scan_status(scan_type="2x Scan", stage="2x Scan", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")

        for i, (offset_x, offset_y) in enumerate(coords, start=1):
            target_x = (center_x + offset_x * PIXEL_SIZE["2X"][self.resolution] * RESOLUTION_DIM[self.resolution]["x"] * CROP_RATIO["2X"]["x"])
            target_y = (center_y - offset_y * PIXEL_SIZE["2X"][self.resolution] * RESOLUTION_DIM[self.resolution]["y"] * CROP_RATIO["2X"]["y"])

            self.stage.move_to_xy(target_x, target_y)
            self.stage.wait_until_not_busy()

            img = self.frame_processor.capture_frame()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image_path = path / "Raw" / f"img_2x_{i}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)

            self.frame_processor.save_image(image=img, filename=image_path)

            binary = wafer_detection.wafer_filter(img, display=False)
            img_binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

            if self.app.get_view() == "Camera View":
                if self.app.get_filter():
                    self.app.display_image(img_binary_rgb)
                else:
                    self.app.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            filter_map = self.app.get_filter_map()
            true_map = self.app.get_true_map()

            map_x = int(filter_map.shape[1] / 2 - (offset_x + 0.5) * img_binary_rgb.shape[1] / zoom)
            map_y = int(filter_map.shape[0] / 2 + (offset_y - 0.5) * img_binary_rgb.shape[0] / zoom)

            img_small = img_rgb[::zoom, ::zoom]
            img_binary_small = img_binary_rgb[::zoom, ::zoom, 0]

            x_start = max(0, map_x)
            y_start = max(0, map_y)
            x_end = min(filter_map.shape[1], x_start + img_binary_small.shape[1])
            y_end = min(filter_map.shape[0], y_start + img_binary_small.shape[0])

            true_map[y_start:y_end, x_start:x_end] = img_small[:y_end-y_start, :x_end-x_start]

            filter_map[y_start:y_end, x_start:x_end] = img_binary_small[:y_end-y_start, :x_end-x_start]

            stage_elapsed = time.time() - start_time

            if full_scan_start_time is not None:
                total_elapsed = time.time() - full_scan_start_time
                total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
            else:
                total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stage_elapsed))

            stage_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stage_elapsed))
            progress_percent = f"{i}/{total_frames} ({i * 100 // total_frames}%)"

            if full_scan:
                self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=total_elapsed_str)
            else:
                self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=stage_elapsed_str)

        self.stage.move_to_xy(center_x, center_y)

        print("2x scan imaging finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")

        return center_x, center_y, zoom

    def run_10x_scan(
        self,
        scan_coordinates_10x=None,
        scan_path=None,
        image_queue=None,
        zoom=4,
        full_scan=False,
        full_scan_start_time=None,
    ):
        self.app.open_panel("Info Panel")

        start_time = time.time()

        self.app.set_view("Map", False)
        self.turret_controller.change_objective(2)

        input("Press Enter to start 10x scan...")

        if scan_path is None:
            path = HOME_DIR / "Scans" / datetime.now().strftime("10x (%Y-%m-%d) (%H-%M-%S)")
        else:
            path = scan_path / "All Images" / "10x"

        if scan_coordinates_10x is None:
            x, y, _ = self.stage.get_position()
            scan_coordinates_10x = [[x, y, 10, 10]]

        cropped_flatfield = self.frame_processor.crop_frame(FLATFIELD_IMG)

        for i, coordinates in enumerate(scan_coordinates_10x, start=1):
            wafer_time = time.time()

            self.app.set_true_map(np.zeros((3000, 3000, 3), dtype=np.uint8))

            if full_scan:
                self.update_scan_status(stage=f"10x Scan - wafer {i} / {len(scan_coordinates_10x)}", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")
            else:
                self.update_scan_status(scan_type="10x Scan", stage="10x Scan", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")

            center_x = coordinates[0]
            center_y = coordinates[1]

            coords, total_frames = coordinate_generator.generate_rect_coords(coordinates[2], coordinates[3])

            self.stage.move_to_xy(center_x, center_y)
            self.stage.wait_until_not_busy()

            camera_width, camera_height = self.camera.get_Size()

            max_zoom = max(zoom, int(camera_height / (self.app.get_true_map().shape[0] / coordinates[3])), int(camera_width / (self.app.get_true_map().shape[1] / coordinates[2])))

            for j, (offset_x, offset_y) in enumerate(coords):
                target_x = center_x + offset_x * PIXEL_SIZE["10X"][self.resolution] * RESOLUTION_DIM[self.resolution]["x"] * CROP_RATIO["10X"]["x"]
                target_y = center_y - offset_y * PIXEL_SIZE["10X"][self.resolution] * RESOLUTION_DIM[self.resolution]["y"] * CROP_RATIO["10X"]["y"]

                self.stage.move_to_xy(target_x, target_y)
                self.stage.wait_until_not_busy()

                img = self.frame_processor.capture_frame()

                img = vignetting_corrector.correct_vignetting_effect(
                    img,
                    cropped_flatfield,
                    reference_point=(img.shape[1] // 2, img.shape[0] // 2),
                )

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                image_path = path / f"wafer {i} ({center_x}, {center_y})" / "Raw" / f"img_10x_{j}.png"

                image_path.parent.mkdir(parents=True, exist_ok=True)

                self.frame_processor.save_image(image=img, filename=image_path)

                if image_queue is not None:
                    image_queue.put(image_path)

                if self.app.get_view() == "Camera View":
                    if self.app.get_filter():
                        self.app.set_filter(False)

                    self.root.after(0, lambda img=img: self.app.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

                true_map = self.app.get_true_map()
                filter_map = self.app.get_filter_map()

                map_x = int(filter_map.shape[1] / 2 - (offset_x + 0.5) * img_rgb.shape[1] / max_zoom)
                map_y = int(filter_map.shape[0] / 2 + (offset_y - 0.5) * img_rgb.shape[0] / max_zoom)

                img_small = img_rgb[::max_zoom, ::max_zoom]

                x_start = max(0, map_x)
                y_start = max(0, map_y)
                x_end = min(filter_map.shape[1], x_start + img_small.shape[1])
                y_end = min(filter_map.shape[0], y_start + img_small.shape[0])

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

            print(f"wafer {i} imaging finished!")
            print(f"Time taken: {time.time() - wafer_time:.2f}s")

        if image_queue is not None:
            image_queue.put(None)

        print("10x scan imaging finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")
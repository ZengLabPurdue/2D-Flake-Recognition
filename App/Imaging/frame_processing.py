from collections import deque
from datetime import datetime
from pathlib import Path
import ctypes
import time

import cv2
import numpy as np

from config import HOME_DIR, CROP_RATIO, RESOLUTION, PROCESS_FRAME_RATE

from . import amcam
from . import chip_edge_classifier
from . import image_metadata
from . import vignetting_corrector

class FrameProcessor:
    def __init__(
        self, 
        root, 
        app,
        stage, 
        get_live_mapping, 
    ):
        self.root = root
        self.app = app
        self.stage = stage

        self.get_live_mapping = get_live_mapping
        self.place_frame_on_map = None

        self.hcam = None
        self.buf = None
        self.width = 0
        self.height = 0

        self.frame_id = 0
        self.frame_buffer = deque(maxlen=5)

        self.last_used_capture_frame_id = -1
        self.last_processed_frame_id = -1

        self.was_busy = False
        self.capture_after_move = False

        self.camera_last_time = time.perf_counter()
        self.camera_fps = 0.0
        self.display_last_time = time.perf_counter()
        self.display_fps = 0.0

        self.vignette_filter_key = None
        self.vignette_filter = None
        self.vignette_gain = None

    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            ctx.on_image()

    def on_image(self):
        try:
            self.hcam.PullImageV2(self.buf, 24, None)

            row_bytes = ((self.width * 24 + 31) // 32 * 4)

            img = np.frombuffer(self.buf, dtype=np.uint8).reshape(self.height, row_bytes)

            img = img[:, :self.width * 3].reshape(self.height, self.width, 3)

            img = cv2.flip(img, -1)

            x, y, z = self.stage.get_position()

            frame_data = {
                "frame": img,
                "timestamp": time.time(),
                "x": x,
                "y": y,
                "z": z,
                "frame_id": self.frame_id,
                "stage_busy": self.stage.is_busy(),
            }

            self.frame_buffer.append(frame_data)
            self.frame_id += 1

            new_time = time.perf_counter()
            dt = new_time - self.camera_last_time
            
            if dt > 0:
                self.camera_fps = 1.0 / dt
                self.camera_ms_per_frame = 1000.0 * dt
            
            self.camera_last_time = new_time

        except amcam.HRESULTException as ex:
            print("Camera error:", ex)

    def start_processing_loop(self):
        self.last_used_capture_frame_id = -1
        self.last_processed_frame_id = -1
        self.process_frame()

    def process_frame(self):
        try:
            if len(self.frame_buffer) == 0:
                self.root.after(PROCESS_FRAME_RATE, self.process_frame)
                return

            data = self.frame_buffer[-1]
            img = data["frame"]

            if data["frame_id"] == self.last_processed_frame_id:
                self.root.after(PROCESS_FRAME_RATE, self.process_frame)
                return

            self.last_processed_frame_id = data["frame_id"]

            sharpness, _ = self.app.focus_controller.find_sharpness(img)
            self.app.focus_panel.update_sharpness(sharpness)

            if self.get_live_mapping():
                busy = self.stage.is_busy()

                if self.was_busy and not busy:
                    self.capture_after_move = True

                self.was_busy = busy

                if busy:
                    self.root.after(PROCESS_FRAME_RATE, self.process_frame)
                    return

                if self.capture_after_move:
                    self.capture_after_move = False
                    cropped = self.crop_frame(img)
                    self.place_frame_on_map(cropped, zoom=3)

            if self.app.get_view() == "Camera View":
                camera_image = img
                if self.app.get_camera_vignette_filter():
                    try:
                        camera_image = self.apply_vignette_filter(camera_image)
                    except (FileNotFoundError, ValueError) as exc:
                        self.app.disable_camera_vignette_filter(str(exc))

                if self.app.get_camera_chip_filter():
                    display = cv2.cvtColor(
                        chip_edge_classifier.chip_filter(camera_image),
                        cv2.COLOR_GRAY2RGB,
                    )
                else:
                    display = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

                self.app.display_image(display)

            elif self.app.get_view() == "Map":
                self.app.display_map()

        except Exception as ex:
            print("Frame processing error:", ex)

        new_time = time.perf_counter()
        dt = new_time - self.display_last_time
        
        if dt > 0:
            self.display_fps = 1.0 / dt
            self.display_ms_per_frame = 1000.0 * dt
        
        self.display_last_time = new_time

        self.app.info_panel.update_fps(self.camera_fps, self.display_fps)

        self.root.after(PROCESS_FRAME_RATE, self.process_frame)

    def run_camera(self):
        cams = amcam.Amcam.EnumV2()

        if not cams:
            print("No camera found")
            return

        self.hcam = amcam.Amcam.Open(cams[0].id)

        self.reset_camera_settings()

        self.hcam.put_eSize(RESOLUTION[self.app.get_resolution()])

        self.width, self.height = self.hcam.get_Size()

        bufsize = ((self.width * 24 + 31) // 32 * 4) * self.height
        self.buf = ctypes.create_string_buffer(bufsize)

        self.frame_id = 0

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        scale = min(
            screen_width / self.width,
            screen_height / self.height,
            1.0
        )

        win_width = int(self.width * scale)
        win_height = int(self.height * scale)

        self.root.geometry(f"{win_width}x{win_height}")
        self.root.update_idletasks()
        self.root.update()

        self.hcam.StartPullModeWithCallback(self.cameraCallback, self)

        self.start_processing_loop()

        num_res = self.hcam.ResolutionNumber()
        for i in range(num_res):
            print(f"Resolution {i}: {self.hcam.get_Resolution(i)}")

        print(f"Current Resolution: ({self.width}, {self.height})")

        max_speed = self.hcam.MaxSpeed()
        self.hcam.put_Speed(max_speed)

    def capture_frame(
        self,
        num_images=1,
        cancel_check=None,
        timeout_seconds=None,
    ): #TODO: Figure out last position mixing bug
        if cancel_check is not None:
            cancel_check()
        if len(self.frame_buffer) == 0:
            print("No frames available.")
            return None

        sum_frame = np.zeros_like(
            self.frame_buffer[-1]["frame"],
            dtype=np.float32
        )

        count = 0
        start_time = time.time()
        timeout_seconds = (
            max(30.0, float(num_images) * 10.0)
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        deadline = time.monotonic() + timeout_seconds

        while count < num_images:
            if cancel_check is not None:
                cancel_check()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:g} seconds waiting "
                    "for a fresh camera frame."
                )
            data = self.frame_buffer[-1]

            if data["stage_busy"]:
                time.sleep(0.05)
                continue

            if data["frame_id"] <= self.last_used_capture_frame_id:
                time.sleep(0.05)
                continue

            if data["timestamp"] < start_time:
                time.sleep(0.05)
                continue

            self.last_used_capture_frame_id = data["frame_id"]

            sum_frame += data["frame"].astype(np.float32)
            count += 1

        if count == 0:
            return None

        avg_frame = (sum_frame / count).astype(np.uint8)
        avg_frame = self.crop_frame(avg_frame)

        return avg_frame

    def capture_frame_raw(
        self,
        num_images=100,
        cancel_check=None,
        timeout_seconds=None,
    ):
        if cancel_check is not None:
            cancel_check()
        if len(self.frame_buffer) == 0:
            print("No frames available.")
            return None

        sum_frame = np.zeros_like(
            self.frame_buffer[-1]["frame"],
            dtype=np.float32
        )

        count = 0
        start_time = time.time()
        timeout_seconds = (
            max(30.0, float(num_images) * 10.0)
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        deadline = time.monotonic() + timeout_seconds

        while count < num_images:
            if cancel_check is not None:
                cancel_check()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:g} seconds waiting "
                    "for fresh raw camera frames."
                )
            data = self.frame_buffer[-1]

            if data["stage_busy"]:
                time.sleep(0.05)
                continue

            if data["frame_id"] <= self.last_used_capture_frame_id:
                time.sleep(0.05)
                continue

            if data["timestamp"] < start_time:
                time.sleep(0.05)
                continue

            self.last_used_capture_frame_id = data["frame_id"]

            sum_frame += data["frame"].astype(np.float32)
            count += 1

        if count == 0:
            print("No frames captured.")
            return None

        avg_frame = (sum_frame / count).astype(np.uint8)
        return avg_frame

    def crop_frame(self, frame):
        h, w = frame.shape[:2]

        if self.app.get_magnification() == "2X":
            crop_w = int(w * CROP_RATIO["2X"]["x"])
            crop_h = int(h * CROP_RATIO["2X"]["y"])
        elif self.app.get_magnification() == "10X":
            crop_w = int(w * CROP_RATIO["10X"]["x"])
            crop_h = int(h * CROP_RATIO["10X"]["y"])
        elif self.app.get_magnification() == "20X":
            crop_w = int(w * CROP_RATIO["20X"]["x"])
            crop_h = int(h * CROP_RATIO["20X"]["y"])
        elif self.app.get_magnification() == "100X":
            crop_w = int(w * CROP_RATIO["100X"]["x"])
            crop_h = int(h * CROP_RATIO["100X"]["y"])
        else:
            crop_w = w
            crop_h = h

        cx, cy = w // 2, h // 2

        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = cx + crop_w // 2
        y2 = cy + crop_h // 2

        return frame[y1:y2, x1:x2]

    def get_vignette_filter_path(self):
        magnification = self.app.get_magnification().lower()
        resolution = self.app.get_resolution().lower()
        return HOME_DIR / "Flatfields" / f"vignette_filter_{magnification}_{resolution}.png"

    def load_vignette_filter(self, image_shape=None):
        key = (self.app.get_magnification(), self.app.get_resolution())
        if (
            self.vignette_filter_key != key
            or self.vignette_filter is None
            or self.vignette_gain is None
        ):
            path = self.get_vignette_filter_path()
            if not path.is_file():
                raise FileNotFoundError(
                    f"No vignette filter is available for {key[0]} at {key[1]} resolution.\n\n"
                    f"Expected file:\n{path}"
                )
            vignette_filter = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if vignette_filter is None:
                raise FileNotFoundError(
                    f"No vignette filter is available for {key[0]} at {key[1]} resolution.\n\n"
                    f"Expected file:\n{path}"
                )
            self.vignette_filter_key = key
            self.vignette_filter = vignette_filter
            self.vignette_gain = vignetting_corrector.create_vignette_gain(
                vignette_filter,
                reference_point=(
                    vignette_filter.shape[1] // 2,
                    vignette_filter.shape[0] // 2,
                ),
            )

        vignette_filter = self.vignette_filter
        if image_shape is not None and vignette_filter.shape != image_shape:
            cropped_filter = self.crop_frame(vignette_filter)
            if cropped_filter.shape != image_shape:
                raise ValueError(
                    "The vignette filter dimensions do not match the camera image."
                )
            vignette_filter = cropped_filter
        return vignette_filter

    def apply_vignette_filter(self, image):
        self.load_vignette_filter(image.shape)
        vignette_gain = self.vignette_gain
        if vignette_gain.shape != image.shape[:2]:
            vignette_gain = self.crop_frame(vignette_gain)
        return vignetting_corrector.apply_vignette_gain(image, vignette_gain)

    def clear_vignette_filter_cache(self):
        self.vignette_filter_key = None
        self.vignette_filter = None
        self.vignette_gain = None

    def save_image(
        self,
        image=None,
        save_dir=None,
        filename=None,
        output=False,
        crop=False,
        apply_vignette=False,
        apply_chip_filter=False,
        vignette_applied=None,
        metadata=None,
    ):
        if image is None:
            image = self.capture_frame_raw()

        if image is None:
            print("No image to save.")
            return None

        if apply_vignette:
            image = self.apply_vignette_filter(image)
        if crop:
            image = self.crop_frame(image)
        if apply_chip_filter:
            image = chip_edge_classifier.chip_filter(image)

        if vignette_applied is None:
            vignette_applied = apply_vignette

        if save_dir is None:
            save_dir = HOME_DIR / "Saved Images"
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filepath = save_dir / f"image_{timestamp}.png"
        else:
            filepath = save_dir / filename

        if filepath.suffix.lower() == ".png":
            png_metadata = dict(metadata or {})
            png_metadata.update({
                "vignette_applied": bool(vignette_applied),
                "chip_filter_applied": bool(apply_chip_filter),
                "magnification": self.app.get_magnification(),
                "resolution": self.app.get_resolution(),
            })
            image_metadata.save_png(
                filepath,
                image,
                metadata=png_metadata,
            )
        elif not cv2.imwrite(str(filepath), image):
            raise OSError(f"OpenCV could not write the image to {filepath}")

        if output:
            print(f"Image saved to {filepath}")

        return filepath

    def get_camera(self):
        return self.hcam

    def close(self):
        try:
            if self.hcam is not None:
                self.hcam.Close()
        except Exception as e:
            print("Camera close error:", e)

        self.hcam = None
        self.buf = None

    # ------------- Camera Settings -------------

    def set_default_exposure(self):
        self.hcam.put_AutoExpoEnable(False)

        self.hcam.put_ExpoTime(1500)
        self.hcam.put_ExpoAGain(100)        

    def set_auto_exposure(self, active : bool):
        self.hcam.put_AutoExpoEnable(active)        

    def get_auto_exposure(self):
        return int(float(self.hcam.get_AutoExpoTarget()))

    def reset_camera_settings(self):
        self.hcam.put_AutoExpoEnable(False)

        self.hcam.put_ExpoTime(1500)
        self.hcam.put_ExpoAGain(100)

        self.hcam.put_Option(amcam.AMCAM_OPTION_RAW, 0)

        self.hcam.put_Option(amcam.AMCAM_OPTION_COLORMATIX, 1)
        self.hcam.put_Option(amcam.AMCAM_OPTION_LINEAR, 1)
        self.hcam.put_Option(amcam.AMCAM_OPTION_CURVE, 1)

        self.hcam.put_Option(amcam.AMCAM_OPTION_SHARPENING, 0)
        self.hcam.put_Option(amcam.AMCAM_OPTION_DENOISE, 0)
        self.hcam.put_Option(amcam.AMCAM_OPTION_DEFECT_PIXEL, 1)

        self.hcam.put_Brightness(0)
        self.hcam.put_Contrast(0)
        self.hcam.put_Gamma(100)
        self.hcam.put_Saturation(128)

    def change_resolution(self, resolution):
        self.app.set_resolution(resolution)

        self.app.disable_buttons()

        try:
            if self.hcam is not None:
                self.hcam.Close()

            cams = amcam.Amcam.EnumV2()
            self.hcam = amcam.Amcam.Open(cams[0].id)

            self.reset_camera_settings()

            self.hcam.put_eSize(RESOLUTION[self.app.get_resolution()])

            self.width, self.height = self.hcam.get_Size()

            bufsize = ((self.width * 24 + 31) // 32 * 4) * self.height
            self.buf = ctypes.create_string_buffer(bufsize)

            self.frame_buffer.clear()
            self.frame_id = 0
            self.last_used_capture_frame_id = -1
            self.last_processed_frame_id = -1

            self.hcam.StartPullModeWithCallback(self.cameraCallback, self)

            num_res = self.hcam.ResolutionNumber()
            for i in range(num_res):
                print(f"Resolution {i}: {self.hcam.get_Resolution(i)}")

            print(f"Current Resolution: ({self.width}, {self.height})")

        except Exception as e:
            print(f"Error changing resolution: {e}")

        finally:
            self.app.enable_buttons()

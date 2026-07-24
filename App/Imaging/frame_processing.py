from collections import deque
from datetime import datetime
from pathlib import Path
import ctypes
import threading
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
        self._frame_condition = threading.Condition()
        self._capture_barrier = None
        self._live_capture_min_frame_id = None
        self._live_capture_expected_xy = None
        self._camera_flush_warning_shown = False

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
                "monotonic_timestamp": time.monotonic(),
                "x": x,
                "y": y,
                "z": z,
                "stage_busy": self.stage.is_busy(),
            }

            with self._frame_condition:
                frame_data["frame_id"] = self.frame_id
                self.frame_buffer.append(frame_data)
                self.frame_id += 1
                self._frame_condition.notify_all()

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
        self._capture_barrier = None
        self.reset_live_mapping_capture()
        self.process_frame()

    def process_frame(self):
        display_metrics = None
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

                if busy:
                    self.was_busy = True
                    self.capture_after_move = False
                elif self.was_busy:
                    stage_x, stage_y, _stage_z = self.stage.get_position(
                        strict=True
                    )
                    self.was_busy = False
                    self.capture_after_move = True
                    self._arm_live_map_capture((stage_x, stage_y))

                if busy:
                    self.root.after(PROCESS_FRAME_RATE, self.process_frame)
                    return

                if self.capture_after_move:
                    stage_position = self._consume_live_map_frame(data)
                    if stage_position is not None:
                        self.capture_after_move = False
                        cropped = self.crop_frame(img)
                        cropped = self.apply_vignette_filter(cropped)
                        self.place_frame_on_map(
                            cropped,
                            zoom=3,
                            stage_position=stage_position,
                        )

            if self.app.get_view() == "Camera View":
                camera_image = img
                if self.app.get_camera_vignette_filter():
                    try:
                        camera_image = self.apply_vignette_filter(camera_image)
                    except (FileNotFoundError, ValueError) as exc:
                        self.app.disable_camera_vignette_filter(str(exc))

                if self.app.get_camera_chip_filter():
                    display_metrics = self.app.display_image(
                        chip_edge_classifier.chip_filter(camera_image),
                        color_order="GRAY",
                    )
                else:
                    # Resize the camera's native BGR frame before converting
                    # the much smaller display image to RGB.
                    display_metrics = self.app.display_image(
                        camera_image,
                        color_order="BGR",
                    )
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

        self.app.info_panel.update_fps(
            self.camera_fps,
            self.display_fps,
            render_ms=(
                display_metrics["total_ms"]
                if display_metrics is not None
                else None
            ),
            render_backend=(
                display_metrics["backend"]
                if display_metrics is not None
                else None
            ),
        )

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

    def reset_live_mapping_capture(self):
        self.was_busy = False
        self.capture_after_move = False
        self._live_capture_min_frame_id = None
        self._live_capture_expected_xy = None

    def _flush_camera_queue(self, discard_frames=1):
        if self.hcam is not None:
            try:
                # Flush camera DDR and both SDK-side frame deques. A callback
                # already in progress is handled by discarding the next frame.
                self.hcam.put_Option(amcam.AMCAM_OPTION_FLUSH, 3)
            except Exception as exc:
                if not self._camera_flush_warning_shown:
                    print(
                        "Camera queue flush is unavailable; using the "
                        f"software frame barrier instead: {exc}"
                    )
                    self._camera_flush_warning_shown = True
        with self._frame_condition:
            self.frame_buffer.clear()
            return self.frame_id + max(0, int(discard_frames))

    def arm_capture_after_motion(self, expected_xy):
        """Flush moving-stage frames and bind the next capture to one XY."""
        expected_xy = (
            int(round(expected_xy[0])),
            int(round(expected_xy[1])),
        )
        minimum_frame_id = self._flush_camera_queue(discard_frames=1)
        with self._frame_condition:
            self._capture_barrier = (
                minimum_frame_id,
                expected_xy,
                getattr(self.stage, "motion_sequence", None),
            )
            self._frame_condition.notify_all()
        return expected_xy

    def _arm_live_map_capture(self, expected_xy):
        minimum_frame_id = self._flush_camera_queue(discard_frames=1)
        with self._frame_condition:
            self._live_capture_min_frame_id = minimum_frame_id
            self._live_capture_expected_xy = (
                int(round(expected_xy[0])),
                int(round(expected_xy[1])),
            )

    def _frame_matches_barrier(
        self,
        data,
        minimum_frame_id,
        expected_xy,
    ):
        if data["frame_id"] < minimum_frame_id or data["stage_busy"]:
            return False
        if expected_xy is None:
            return True
        tolerance = getattr(self.stage, "POSITION_TOLERANCE_UM", 1.0)
        return (
            abs(data["x"] - expected_xy[0]) <= tolerance
            and abs(data["y"] - expected_xy[1]) <= tolerance
        )

    def _consume_live_map_frame(self, data):
        minimum_frame_id = self._live_capture_min_frame_id
        expected_xy = self._live_capture_expected_xy
        if minimum_frame_id is None or not self._frame_matches_barrier(
            data,
            minimum_frame_id,
            expected_xy,
        ):
            return None
        self._live_capture_min_frame_id = None
        self._live_capture_expected_xy = None
        return expected_xy

    def _begin_capture_sequence(self, cancel_check=None):
        if cancel_check is not None:
            cancel_check()

        current_motion_sequence = getattr(
            self.stage,
            "motion_sequence",
            None,
        )
        with self._frame_condition:
            barrier = self._capture_barrier
            self._capture_barrier = None
        if barrier is not None:
            minimum_frame_id, expected_xy, motion_sequence = barrier
            if motion_sequence == current_motion_sequence:
                return (
                    minimum_frame_id,
                    expected_xy,
                    current_motion_sequence,
                )

        # Standalone captures may be requested outside the scan movement
        # wrappers. Wait adaptively for controller idle before flushing.
        if self.stage.is_busy():
            self.stage.wait_until_not_busy(cancel_check=cancel_check)

        minimum_frame_id = self._flush_camera_queue(discard_frames=1)
        return (
            minimum_frame_id,
            None,
            getattr(self.stage, "motion_sequence", None),
        )

    def _restart_capture_after_motion(self, cancel_check=None):
        """Discard an interrupted average and start after motion is idle."""
        self.stage.wait_until_not_busy(cancel_check=cancel_check)
        minimum_frame_id = self._flush_camera_queue(discard_frames=1)
        return (
            minimum_frame_id,
            None,
            getattr(self.stage, "motion_sequence", None),
        )

    def _capture_average(
        self,
        num_images,
        crop,
        cancel_check=None,
        timeout_seconds=None,
    ):
        if (
            isinstance(num_images, bool)
            or not isinstance(num_images, (int, np.integer))
            or int(num_images) < 1
        ):
            raise ValueError("num_images must be a positive whole number.")
        num_images = int(num_images)
        timeout_seconds = (
            max(30.0, float(num_images) * 10.0)
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        deadline = time.monotonic() + timeout_seconds
        (
            minimum_frame_id,
            expected_xy,
            capture_motion_sequence,
        ) = self._begin_capture_sequence(cancel_check=cancel_check)
        accumulator = None
        count = 0

        while count < num_images:
            if cancel_check is not None:
                cancel_check()
            current_motion_sequence = getattr(
                self.stage,
                "motion_sequence",
                None,
            )
            if current_motion_sequence != capture_motion_sequence:
                (
                    minimum_frame_id,
                    expected_xy,
                    capture_motion_sequence,
                ) = self._restart_capture_after_motion(
                    cancel_check=cancel_check
                )
                accumulator = None
                count = 0
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                description = "raw camera frames" if not crop else "camera frames"
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:g} seconds waiting "
                    f"for fresh {description}."
                )

            with self._frame_condition:
                data = self.frame_buffer[-1] if self.frame_buffer else None
                motion_observed = bool(
                    data is not None
                    and data["frame_id"] >= minimum_frame_id
                    and data["stage_busy"]
                )
                if (
                    not motion_observed
                    and (
                        data is None
                        or data["frame_id"] <= self.last_used_capture_frame_id
                        or not self._frame_matches_barrier(
                            data,
                            minimum_frame_id,
                            expected_xy,
                        )
                    )
                ):
                    self._frame_condition.wait(
                        timeout=min(0.1, remaining)
                    )
                    continue
                if not motion_observed:
                    self.last_used_capture_frame_id = data["frame_id"]
                    frame = data["frame"]

            if motion_observed:
                (
                    minimum_frame_id,
                    expected_xy,
                    capture_motion_sequence,
                ) = self._restart_capture_after_motion(
                    cancel_check=cancel_check
                )
                accumulator = None
                count = 0
                continue

            if crop:
                frame = self.crop_frame(frame)
            if num_images == 1:
                return frame.copy()
            if accumulator is None:
                accumulator = np.zeros(frame.shape, dtype=np.float32)
            cv2.accumulate(frame, accumulator)
            count += 1

        return cv2.convertScaleAbs(
            accumulator,
            alpha=1.0 / count,
        )

    def capture_frame(
        self,
        num_images=1,
        cancel_check=None,
        timeout_seconds=None,
    ):
        return self._capture_average(
            num_images=num_images,
            crop=True,
            cancel_check=cancel_check,
            timeout_seconds=timeout_seconds,
        )

    def capture_frame_raw(
        self,
        num_images=100,
        cancel_check=None,
        timeout_seconds=None,
    ):
        return self._capture_average(
            num_images=num_images,
            crop=False,
            cancel_check=cancel_check,
            timeout_seconds=timeout_seconds,
        )

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
        if vignette_gain.shape[:2] != image.shape[:2]:
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

            with self._frame_condition:
                self.frame_buffer.clear()
                self.frame_id = 0
                self.last_used_capture_frame_id = -1
                self.last_processed_frame_id = -1
                self._capture_barrier = None
                self.reset_live_mapping_capture()
                self._frame_condition.notify_all()

            self.hcam.StartPullModeWithCallback(self.cameraCallback, self)

            num_res = self.hcam.ResolutionNumber()
            for i in range(num_res):
                print(f"Resolution {i}: {self.hcam.get_Resolution(i)}")

            print(f"Current Resolution: ({self.width}, {self.height})")

        except Exception as e:
            print(f"Error changing resolution: {e}")

        finally:
            self.app.enable_buttons()

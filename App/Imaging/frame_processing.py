from collections import deque
from datetime import datetime
from pathlib import Path
import ctypes
import time

import cv2
import numpy as np

from config import HOME_DIR, CROP_RATIO

from . import amcam
from . import chip_edge_classifier

PROCESS_FRAME_RATE = 30

class FrameProcessor:
    def __init__(
        self, 
        root, 
        app,
        stage, 
        #get_live_mapping_status, 
        #place_live_frame_on_map,
    ):
        self.root = root
        self.app = app
        self.stage = stage

        #self.get_live_mapping_status = get_live_mapping_status
        #self.place_live_frame_on_map = place_live_frame_on_map #TODO: Re-add mapping

        self.hcam = None
        self.buf = None
        self.width = 0
        self.height = 0
        self.resolution = None

        self.frame_id = 0
        self.frame_buffer = deque(maxlen=10)

        self.last_used_capture_frame_id = -1
        self.last_processed_frame_id = -1

        self.was_busy = False
        self.capture_after_move = False

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

            pos = self.stage.get_position()

            frame_data = {
                "frame": img,
                "timestamp": time.time(),
                "x": pos.x if hasattr(pos, "x") else pos[0],
                "y": pos.y if hasattr(pos, "y") else pos[1],
                "frame_id": self.frame_id,
                "stage_busy": self.stage.is_busy(),
            }

            self.frame_buffer.append(frame_data)
            self.frame_id += 1

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

            '''
            if self.get_live_mapping_status():
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
                    #self.place_live_frame_on_map(cropped, zoom=3)
            '''

            if self.app.get_view() == "Camera View":
                if self.app.get_filter():
                    display = cv2.cvtColor(chip_edge_classifier.chip_filter(img), cv2.COLOR_GRAY2RGB)
                else:
                    display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                self.app.display_image(display)

            elif self.app.get_view() == "Map":
                self.app.display_map()

        except Exception as ex:
            print("Frame processing error:", ex)

        self.root.after(PROCESS_FRAME_RATE, self.process_frame)

    def run_camera(self):
        cams = amcam.Amcam.EnumV2()

        if not cams:
            print("No camera found")
            return

        self.hcam = amcam.Amcam.Open(cams[0].id)

        self.reset_camera_settings(self.hcam)

        self.hcam.put_eSize(1)
        self.resolution = self.hcam.get_eSize()

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

    def reset_camera_settings(self, cam):
        cam.put_AutoExpoEnable(False)

        cam.put_ExpoTime(1500)
        cam.put_ExpoAGain(100)

        cam.put_Option(amcam.AMCAM_OPTION_RAW, 0)

        cam.put_Option(amcam.AMCAM_OPTION_COLORMATIX, 1)
        cam.put_Option(amcam.AMCAM_OPTION_LINEAR, 1)
        cam.put_Option(amcam.AMCAM_OPTION_CURVE, 1)

        cam.put_Option(amcam.AMCAM_OPTION_SHARPENING, 0)
        cam.put_Option(amcam.AMCAM_OPTION_DENOISE, 0)
        cam.put_Option(amcam.AMCAM_OPTION_DEFECT_PIXEL, 1)

        cam.put_Brightness(0)
        cam.put_Contrast(0)
        cam.put_Gamma(100)
        cam.put_Saturation(128)

    def change_resolution(self, index):
        self.resolution = index

        self.app.disable_buttons()

        try:
            if self.hcam is not None:
                self.hcam.Close()

            cams = amcam.Amcam.EnumV2()
            self.hcam = amcam.Amcam.Open(cams[0].id)

            self.reset_camera_settings(self.hcam)

            self.hcam.put_eSize(index)

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

        finally:
            self.app.enable_buttons()

    def capture_frame(self, num_images=2):
        if len(self.frame_buffer) == 0:
            print("No frames available.")
            return None

        sum_frame = np.zeros_like(
            self.frame_buffer[-1]["frame"],
            dtype=np.float32
        )

        count = 0
        start_time = time.time()

        while count < num_images:
            try:
                self.root.update_idletasks()
                self.root.update()

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

            except Exception as e:
                print(f"Error occurred while capturing frame: {e}")
                break

        if count == 0:
            return None

        avg_frame = (sum_frame / count).astype(np.uint8)
        avg_frame = self.crop_frame(avg_frame)

        return avg_frame

    def capture_frame_raw(self, num_images=100):
        if len(self.frame_buffer) == 0:
            print("No frames available.")
            return None

        sum_frame = np.zeros_like(
            self.frame_buffer[-1]["frame"],
            dtype=np.float32
        )

        count = 0
        start_time = time.time()

        while count < num_images:
            try:
                print(f"Capturing frame {count + 1}/{num_images}...")

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

            except Exception:
                print("Frame timeout")
                break

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

    def save_image(self, image=None, save_dir=None, filename=None, output=True):
        if image is None:
            image = self.capture_frame_raw()

        if image is None:
            print("No image to save.")
            return None

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

        cv2.imwrite(str(filepath), image)

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
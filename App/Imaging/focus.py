import threading
import time

import numpy as np
import cv2

class FocusController:
    def __init__(
        self,
        app,
        stage,
        frame_processor,
        sharpness_callback=None,
    ):
        self.app = app
        self.stage = stage
        self.frame_processor = frame_processor

        self.sharpness_callback = sharpness_callback or (lambda sharpness: None)

        self.focus_thread = None
        self.stop_focus_event = threading.Event()
        self.focus_running = False

    def _run_on_ui_thread(self, callback, *args):
        dispatcher = getattr(self.app, "call_on_ui_thread", None)
        if dispatcher is not None:
            dispatcher(callback, *args)
            return
        if threading.current_thread() is threading.main_thread():
            callback(*args)
        else:
            self.app.root.after(0, callback, *args)

    def _wait_for_focus_thread(self, focus_thread):
        if threading.current_thread() is threading.main_thread():
            while focus_thread.is_alive():
                self.app.root.update()
                focus_thread.join(timeout=0.05)
            self.app.root.update()
        else:
            focus_thread.join()

    def start_auto_focus_thread(self, focus_range=1000, z_velo=500, z_accel=10000, peak_found_threshold=100, wait=True):
        if self.focus_thread is not None and self.focus_thread.is_alive():
            print("Autofocus already running")

            if wait:
                self._wait_for_focus_thread(self.focus_thread)

            return

        self.stop_focus_event.clear()
        self.focus_running = True

        self._run_on_ui_thread(self.app.disable_buttons)

        self.focus_thread = threading.Thread(
            target=self._auto_focus_worker,
            args=(focus_range, z_velo, z_accel, peak_found_threshold,),
            daemon=True,
        )

        self.focus_thread.start()

        if wait:
            self._wait_for_focus_thread(self.focus_thread)

        return self.focus_thread

    def _auto_focus_worker(self, focus_range, z_velo, z_accel, peak_found_threshold):
        try:
            self.auto_focus(focus_range=focus_range, z_velo=z_velo, z_accel=z_accel, peak_found_threshold=peak_found_threshold)

        except Exception as e:
            import traceback
            print("Autofocus error:", e)
            traceback.print_exc()

        finally:
            self.focus_running = False
            self._run_on_ui_thread(self.app.enable_buttons)

    def stop_auto_focus(self):
        self.stop_focus_event.set()

    def get_latest_frame(self):
        if len(self.frame_processor.frame_buffer) == 0:
            return None, None, None

        data = self.frame_processor.frame_buffer[-1]

        frame = data["frame"]
        frame_id = data["frame_id"]
        z = data["z"]

        return frame, frame_id, z

    def find_sharpness(self, image):
        start_time = time.perf_counter()

        if image is None:
            return 0, 0

        h, w = image.shape[:2]

        roi_w = int(w * 0.4)
        roi_h = int(h * 0.4)

        x1 = w // 2 - roi_w // 2
        y1 = h // 2 - roi_h // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        roi = image[y1:y2, x1:x2]

        roi = cv2.resize(
            roi,
            None,
            fx=0.25,
            fy=0.25,
            interpolation=cv2.INTER_AREA
        )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        sharpness = float(np.mean(lap * lap))

        self._run_on_ui_thread(self.sharpness_callback, sharpness)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return sharpness, elapsed_ms

    def auto_focus(self, focus_range=3000, z_velo=500, z_accel=10000, peak_found_threshold=100):
        start_time = time.perf_counter()

        default_z_velo = self.stage.get_z_velocity()
        default_z_accel = self.stage.get_z_acceleration()

        self.stage.set_z_velocity(z_velo)
        self.stage.set_z_acceleration(z_accel)

        current_z = self.stage.get_z_position()

        z_start = current_z - focus_range
        z_end = current_z + focus_range

        print(
            f"Starting autofocus | "
            f"Current Z: {current_z:.1f} | "
            f"Range: ±{focus_range} | "
            f"Search: {z_start:.1f} to {z_end:.1f}"
        )

        def find_peak(peak_found_threshold=100, stop_after_drop=True):

            peak_found = False
            best_sharpness = None
            best_z = None

            last_frame_id = -1
            is_not_busy_check = 0

            while is_not_busy_check < 10:

                if self.stop_focus_event.is_set():
                    self.stage.stop_z()
                    self.stage.wait_until_not_busy()
                    break

                if self.stage.is_busy():
                    is_not_busy_check = 0
                else:
                    is_not_busy_check += 1

                frame, frame_id, z = self.get_latest_frame()

                if frame is None:
                    time.sleep(0.001)
                    continue

                if frame_id == last_frame_id:
                    time.sleep(0.001)
                    continue

                last_frame_id = frame_id

                sharpness, elapsed_ms = self.find_sharpness(frame)

                if best_sharpness is None:
                    best_sharpness = sharpness
                    best_z = z
                elif sharpness > best_sharpness:
                    if sharpness > best_sharpness + peak_found_threshold:
                        peak_found = True
                    best_sharpness = sharpness
                    best_z = z

                print(
                    f"Busy: {self.stage.is_busy()} | "
                    f"Z: {z:>12.1f} | "
                    f"Sharpness: {sharpness:>10.3f} | "
                    f"Best Z: {best_z:>12.1f} | "
                    f"Best Sharpness: {best_sharpness:>10.3f} | "
                    f"Time: {elapsed_ms:>8.2f}ms"
                )

                if stop_after_drop and sharpness < best_sharpness - peak_found_threshold:
                    print("Peak passed, stopping Z motion")
                    self.stage.stop_z()
                    self.stage.wait_until_not_busy()
                    break

            return peak_found, best_sharpness, best_z

        self.stage.move_to_z(z_start, wait=False)

        _, best_sharpness, best_z = find_peak(peak_found_threshold)

        if self.stop_focus_event.is_set():
            self.stage.set_z_velocity(default_z_velo)
            self.stage.set_z_acceleration(default_z_accel)
            return current_z

        self.stage.move_to_z(z_end, wait=False)

        _, upward_best_sharpness, upward_best_z = find_peak(
            peak_found_threshold,
            stop_after_drop=False,
        )
        if (
            upward_best_sharpness is not None
            and (
                best_sharpness is None
                or upward_best_sharpness > best_sharpness
            )
        ):
            best_sharpness = upward_best_sharpness
            best_z = upward_best_z

        if self.stop_focus_event.is_set():
            self.stage.set_z_velocity(default_z_velo)
            self.stage.set_z_acceleration(default_z_accel)
            return current_z

        if best_sharpness is None or best_z is None or best_sharpness < 0:
            print("No valid focus frames found")
            self.stage.move_to_z(current_z)
            self.stage.wait_until_not_busy()
            self.stage.set_z_velocity(default_z_velo)
            self.stage.set_z_acceleration(default_z_accel)
            return current_z

        time_taken = (time.perf_counter() - start_time)

        print(
            f"Autofocus complete | "
            f"Best Z: {best_z:.1f} | "
            f"Best Sharpness: {best_sharpness:.3f} | "
            f"Time: {int(time_taken)}s"
        )

        self.stage.move_to_z(best_z, wait=True)

        final_z = self.stage.get_z_position()

        frame, _, _ = self.get_latest_frame()
        sharpness, _ = self.find_sharpness(frame)

        print(
            f"Stage Position: {final_z:.1f} | "
            f"Target Best Z: {best_z:.1f} | "
            f"Sharpness: {sharpness:>10.3f}"
        )

        self.stage.set_z_velocity(default_z_velo)
        self.stage.set_z_acceleration(default_z_accel)

        return best_z

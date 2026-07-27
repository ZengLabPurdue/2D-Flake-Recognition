import math
import threading
import time

import numpy as np
import cv2


class FocusCancelled(RuntimeError):
    pass


class FocusController:
    FOCUS_PROFILES = {
        "2X": {
            "fine_radius_fraction": 0.12,
            "maximum_fine_radius": 80.0,
            "target_precision": 5.0,
            "fine_velocity": 500,
            "fine_frame_count": 1,
            "coarse_drop_ratio": 0.18,
            "sharpness_resize_scale": 0.25,
        },
        "10X": {
            "fine_radius_fraction": 0.12,
            "maximum_fine_radius": 30.0,
            "target_precision": 1.0,
            "fine_velocity": 50,
            "fine_frame_count": 2,
            "coarse_drop_ratio": 0.15,
            "sharpness_resize_scale": 0.50,
        },
        "20X": {
            "fine_radius_fraction": 0.12,
            "maximum_fine_radius": 15.0,
            "target_precision": 0.5,
            "fine_velocity": 25,
            "fine_frame_count": 2,
            "coarse_drop_ratio": 0.12,
            "sharpness_resize_scale": 0.50,
        },
        "100X": {
            "fine_radius_fraction": 0.15,
            "maximum_fine_radius": 3.0,
            "target_precision": 0.1,
            "fine_velocity": 5,
            "fine_frame_count": 3,
            "coarse_drop_ratio": 0.10,
            "sharpness_resize_scale": 0.75,
        },
    }
    DEFAULT_FOCUS_PROFILE = {
        "fine_radius_fraction": 0.12,
        "maximum_fine_radius": 30.0,
        "target_precision": 1.0,
        "fine_velocity": 50,
        "fine_frame_count": 2,
        "coarse_drop_ratio": 0.15,
        "sharpness_resize_scale": 0.50,
    }
    FOCUS_FRAME_TIMEOUT_SECONDS = 5.0
    MAX_REFINEMENT_ITERATIONS = 12

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

    def _check_focus_cancelled(self):
        if self.stop_focus_event.is_set():
            raise FocusCancelled("Autofocus was stopped.")

    def get_latest_frame(self):
        data = self._latest_frame_data()
        if data is None:
            return None, None, None

        frame = data["frame"]
        frame_id = data["frame_id"]
        z = data["z"]

        return frame, frame_id, z

    def _latest_frame_data(self):
        condition = getattr(self.frame_processor, "_frame_condition", None)
        if condition is None:
            if len(self.frame_processor.frame_buffer) == 0:
                return None
            return self.frame_processor.frame_buffer[-1]
        with condition:
            if len(self.frame_processor.frame_buffer) == 0:
                return None
            return self.frame_processor.frame_buffer[-1]

    def _wait_for_new_frame_data(self, after_frame_id, timeout):
        deadline = time.monotonic() + max(0.0, timeout)
        condition = getattr(self.frame_processor, "_frame_condition", None)
        while True:
            self._check_focus_cancelled()
            data = self._latest_frame_data()
            if data is not None and data["frame_id"] > after_frame_id:
                return data
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if condition is None:
                time.sleep(min(0.005, remaining))
                continue
            with condition:
                condition.wait(timeout=min(0.05, remaining))

    def find_sharpness(self, image, resize_scale=0.25):
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

        resize_scale = min(1.0, max(0.1, float(resize_scale)))
        if resize_scale < 1.0:
            roi = cv2.resize(
                roi,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        sharpness = float(np.mean(lap * lap))

        self._run_on_ui_thread(self.sharpness_callback, sharpness)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return sharpness, elapsed_ms

    def _focus_profile(self, focus_range):
        magnification = str(self.app.get_magnification()).upper()
        profile = dict(
            self.FOCUS_PROFILES.get(
                magnification,
                self.DEFAULT_FOCUS_PROFILE,
            )
        )
        target_precision = min(
            float(profile["target_precision"]),
            max(0.1, float(focus_range) / 4),
        )
        profile["target_precision"] = target_precision
        profile["initial_fine_radius"] = min(
            float(profile["maximum_fine_radius"]),
            max(
                target_precision * 4,
                float(focus_range) * profile["fine_radius_fraction"],
            ),
        )
        return magnification, profile

    def _move_to_focus_z(self, target_z, wait=True):
        self._check_focus_cancelled()
        self.stage.wait_until_not_busy(
            cancel_check=self._check_focus_cancelled
        )
        moved = self.stage.move_to_z(
            float(target_z),
            wait=wait,
            cancel_check=self._check_focus_cancelled,
        )
        if not moved:
            raise RuntimeError(
                f"The stage could not start the autofocus Z move to {target_z:g}."
            )

    def _measure_focus_at(self, target_z, frame_count=1):
        self._move_to_focus_z(target_z, wait=True)
        self._check_focus_cancelled()
        frame = self.frame_processor.capture_frame(
            num_images=max(1, int(frame_count)),
            cancel_check=self._check_focus_cancelled,
            timeout_seconds=self.FOCUS_FRAME_TIMEOUT_SECONDS,
        )
        magnification = str(self.app.get_magnification()).upper()
        metric_profile = self.FOCUS_PROFILES.get(
            magnification,
            self.DEFAULT_FOCUS_PROFILE,
        )
        sharpness, elapsed_ms = self.find_sharpness(
            frame,
            resize_scale=metric_profile["sharpness_resize_scale"],
        )
        actual_z = self.stage.get_z_position(strict=True)
        print(
            f"Fine focus | Z: {actual_z:>10.2f} | "
            f"Sharpness: {sharpness:>10.3f} | "
            f"Metric: {elapsed_ms:>7.2f}ms"
        )
        return float(sharpness), float(actual_z)

    @staticmethod
    def _focus_comparison_margin(
        scores,
        peak_found_threshold,
    ):
        largest_score = max((abs(float(score)) for score in scores), default=0.0)
        return max(
            1e-6,
            float(peak_found_threshold) * 0.25,
            largest_score * 0.02,
        )

    def _adaptive_focus_bracket(
        self,
        current_z,
        z_start,
        z_end,
        peak_found_threshold,
        profile,
    ):
        """Probe locally and expand only in the direction that gets sharper."""
        half_range = (float(z_end) - float(z_start)) / 2
        initial_step = min(
            profile["initial_fine_radius"],
            max(profile["target_precision"] * 4, half_range / 4),
        )
        cache = {}

        def evaluate(target_z):
            target_z = round(
                min(float(z_end), max(float(z_start), float(target_z))),
                1,
            )
            if target_z not in cache:
                score, actual_z = self._measure_focus_at(
                    target_z,
                    frame_count=1,
                )
                cache[target_z] = (actual_z, score)
            return cache[target_z]

        center = evaluate(current_z)
        left = evaluate(current_z - initial_step)
        right = evaluate(current_z + initial_step)
        initial_samples = (left, center, right)
        margin = self._focus_comparison_margin(
            [sample[1] for sample in initial_samples],
            peak_found_threshold,
        )

        if (
            center[1] >= left[1] + margin
            and center[1] >= right[1] + margin
        ):
            return list(cache.values()), (
                min(left[0], right[0]),
                max(left[0], right[0]),
            )

        direction = None
        if (
            right[1] >= center[1] + margin
            and right[1] >= left[1] + margin
        ):
            direction = 1
            previous, best = center, right
        elif (
            left[1] >= center[1] + margin
            and left[1] >= right[1] + margin
        ):
            direction = -1
            previous, best = center, left

        # A nearly flat local curve is unsafe to optimize as if it contained
        # a peak. Fall back to the continuous range sweep in that case.
        if direction is None:
            return None

        step = initial_step
        for _ in range(6):
            self._check_focus_cancelled()
            step = min(half_range, step * 1.6)
            target_z = best[0] + direction * step
            bounded_target = min(z_end, max(z_start, target_z))
            if abs(bounded_target - best[0]) < 0.05:
                break
            candidate = evaluate(bounded_target)
            margin = self._focus_comparison_margin(
                (previous[1], best[1], candidate[1]),
                peak_found_threshold,
            )
            if candidate[1] <= best[1] - margin:
                return list(cache.values()), (
                    min(previous[0], candidate[0]),
                    max(previous[0], candidate[0]),
                )
            if candidate[1] > best[1]:
                previous, best = best, candidate
            # Across a shallow plateau, retain the last point before the best
            # so a later clear drop still brackets the whole candidate peak.

        best_sample = max(cache.values(), key=lambda sample: sample[1])
        if best_sample[0] in (float(z_start), float(z_end)):
            return list(cache.values()), (
                max(z_start, best_sample[0] - initial_step),
                min(z_end, best_sample[0] + initial_step),
            )
        return None

    @staticmethod
    def _coarse_peak_has_passed(
        samples,
        peak_found_threshold,
        coarse_drop_ratio,
        minimum_travel,
    ):
        if len(samples) < 6:
            return False
        scores = np.asarray(
            [sample[1] for sample in samples],
            dtype=np.float64,
        )
        smoothed = np.convolve(scores, np.ones(3) / 3, mode="valid")
        best_smoothed_index = int(np.argmax(smoothed))
        best_sample_index = best_smoothed_index + 1
        if best_sample_index >= len(samples) - 3:
            return False
        best_score = float(smoothed[best_smoothed_index])
        baseline = float(np.min(scores[:best_sample_index + 1]))
        required_drop = max(
            float(peak_found_threshold),
            abs(best_score) * float(coarse_drop_ratio),
        )
        recent_score = float(np.mean(scores[-3:]))
        peak_is_prominent = (
            best_score - baseline >= float(peak_found_threshold)
        )
        travelled_past_peak = (
            abs(samples[-1][0] - samples[best_sample_index][0])
            >= float(minimum_travel)
        )
        return (
            peak_is_prominent
            and travelled_past_peak
            and recent_score <= best_score - required_drop
        )

    def _coarse_focus_sweep(
        self,
        z_start,
        z_end,
        z_velocity,
        peak_found_threshold,
        profile,
    ):
        self._move_to_focus_z(z_start, wait=True)
        starting_frame = self._latest_frame_data()
        last_frame_id = (
            starting_frame["frame_id"]
            if starting_frame is not None
            else -1
        )
        self.stage.set_z_velocity(z_velocity)
        self._check_focus_cancelled()
        if not self.stage.move_to_z(
            z_end,
            wait=False,
            cancel_check=self._check_focus_cancelled,
        ):
            raise RuntimeError(
                "The stage could not start the coarse autofocus sweep."
            )

        sweep_distance = abs(float(z_end) - float(z_start))
        expected_seconds = sweep_distance / max(1.0, float(z_velocity))
        deadline = time.monotonic() + min(
            120.0,
            max(5.0, expected_seconds * 2 + 5.0),
        )
        samples = []
        stable_idle_samples = 0
        minimum_travel = max(
            profile["target_precision"] * 3,
            profile["initial_fine_radius"] * 0.35,
        )

        while True:
            self._check_focus_cancelled()
            data = self._wait_for_new_frame_data(
                last_frame_id,
                timeout=0.05,
            )
            if data is not None:
                last_frame_id = data["frame_id"]
                frame_z = float(data["z"])
                if (
                    min(z_start, z_end) - 1.0
                    <= frame_z
                    <= max(z_start, z_end) + 1.0
                ):
                    sharpness, elapsed_ms = self.find_sharpness(
                        data["frame"]
                    )
                    samples.append((frame_z, float(sharpness)))
                    best_z, best_sharpness = max(
                        samples,
                        key=lambda sample: sample[1],
                    )
                    print(
                        f"Coarse focus | Z: {frame_z:>10.2f} | "
                        f"Sharpness: {sharpness:>10.3f} | "
                        f"Best Z: {best_z:>10.2f} | "
                        f"Best: {best_sharpness:>10.3f} | "
                        f"Metric: {elapsed_ms:>7.2f}ms"
                    )
                    if self._coarse_peak_has_passed(
                        samples,
                        peak_found_threshold,
                        profile["coarse_drop_ratio"],
                        minimum_travel,
                    ):
                        print(
                            "Coarse peak bracketed; stopping the sweep early"
                        )
                        self.stage.stop_z()
                        self.stage.wait_until_not_busy(
                            cancel_check=self._check_focus_cancelled
                        )
                        break

            is_busy = (
                self.stage.is_z_busy(strict=True)
                if hasattr(self.stage, "is_z_busy")
                else self.stage.is_busy()
            )
            if is_busy:
                stable_idle_samples = 0
            else:
                stable_idle_samples += 1
                if stable_idle_samples >= 3:
                    break

            if time.monotonic() >= deadline:
                self.stage.stop_z()
                self.stage.wait_until_not_busy()
                raise TimeoutError("The coarse autofocus sweep timed out.")

        if len(samples) >= 3:
            return samples

        print("Too few live coarse frames; using settled fallback samples")
        fallback_samples = []
        for target_z in (z_start, (z_start + z_end) / 2, z_end):
            score, actual_z = self._measure_focus_at(
                target_z,
                frame_count=1,
            )
            fallback_samples.append((actual_z, score))
        return fallback_samples

    @staticmethod
    def _quadratic_peak(samples):
        if len(samples) != 3:
            return None
        ordered = sorted(samples)
        x_values = np.asarray(
            [sample[0] for sample in ordered],
            dtype=np.float64,
        )
        y_values = np.asarray(
            [sample[1] for sample in ordered],
            dtype=np.float64,
        )
        if len(np.unique(x_values)) != 3:
            return None
        coefficient, slope, _ = np.polyfit(x_values, y_values, 2)
        if not np.isfinite(coefficient) or coefficient >= 0:
            return None
        vertex = -slope / (2 * coefficient)
        if not x_values[0] <= vertex <= x_values[-1]:
            return None
        return float(vertex)

    def _refine_focus_peak(
        self,
        coarse_samples,
        z_start,
        z_end,
        profile,
        focus_bounds=None,
        settled_samples=None,
    ):
        finite_samples = [
            (float(z), float(score))
            for z, score in coarse_samples
            if np.isfinite(z) and np.isfinite(score)
        ]
        if not finite_samples:
            raise RuntimeError("No valid focus frames were measured.")

        coarse_best_z, _ = max(
            finite_samples,
            key=lambda sample: sample[1],
        )
        ordered_z = sorted({sample[0] for sample in finite_samples})
        spacings = [
            right - left
            for left, right in zip(ordered_z, ordered_z[1:])
            if right > left
        ]
        observed_spacing = (
            float(np.median(spacings))
            if spacings
            else 0.0
        )
        if focus_bounds is None:
            radius = min(
                (z_end - z_start) / 2,
                max(
                    profile["initial_fine_radius"],
                    observed_spacing * 2,
                    profile["target_precision"] * 4,
                ),
            )
            lower = max(float(z_start), coarse_best_z - radius)
            upper = min(float(z_end), coarse_best_z + radius)
        else:
            lower = max(float(z_start), float(min(focus_bounds)))
            upper = min(float(z_end), float(max(focus_bounds)))
            if upper <= lower:
                raise RuntimeError("The adaptive focus bracket is empty.")
        target_precision = float(profile["target_precision"])
        frame_count = int(profile["fine_frame_count"])
        cache = {
            round(float(z), 1): (float(z), float(score))
            for z, score in (settled_samples or ())
            if (
                np.isfinite(z)
                and np.isfinite(score)
                and lower <= float(z) <= upper
            )
        }

        def evaluate(target_z):
            target_z = min(upper, max(lower, float(target_z)))
            target_z = round(target_z, 1)
            if target_z not in cache:
                score, actual_z = self._measure_focus_at(
                    target_z,
                    frame_count=frame_count,
                )
                cache[target_z] = (actual_z, score)
            return cache[target_z]

        inverse_phi = (math.sqrt(5) - 1) / 2
        a, b = lower, upper
        c = b - inverse_phi * (b - a)
        d = a + inverse_phi * (b - a)
        measured_c = evaluate(c)
        measured_d = evaluate(d)

        iterations = 0
        while (
            b - a > target_precision * 2
            and iterations < self.MAX_REFINEMENT_ITERATIONS
        ):
            self._check_focus_cancelled()
            if measured_c[1] >= measured_d[1]:
                b, d, measured_d = d, c, measured_c
                c = b - inverse_phi * (b - a)
                measured_c = evaluate(c)
            else:
                a, c, measured_c = c, d, measured_d
                d = a + inverse_phi * (b - a)
                measured_d = evaluate(d)
            iterations += 1

        evaluate((a + b) / 2)
        measured = sorted(
            {
                actual_z: score
                for actual_z, score in cache.values()
            }.items()
        )
        best_index = max(
            range(len(measured)),
            key=lambda index: measured[index][1],
        )
        if 0 < best_index < len(measured) - 1:
            vertex = self._quadratic_peak(
                measured[best_index - 1:best_index + 2]
            )
            if vertex is not None:
                evaluate(vertex)

        best_z, _ = max(
            cache.values(),
            key=lambda sample: sample[1],
        )
        for target_z in (
            best_z - target_precision,
            best_z,
            best_z + target_precision,
        ):
            if z_start <= target_z <= z_end:
                evaluate(target_z)
        return max(cache.values(), key=lambda sample: sample[1])

    def auto_focus(
        self,
        focus_range=3000,
        z_velo=500,
        z_accel=10000,
        peak_found_threshold=100,
    ):
        if (
            isinstance(focus_range, bool)
            or not isinstance(focus_range, (int, float))
            or focus_range <= 0
        ):
            raise ValueError("The autofocus range must be positive.")
        if z_velo <= 0 or z_accel <= 0:
            raise ValueError(
                "Autofocus velocity and acceleration must be positive."
            )
        if peak_found_threshold < 0:
            raise ValueError("The peak threshold cannot be negative.")

        start_time = time.perf_counter()
        default_z_velo = self.stage.get_z_velocity()
        default_z_accel = self.stage.get_z_acceleration()
        current_z = float(self.stage.get_z_position(strict=True))
        magnification, profile = self._focus_profile(focus_range)
        z_start = current_z - float(focus_range)
        z_end = current_z + float(focus_range)

        print(
            f"Starting adaptive autofocus | Magnification: {magnification} | "
            f"Current Z: {current_z:.1f} | Range: +/-{focus_range:g} | "
            f"Target precision: {profile['target_precision']:g}"
        )

        try:
            self.stage.set_z_velocity(z_velo)
            self.stage.set_z_acceleration(z_accel)
            adaptive_result = self._adaptive_focus_bracket(
                current_z,
                z_start,
                z_end,
                peak_found_threshold,
                profile,
            )
            if adaptive_result is None:
                print(
                    "Local focus probes were inconclusive; "
                    "running the continuous coarse sweep"
                )
                coarse_samples = self._coarse_focus_sweep(
                    z_start,
                    z_end,
                    z_velo,
                    peak_found_threshold,
                    profile,
                )
                focus_bounds = None
                settled_samples = None
            else:
                coarse_samples, focus_bounds = adaptive_result
                # Local probes use one frame for speed. Re-measure the fine
                # bracket with the magnification-specific frame average so a
                # single noisy probe cannot become the final focus position.
                settled_samples = None
                print(
                    "Local focus peak bracketed | "
                    f"Z: {focus_bounds[0]:.2f} to {focus_bounds[1]:.2f}"
                )
            self._check_focus_cancelled()

            fine_velocity = max(
                1,
                min(int(z_velo), int(profile["fine_velocity"])),
            )
            self.stage.set_z_velocity(fine_velocity)
            best_z, best_sharpness = self._refine_focus_peak(
                coarse_samples,
                z_start,
                z_end,
                profile,
                focus_bounds=focus_bounds,
                settled_samples=settled_samples,
            )
            self._move_to_focus_z(best_z, wait=True)
            final_z = float(self.stage.get_z_position(strict=True))

            time_taken = time.perf_counter() - start_time
            print(
                f"Autofocus complete | Best Z: {best_z:.2f} | "
                f"Stage Z: {final_z:.2f} | "
                f"Sharpness: {best_sharpness:.3f} | "
                f"Time: {time_taken:.2f}s"
            )
            return best_z
        except FocusCancelled:
            self.stage.stop_z()
            self.stage.wait_until_not_busy()
            print("Autofocus stopped")
            return current_z
        finally:
            self.stage.set_z_velocity(default_z_velo)
            self.stage.set_z_acceleration(default_z_accel)

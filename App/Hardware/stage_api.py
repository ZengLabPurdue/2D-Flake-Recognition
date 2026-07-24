from ctypes import WinDLL, create_string_buffer
import os
import threading
import time

class stage:
    DEFAULT_WAIT_TIMEOUT_SECONDS = 120
    MOTION_POLL_INTERVAL_SECONDS = 0.01
    POSITION_TOLERANCE_UM = 1.0
    POSITION_STABLE_SAMPLES = 3

    def __init__(self, port_num, sdk_path):
        self.port_num = port_num
        self.sdk_path = sdk_path

        self.velocity = 2600
        self.acceleration = 134442

        self.z_velocity = 500
        self.z_acceleration = 10000

        self.xy_step_size = 1000
        self.z_step_size = 500

        self.x = 0
        self.y = 0
        self.z = 0.0
        self.last_confirmed_xy = None
        self.motion_sequence = 0
        # The camera callback, UI controls, and scan worker all query the same
        # SDK session and shared response buffer.  Keep each command/response
        # pair atomic so their replies cannot overwrite one another.
        self._cmd_lock = threading.RLock()

        self._connect()
        self.initialize_stage()

    def _connect(self):
        print("Starting prior controller...")

        dll_folder = os.path.dirname(self.sdk_path)
        os.environ["PATH"] = dll_folder + os.pathsep + os.environ.get("PATH", "")

        if not os.path.exists(self.sdk_path):
            raise RuntimeError("DLL could not be loaded.")

        self.sdk = WinDLL(self.sdk_path, winmode=0)
        self.rx = create_string_buffer(1000)

        ret = self.sdk.PriorScientificSDK_Initialise()
        if ret:
            raise RuntimeError(f"Could not initialize Prior SDK. Error code: {ret}")

        print("Connecting to prior controller...")

        self.sdk.PriorScientificSDK_Version(self.rx)
        self.session_id = self.sdk.PriorScientificSDK_OpenNewSession()

        self.cmd("dll.apitest 33 goodresponse")
        self.cmd("dll.apitest -300 stillgoodresponse")

        self.cmd(f"controller.connect {self.port_num}")

    def cmd(self, msg):
        with self._cmd_lock:
            ret = self.sdk.PriorScientificSDK_cmd(
                self.session_id,
                create_string_buffer(msg.encode()),
                self.rx
            )

            response = self.rx.value.decode().strip()

        return ret, response

    def initialize_stage(self):
        print("Initializing prior controller...")

        self.get_position()

        self.backlash_en, self.backlash_dist = self.get_backlash()

        self.wait_until_not_busy()

        self.set_acceleration(self.acceleration)
        self.set_z_acceleration(self.z_acceleration)

        self.set_velocity(self.velocity)
        self.set_z_velocity(self.z_velocity)

        print("Prior controller setup complete!")

    def _controller_busy(self, command, strict=False):
        return_code, response = self.cmd(command)
        if isinstance(response, str):
            response = response.strip()
        try:
            busy_status = int(response, 10)
            if busy_status < 0:
                raise ValueError("negative busy status")
        except (TypeError, ValueError):
            busy_status = None

        if return_code or busy_status is None:
            if strict:
                raise RuntimeError(
                    "Could not read controller busy state from "
                    f"{command!r}: {response!r}"
                )
            # An unreadable controller state must never be treated as idle.
            return True
        # Prior reports a motion bitmask: 0 is idle and values such as
        # 1/2/3 (XY axes) and 4 (Z) are valid busy states.
        return busy_status != 0

    def is_xy_busy(self, strict=False):
        return self._controller_busy(
            "controller.stage.busy.get",
            strict=strict,
        )

    def is_z_busy(self, strict=False):
        return self._controller_busy(
            "controller.z.busy.get",
            strict=strict,
        )

    def is_busy(self):
        return self.is_xy_busy() or self.is_z_busy()

    def wait_until_not_busy(self, timeout=None, cancel_check=None):
        start_time = time.monotonic()
        if timeout is None:
            timeout = self.DEFAULT_WAIT_TIMEOUT_SECONDS

        stable_idle_samples = 0
        while True:
            if cancel_check is not None:
                try:
                    cancel_check()
                except Exception:
                    # A cancelled scan must not leave a long-running move in
                    # progress while its worker unwinds.
                    try:
                        self.stop_all()
                    except Exception:
                        pass
                    raise

            if self.is_busy():
                stable_idle_samples = 0
            else:
                stable_idle_samples += 1
                if stable_idle_samples >= self.POSITION_STABLE_SAMPLES:
                    break

            if (
                timeout is not None
                and time.monotonic() - start_time >= timeout
            ):
                try:
                    self.stop_all()
                except Exception:
                    pass
                raise TimeoutError(
                    f"Stage remained busy for more than {timeout:g} seconds."
                )
            time.sleep(self.MOTION_POLL_INTERVAL_SECONDS)

        return time.monotonic() - start_time

    def wait_for_xy_target(
        self,
        x,
        y,
        timeout=None,
        cancel_check=None,
    ):
        """Wait for an XY command to reach and remain at its target.

        A controller may briefly report idle immediately after accepting a
        move. Position convergence prevents that start-up race without adding
        an unconditional settling delay.
        """
        target_x = int(x)
        target_y = int(y)
        start_time = time.monotonic()
        if timeout is None:
            timeout = self.DEFAULT_WAIT_TIMEOUT_SECONDS
        stable_samples = 0

        while True:
            self._check_wait_cancelled(cancel_check)
            busy = self.is_xy_busy(strict=True)
            current_x, current_y = self.get_xy_position(strict=True)
            at_target = (
                abs(current_x - target_x) <= self.POSITION_TOLERANCE_UM
                and abs(current_y - target_y) <= self.POSITION_TOLERANCE_UM
            )

            if not busy and at_target:
                stable_samples += 1
                if stable_samples >= self.POSITION_STABLE_SAMPLES:
                    self.last_confirmed_xy = (current_x, current_y)
                    return self.last_confirmed_xy
            else:
                stable_samples = 0

            self._raise_motion_timeout(
                start_time,
                timeout,
                f"XY stage did not reach ({target_x}, {target_y})",
            )
            time.sleep(self.MOTION_POLL_INTERVAL_SECONDS)

    def wait_for_z_target(
        self,
        z,
        timeout=None,
        cancel_check=None,
    ):
        target_z = float(z)
        start_time = time.monotonic()
        if timeout is None:
            timeout = self.DEFAULT_WAIT_TIMEOUT_SECONDS
        stable_samples = 0

        while True:
            self._check_wait_cancelled(cancel_check)
            busy = self.is_z_busy(strict=True)
            current_z = self.get_z_position(strict=True)
            at_target = (
                abs(current_z - target_z)
                <= self.POSITION_TOLERANCE_UM
            )

            if not busy and at_target:
                stable_samples += 1
                if stable_samples >= self.POSITION_STABLE_SAMPLES:
                    return current_z
            else:
                stable_samples = 0

            self._raise_motion_timeout(
                start_time,
                timeout,
                f"Z stage did not reach {target_z:g}",
            )
            time.sleep(self.MOTION_POLL_INTERVAL_SECONDS)

    def _check_wait_cancelled(self, cancel_check):
        if cancel_check is None:
            return
        try:
            cancel_check()
        except Exception:
            try:
                self.stop_all()
            except Exception:
                pass
            raise

    def _raise_motion_timeout(self, start_time, timeout, message):
        if (
            timeout is None
            or time.monotonic() - start_time < timeout
        ):
            return
        try:
            self.stop_all()
        except Exception:
            pass
        raise TimeoutError(f"{message} within {timeout:g} seconds.")

    def stop_all(self):
        self.stop_x()
        self.stop_y()
        self.stop_z()

    def _note_motion_command(self):
        self.motion_sequence += 1
        return self.motion_sequence

    # ---------------- Position ----------------

    def get_xy_position(self, strict=False):
        return_code, response = self.cmd(
            "controller.stage.position.get"
        )

        try:
            if return_code:
                raise ValueError(f"SDK error {return_code}")
            parts = response.split(",")
            if len(parts) < 2:
                raise ValueError("missing XY values")
            x = int(float(parts[0]))
            y = int(float(parts[1]))
        except (TypeError, ValueError) as exc:
            if strict:
                raise RuntimeError(
                    f"Could not parse XY position: {response!r}"
                ) from exc
            return self.x, self.y

        self.x = x
        self.y = y
        return self.x, self.y

    def get_position(self, strict=False):
        self.get_xy_position(strict=strict)

        self.get_z_position(strict=strict)

        return self.x, self.y, self.z

    def get_z_position(self, strict=False):
        return_code, response = self.cmd(
            "controller.z.position.get"
        )
        response = response.strip()

        try:
            if return_code:
                raise ValueError(f"SDK error {return_code}")
            z = int(float(response)) / 10
        except (TypeError, ValueError) as exc:
            if strict:
                raise RuntimeError(
                    f"Could not parse Z position: {response!r}"
                ) from exc
            return self.z

        self.z = z
        return self.z

    # ---------------- Absolute Movement ----------------

    def move_to_xy(
        self,
        x,
        y,
        wait=True,
        timeout=None,
        cancel_check=None,
    ):
        if self.is_busy():
            return False

        target_x = int(x)
        target_y = int(y)
        return_code, response = self.cmd(
            f"controller.stage.goto-position {target_x} {target_y}"
        )
        if return_code:
            raise RuntimeError(
                "The controller rejected the XY move "
                f"to ({target_x}, {target_y}): {response!r}"
            )
        self._note_motion_command()

        if wait:
            self.wait_for_xy_target(
                target_x,
                target_y,
                timeout=timeout,
                cancel_check=cancel_check,
            )
        return True

    def move_to_z(
        self,
        z,
        wait=True,
        timeout=None,
        cancel_check=None,
    ):
        if self.is_busy():
            return False

        z_prior_units = int(float(z) * 10)
        return_code, response = self.cmd(
            f"controller.z.goto-position {z_prior_units}"
        )
        if return_code:
            raise RuntimeError(
                f"The controller rejected the Z move to {z}: {response!r}"
            )
        self._note_motion_command()

        if wait:
            self.wait_for_z_target(
                z,
                timeout=timeout,
                cancel_check=cancel_check,
            )
        return True

    # ---------------- Step Movement ----------------

    def set_xy_step_size(self, step_size):
        self.xy_step_size = int(step_size)

    def set_z_step_size(self, step_size):
        self.z_step_size = int(step_size)

    def step_forward(self):
        self.get_position()
        return self.move_to_xy(self.x, self.y - self.xy_step_size)

    def step_backward(self):
        self.get_position()
        return self.move_to_xy(self.x, self.y + self.xy_step_size)

    def step_left(self):
        self.get_position()
        return self.move_to_xy(self.x + self.xy_step_size, self.y)

    def step_right(self):
        self.get_position()
        return self.move_to_xy(self.x - self.xy_step_size, self.y)

    def step_up(self):
        self.get_position()
        return self.move_to_z(self.z - self.z_step_size)

    def step_down(self):
        self.get_position()
        return self.move_to_z(self.z + self.z_step_size)

    # ---------------- Continuous Movement ----------------

    def start_forward(self):
        self.cmd(f"controller.stage.move-at-velocity 0 -{self.velocity}")
        self._note_motion_command()

    def start_backward(self):
        self.cmd(f"controller.stage.move-at-velocity 0 {self.velocity}")
        self._note_motion_command()

    def start_left(self):
        self.cmd(f"controller.stage.move-at-velocity {self.velocity} 0")
        self._note_motion_command()

    def start_right(self):
        self.cmd(f"controller.stage.move-at-velocity -{self.velocity} 0")
        self._note_motion_command()

    def stop_x(self):
        self.cmd("controller.stage.move-at-velocity 0 0")

    def stop_y(self):
        self.cmd("controller.stage.move-at-velocity 0 0")

    def start_up(self):
        self.cmd(f"controller.z.move-at-velocity -{self.z_velocity}")
        self._note_motion_command()

    def start_down(self):
        self.cmd(f"controller.z.move-at-velocity {self.z_velocity}")
        self._note_motion_command()

    def stop_z(self):
        self.cmd("controller.z.move-at-velocity 0")

    # ---------------- Speed / Acceleration ----------------

    def set_velocity(self, velocity):
        self.velocity = int(velocity)
        self.cmd(f"controller.stage.speed.set {self.velocity}")
        return self.get_velocity()

    def get_velocity(self):
        _, response = self.cmd("controller.stage.speed.get")
        self.velocity = int(float(response))
        return self.velocity

    def set_acceleration(self, acceleration):
        self.acceleration = int(acceleration)
        self.cmd(f"controller.stage.acc.set {self.acceleration}")

    def get_acceleration(self):
        _, response = self.cmd("controller.stage.acc.get")
        self.velocity = int(float(response))
        return self.velocity

    def set_z_velocity(self, velocity):
        self.z_velocity = int(velocity)
        self.cmd(f"controller.z.speed.set {self.z_velocity}")
        return self.get_z_velocity()

    def get_z_velocity(self):
        _, response = self.cmd("controller.z.speed.get")
        self.z_velocity = int(float(response))
        return self.z_velocity

    def set_z_acceleration(self, acceleration):
        self.z_acceleration = int(acceleration)
        self.cmd(f"controller.z.acc.set {self.z_acceleration}")

    def get_z_acceleration(self):
        _, response = self.cmd("controller.z.acc.get")
        self.velocity = int(float(response))
        return self.velocity

    # ---------------- Origin / Backlash ----------------

    def set_origin(self):
        self.cmd("controller.stage.position.set 0 0")
        self.get_position()

    def set_z_zero(self):
        self.cmd("controller.z.position.set 0")
        self.get_z_position()

    def get_backlash(self):
        _, response = self.cmd("controller.stage.backlash.get")
        parts = response.split(",")
        return int(parts[0]), int(parts[1])

    def set_backlash_enabled(self, enabled):
        self.backlash_en = int(enabled)
        self.cmd(f"controller.stage.backlash.set {self.backlash_en} {self.backlash_dist}")

    def set_backlash_distance(self, distance):
        self.backlash_dist = int(distance)
        self.cmd(f"controller.stage.backlash.set {self.backlash_en} {self.backlash_dist}")

    # ---------------- Shutdown ----------------

    def disconnect(self, timeout=5):
        try:
            self.wait_until_not_busy(timeout=timeout)
        except TimeoutError:
            # Shutdown should remain finite even if the controller never
            # clears its busy flag.
            self.stop_all()
        finally:
            self.cmd("controller.disconnect")

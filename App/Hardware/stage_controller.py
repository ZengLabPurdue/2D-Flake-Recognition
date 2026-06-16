from ctypes import WinDLL, create_string_buffer
import os
import sys
import time
from dataclasses import dataclass

class StageController:
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

    def is_busy(self):
        stage_busy = self.cmd("controller.stage.busy.get")[1]
        z_busy = self.cmd("controller.z.busy.get")[1]

        return stage_busy != "0" or z_busy != "0"

    def wait_until_not_busy(self):
        start_time = time.time()

        while self.is_busy():
            time.sleep(0.05)

        return time.time() - start_time

    def stop_all(self):
        self.stop_x()
        self.stop_y()
        self.stop_z()

    # ---------------- Position ----------------

    def get_position(self):
        _, response = self.cmd("controller.stage.position.get")

        try:
            parts = response.split(",")
            self.x = int(float(parts[0]))
            self.y = int(float(parts[1]))
        except Exception:
            print(f"Could not parse XY position: {response!r}")
            self.stop_all()

        self.get_z_position()

        return self.x, self.y, self.z

    def get_z_position(self):
        _, response = self.cmd("controller.z.position.get")
        response = response.strip()

        try:
            self.z = int(float(response)) / 10
        except ValueError:
            print(f"Could not parse Z position: {response!r}")

        return self.z

    # ---------------- Absolute Movement ----------------

    def move_to_xy(self, x, y, wait=True):
        if self.is_busy():
            return False

        self.cmd(f"controller.stage.goto-position {int(x)} {int(y)}")

        if wait:
            self.wait_until_not_busy()

        self.get_position()
        return True

    def move_to_z(self, z, wait=True):
        if self.is_busy():
            return False

        z_prior_units = int(float(z) * 10)
        self.cmd(f"controller.z.goto-position {z_prior_units}")

        if wait:
            self.wait_until_not_busy()

        self.get_z_position()
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

    def start_backward(self):
        self.cmd(f"controller.stage.move-at-velocity 0 {self.velocity}")

    def start_left(self):
        self.cmd(f"controller.stage.move-at-velocity {self.velocity} 0")

    def start_right(self):
        self.cmd(f"controller.stage.move-at-velocity -{self.velocity} 0")

    def stop_x(self):
        self.cmd("controller.stage.move-at-velocity 0 0")

    def stop_y(self):
        self.cmd("controller.stage.move-at-velocity 0 0")

    def start_up(self):
        self.cmd(f"controller.z.move-at-velocity -{self.z_velocity}")

    def start_down(self):
        self.cmd(f"controller.z.move-at-velocity {self.z_velocity}")

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

    def disconnect(self):
        self.wait_until_not_busy()
        self.cmd("controller.disconnect")
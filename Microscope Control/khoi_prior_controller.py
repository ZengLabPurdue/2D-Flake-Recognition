from ctypes import WinDLL, create_string_buffer
import os
import sys
import time

class Prior_Controller():

    def __init__(self, port_num, sdk_path):
        self.port_num = port_num
        self.path = sdk_path
        
        self.velocity = 2600
        self.acceleration = 134442
        self.z_velocity = 1000
        self.z_acceleration = 57100

        print("Starting prior controller...")

        dll_folder = os.path.dirname(self.path)
        os.environ["PATH"] = dll_folder + os.pathsep + os.environ.get("PATH", "")

        if os.path.exists(self.path):
            global SDKPrior
            SDKPrior = WinDLL(self.path, winmode=0)
        else:
            raise RuntimeError("DLL could not be loaded.")
        try:
            global rx
            rx = create_string_buffer(1000)

            ret = SDKPrior.PriorScientificSDK_Initialise()
            if ret:
                #print(f"Error initialising {ret}")
                sys.exit()
            else:
                pass

            print("Connecting to prior controller...")

            ret = SDKPrior.PriorScientificSDK_Version(rx)

            global sessionID
            sessionID = SDKPrior.PriorScientificSDK_OpenNewSession()

            ret = SDKPrior.PriorScientificSDK_cmd(
                sessionID, create_string_buffer(b"dll.apitest 33 goodresponse"), rx
            )

            ret = SDKPrior.PriorScientificSDK_cmd(
                sessionID, create_string_buffer(b"dll.apitest -300 stillgoodresponse"), rx
            )
            self.cmd(f"controller.connect {self.port_num}")

            print("Initializing prior controller...")

            self.get_curr_pos()
            self.backlash_en, self.backlash_dist = self.get_backlash()

            self.cmd(f"controller.z.acc.get")
            self.cmd(f"controller.z.speed.get")

            self.wait_until_not_busy()
            self.cmd(f"controller.stage.acc.set {self.acceleration}")
            self.cmd(f"controller.z.acc.set {self.z_acceleration}")

            self.cmd(f"controller.stage.speed.set {self.velocity}")
            self.cmd(f"controller.z.speed.set {self.z_velocity}")

            print("Prior controller setup complete!")

        except Exception as e:
            print(e)

    def cmd(self, msg):
        #print(msg)
        ret = SDKPrior.PriorScientificSDK_cmd(
            sessionID, create_string_buffer(msg.encode()), rx
        )
        '''
        if ret:
            print(f"Api error {ret}")
        else:
            print(f"OK {rx.value.decode()}")
        '''
        return ret, rx.value.decode()
    
    def wait_until_not_busy(self):
        start_time = time.time()
        while self.is_busy():
            pass
        end_time = time.time() - start_time
        return end_time
        
    def is_busy(self):
        if (self.cmd("controller.stage.busy.get")[1] == "2") | (self.cmd("controller.z.busy.get")[1] == "4") | (self.cmd("controller.stage.busy.get")[1] == "1") | (self.cmd("controller.stage.busy.get")[1] == "3"):
            return True
        else: return False

    def set_velocity(self, velocity):
        self.wait_until_not_busy()
        self.velocity = velocity
        self.cmd(f"controller.stage.speed.set {self.velocity}")
        self.cmd("controller.stage.speed.get")

    def set_acceleration(self, acceleration):
        self.wait_until_not_busy()
        self.acceleration = acceleration
        self.cmd(f"controller.stage.acc.set {self.acceleration}")
        self.cmd("controller.stage.acc.get")

    def go_to_pos(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
        self.wait_until_not_busy()
        self.cmd(f"controller.stage.goto-position {self.x} {self.y}")
        self.cmd("controller.stage.speed.get")
        # time.sleep(1)

    def get_curr_pos(self):
        self.wait_until_not_busy()
        position = self.cmd("controller.stage.position.get")
        self.x = int(position[1].split(",")[0])
        self.y = int(position[1].split(",")[1])
        self.z = self.get_curr_z_pos()

    def set_z_velocity(self, velocity):
        self.wait_until_not_busy()
        self.z_velocity = velocity
        self.cmd(f"controller.z.speed.set {self.velocity}")
        self.cmd("controller.z.speed.get")

    def set_z_acceleration(self, acceleration):
        self.wait_until_not_busy()
        self.z_acceleration = acceleration
        self.cmd(f"controller.z.acc.set {self.acceleration}")
        self.cmd("controller.z.acc.get")

    def go_to_z_pos(self, new_z):
        self.z = new_z * 10
        self.wait_until_not_busy()
        self.cmd(f"controller.z.goto-position {self.z}")
        self.cmd("controller.z.speed.get")
        # time.sleep(1)

    def get_curr_z_pos(self):
        self.wait_until_not_busy()
        position = self.cmd("controller.z.position.get")
        return int(position[1]) / 10
    
    def set_origin(self):
        self.cmd("controller.stage.position.set 0 0")

    def start_forward_x_motor(self):
        self.cmd(f"controller.stage.move-at-velocity {self.velocity} 0")
    def start_backward_x_motor(self):
        self.cmd(f"controller.stage.move-at-velocity -{self.velocity} 0")
    def stop_x_motor(self):
        self.cmd(f"controller.stage.move-at-velocity 0 0")

    def start_forward_y_motor(self):
        self.cmd(f"controller.stage.move-at-velocity 0 -{self.velocity}")
    def start_backward_y_motor(self):
        self.cmd(f"controller.stage.move-at-velocity 0 {self.velocity}")
    def stop_y_motor(self):
        self.cmd(f"controller.stage.move-at-velocity 0 0")

    def start_forward_z_motor(self):
        self.cmd(f"controller.z.move-at-velocity {self.z_velocity}")
    def start_backward_z_motor(self):
        self.cmd(f"controller.z.move-at-velocity -{self.z_velocity}")
    def stop_z_motor(self):
        self.cmd(f"controller.z.move-at-velocity 0")

    def get_backlash(self):
        backlash = self.cmd(f"controller.stage.backlash.get")[1]
        backlash = backlash.split(",")
        return int(backlash[0]), int(backlash[1]) #enable, backlash correction

    def get_z_backlash(self):
        backlash = self.cmd(f"controller.z.backlash.get")[1]
        backlash = backlash.split(",")
        return int(backlash[0]), int(backlash[1]) #enable, backlash correction
    
    def set_backlash_en(self, backlash_en):
        self.backlash_en = backlash_en
        self.cmd(f"controller.stage.backlash.set {self.backlash_en} {self.backlash_dist}")

    def set_backlash_dist(self, backlash_dist):
        self.backlash_dist = backlash_dist
        self.cmd(f"controller.stage.backlash.set {self.backlash_en} {self.backlash_dist}")

    def set_z_backlash_dist(self, backlash_dist):
        self.z_backlash_dist = backlash_dist
        self.cmd(f"controller.z.backlash.set {self.z_backlash_en} {self.z_backlash_dist}")

    def set_z_backlash_en(self, backlash_en):
        self.z_backlash_en = backlash_en
        self.cmd(f"controller.z.backlash.set {self.z_backlash_en} {self.z_backlash_dist}")

    def disconnect(self):
        self.wait_until_not_busy()
        self.cmd("controller.disconnect")
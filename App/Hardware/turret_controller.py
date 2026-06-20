from config import RELATIVE_Z
from Hardware.turret_api import turret

class TurretController:

    def __init__(
        self, 
        app,
        stage,
        turret_port,
        start_auto_focus_thread,
    ):
        self.app = app
        self.stage = stage
        self.turret = turret(turret_port)
        self.start_auto_focus_thread = start_auto_focus_thread

    def get_position(self):
        return self.turret.check_position()
    
    def turn_to_position(self, position):
        self.turret.turn_to_position(position)

    def change_objective(self, position):
        objective_map = {
            1: ("2X", RELATIVE_Z["2X"]),
            2: ("10X", RELATIVE_Z["10X"]),
            3: ("20X", RELATIVE_Z["20X"]),
            4: (None, None),
            5: ("100X", RELATIVE_Z["100X"]),
        }

        self.app.disable_buttons()

        try:
            current_position = self.get_position()

            if position == current_position:
                return

            current_z = self.stage.get_z_position()

            _, current_rel_z = objective_map.get(current_position, (None, 0))
            magnification, target_rel_z = objective_map.get(position, (None, 0))

            change_z = target_rel_z - current_rel_z

            self.stage.move_to_z(current_z + change_z)
            self.turn_to_position(position)

            if position == 1:
                self.start_auto_focus_thread(focus_range=500, z_velo=500, z_accel=10000, peak_found_threshold=100)

            elif position == 2:
                self.start_auto_focus_thread(focus_range=200, z_velo=50, z_accel=10000, peak_found_threshold=20)
                pass

            elif position == 3:
                self.start_auto_focus_thread(focus_range=100, z_velo=10, z_accel=10000, peak_found_threshold=20)
                pass

            elif position == 4:
                pass

            elif position == 5:
                #self.start_auto_focus_thread(focus_range=10, z_velo=3, z_accel=10000, peak_found_threshold=10)
                pass

            self.app.set_magnification(magnification)

        finally:
            self.app.enable_buttons()
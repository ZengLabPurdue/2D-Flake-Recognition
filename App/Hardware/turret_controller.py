from config import RELATIVE_Z
from Hardware.turret_api import turret

class TurretController:

    def __init__(
        self, 
        stage,
        turret,
        auto_focus,
        enable_buttons,
        disable_buttons,
        set_magnification
    ):
        self.stage = stage
        self.turret = turret
        self.auto_focus = auto_focus
        self.enable_buttons = enable_buttons
        self.disable_buttons = disable_buttons
        self.set_magnification = set_magnification

    def get_position(self):
        return self.turret.check_position()
    
    def turn_to_position(self, position):
        self.turret.turn_to_position(position)

    def change_objective(self, position):
        objective_map = {
            1: ("2x", RELATIVE_Z["2x"]),
            2: ("10x", RELATIVE_Z["10x"]),
            3: ("20x", RELATIVE_Z["20x"]),
            4: (None, None),
            5: ("100x", RELATIVE_Z["100x"]),
        }

        self.disable_buttons()

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
                self.auto_focus()

            elif position == 2:
                # self.auto_focus(start_range=500, accuracy=10, steps=20)
                pass

            elif position == 3:
                self.auto_focus(start_range=200, accuracy=5, steps=20)

            elif position == 4:
                # self.auto_focus(start_range=50, accuracy=2, steps=10)
                pass

            self.set_magnification(magnification)

        finally:
            self.enable_buttons()
import config
from turret_api import turret

class TurretController:

    def __init__(
        self, 
        app,
        stage,
        auto_focus,
        objective_panel,
    ):
        self.app = app
        self.stage = stage
        self.auto_focus = auto_focus
        self.objective_panel = objective_panel

    def get_position(self):
        return turret.check_position()

    def change_objective(self, position):
        objective_map = {
            1: ("2x", config.RELATIVE_2X_Z),
            2: ("10x", config.RELATIVE_10X_Z),
            3: ("20x", config.RELATIVE_20X_Z),
            4: (None, config.RELATIVE_20X_Z),
            5: ("100x", config.RELATIVE_100X_Z),
        }

        self.app.disable_buttons()

        try:
            current_position = turret.check_position()

            if position == current_position:
                return

            current_z = self.stage.get_z_position()

            _, current_rel_z = objective_map.get(current_position, (None, 0))
            magnification, target_rel_z = objective_map.get(position, (None, 0))

            change_z = target_rel_z - current_rel_z

            self.stage.move_to_z(current_z + change_z)
            turret.turn_to_position(position)

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
            self.objective_panel.objective_var.set(f"Objective: {position}")

        finally:
            self.app.enable_buttons()
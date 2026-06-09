'''
from controllers.stage_manager import StageManager
from ui.panels.stage_control_panel import StageControlPanel

class App:
    def __init__(self, root):
        self.root = root

        self.stage = StageManager(PRIOR_COM_PORT, DLL_PATH)

        self.stage_control_panel = StageControlPanel(
            parent=self.main_frame,
            root=self.root,
            stage=self.stage,
            disable_buttons=self.disable_buttons,
            enable_buttons=self.enable_buttons,
            register_button=self.buttons.append
        )

        self.panels.append({
            "name": "Stage Control Panel",
            "frame": self.stage_control_panel.frame,
            "var": BooleanVar(value=False)
        })
'''
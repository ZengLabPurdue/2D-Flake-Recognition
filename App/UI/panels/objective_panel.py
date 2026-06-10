import tkinter as tk
from tkinter import ttk

#TODO: Move into config file
RELATIVE_2X_Z = 0
RELATIVE_10X_Z = 1250
RELATIVE_20X_Z = 4300
RELATIVE_100X_Z = 4300

class ObjectiveControlPanel:
    def __init__(
        self,
        parent,
        stage,
        turret,
        get_magnification,
        set_magnification,
        auto_focus,
        enable_buttons=None,
        disable_buttons=None,
        register_button=None,
    ):
        self.parent = parent
        self.stage = stage
        self.turret = turret

        self.get_magnification = get_magnification
        self.set_magnification = set_magnification
        self.auto_focus = auto_focus

        self.enable_buttons = enable_buttons or (lambda: None)
        self.disable_buttons = disable_buttons or (lambda: None)
        self.register_button = register_button or (lambda btn: None)

        self.objective_var = tk.StringVar(value="Objective: Unknown")

        self.frame = self._build_panel()

        # Initialize turret position display.
        self.turret.turn_to_position(1)
        self.objective_var.set("Objective: 1")
        self.set_magnification("2x")

    def _build_panel(self):
        panel = tk.Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=240
        )

        background = tk.Frame(
            panel,
            bg="white",
            width=200,
            height=238
        )
        background.place(x=2, y=0)

        title = tk.Label(
            panel,
            text="Objective Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title.place(relx=0.5, y=5, anchor="n")

        objective_label = tk.Label(
            panel,
            textvariable=self.objective_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        objective_label.place(relx=0.5, y=40, anchor="n")

        style = ttk.Style()
        style.configure(
            "Custom.TButton",
            font=("TkDefaultFont", 10),
            padding=5,
            background="white",
            relief="flat"
        )

        button_panel = tk.Frame(
            panel,
            bg="white",
            width=150,
            height=150
        )
        button_panel.place(x=26, y=70)
        button_panel.pack_propagate(False)

        controls = tk.Frame(button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        self.btn1 = ttk.Button(controls, text="1", style="Custom.TButton")
        self.btn2 = ttk.Button(controls, text="2", style="Custom.TButton")
        self.btn3 = ttk.Button(controls, text="3", style="Custom.TButton")
        self.btn4 = ttk.Button(controls, text="4", style="Custom.TButton")
        self.btn5 = ttk.Button(controls, text="5", style="Custom.TButton")

        self.objective_buttons = [
            self.btn1,
            self.btn2,
            self.btn3,
            self.btn4,
            self.btn5,
        ]

        for btn in self.objective_buttons:
            self.register_button(btn)

        for r in range(3):
            controls.rowconfigure(r, weight=1)

        for c in range(2):
            controls.columnconfigure(c, weight=1)

        self.btn1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.btn2.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.btn3.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.btn4.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        self.btn5.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)

        self.btn1.bind("<ButtonPress-1>", lambda e: self.change_objective(1))
        self.btn2.bind("<ButtonPress-1>", lambda e: self.change_objective(2))
        self.btn3.bind("<ButtonPress-1>", lambda e: self.change_objective(3))
        self.btn4.bind("<ButtonPress-1>", lambda e: self.change_objective(4))
        self.btn5.bind("<ButtonPress-1>", lambda e: self.change_objective(5))

        return panel

    def change_objective(self, position):
        objective_map = {
            1: ("2x", RELATIVE_2X_Z),
            2: ("10x", RELATIVE_10X_Z),
            3: ("20x", RELATIVE_20X_Z),
            4: (None, RELATIVE_20X_Z),
            5: ("100x", RELATIVE_100X_Z),
        }

        self.disable_buttons()

        try:
            current_position = self.turret.check_position()

            if position == current_position:
                return

            current_z = self.stage.get_curr_z_pos()

            _, current_rel_z = objective_map.get(current_position, (None, 0))
            magnification, target_rel_z = objective_map.get(position, (None, 0))

            change_z = target_rel_z - current_rel_z

            self.stage.go_to_z_pos(current_z + change_z)
            self.turret.turn_to_position(position)

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
            self.objective_var.set(f"Objective: {position}")

        finally:
            self.enable_buttons()
import tkinter as tk
from tkinter import ttk

class ObjectiveControlPanel:
    def __init__(
        self,
        parent,
        app,
        stage,
        turret_controller,
    ):
        self.parent = parent
        self.stage = stage
        self.app = app
        self.turret_controller = turret_controller

        self.objective_var = tk.StringVar(value="Objective: Unknown")

        self.frame = self._build_panel()

        # Initialize turret position display.
        self.turret_controller.turn_to_position(1)
        self.objective_var.set("Objective: 1")
        self.app.set_magnification("2X")

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
        title.place(relx=0.5, y=10, anchor="n")

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
            self.app.register_button(btn)

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
        self.turret_controller.change_objective(position)
        self.objective_var.set(f"Objective: {position}")
        #self.app.clear_focus()
import tkinter as tk
from tkinter import ttk

class StageControlPanel:
    def __init__(
            self, 
            parent, 
            root, 
            app,
            stage, 
        ):

        self.parent = parent
        self.root = root
        self.app = app
        self.stage = stage

        self.hold_job = None
        self.is_hold = False

        self.frame = self._build_panel()
        self.update_position_display()

    def _build_panel(self):
        panel = tk.Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=574
        )

        background = tk.Frame(
            panel,
            bg="white",
            width=200,
            height=572
        )
        background.place(x=2, y=0)

        title_label = tk.Label(
            panel,
            text="Manual Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title_label.place(relx=0.5, y=10, anchor="n")

        label_x = 10
        entry_x = 130

        # XY step
        tk.Label(
            panel,
            text="XY Step Size (µm):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        ).place(x=label_x, y=45)

        self.xy_step_var = tk.StringVar(value=str(self.stage.xy_step_size))
        self.xy_step_entry = ttk.Entry(
            panel,
            textvariable=self.xy_step_var,
            width=8
        )
        self.xy_step_entry.place(x=entry_x, y=45)

        # XY speed
        tk.Label(
            panel,
            text="XY Speed (µm/s):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        ).place(x=label_x, y=75)

        self.xy_speed_var = tk.StringVar(value=str(self.stage.velocity))
        self.xy_speed_var.trace_add("write", self.on_speed_change_xy)

        self.xy_speed_entry = ttk.Entry(
            panel,
            textvariable=self.xy_speed_var,
            width=8
        )
        self.xy_speed_entry.place(x=entry_x, y=75)

        # X
        tk.Label(
            panel,
            text="X (µm):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        ).place(x=label_x, y=105)

        self.x_coord_var = tk.StringVar()
        self.x_coord_entry = ttk.Entry(
            panel,
            textvariable=self.x_coord_var,
            width=8
        )
        self.x_coord_entry.place(x=entry_x, y=105)

        # Y
        tk.Label(
            panel,
            text="Y (µm):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        ).place(x=label_x, y=135)

        self.y_coord_var = tk.StringVar()
        self.y_coord_entry = ttk.Entry(
            panel,
            textvariable=self.y_coord_var,
            width=8
        )
        self.y_coord_entry.place(x=entry_x, y=135)

        style = ttk.Style()
        style.configure("Normal.TButton", font="TkDefaultFont", background="white", relief="flat")
        style.configure("Arrow.TButton", font=("TkDefaultFont", 15), padding=5, background="white", relief="flat")

        self.reset_button = ttk.Button(
            panel,
            text="Set Origin",
            style="Normal.TButton",
            command=self.set_origin
        )
        self.reset_button.place(relx=0.5, y=170, anchor="n")
        self.app.register_button(self.reset_button)

        self.move_to_button = ttk.Button(
            panel,
            text="Move to (X, Y)",
            style="Normal.TButton",
            command=self.go_to_position
        )
        self.move_to_button.place(relx=0.5, y=205, anchor="n")
        self.app.register_button(self.move_to_button)

        # Arrow buttons
        button_panel = tk.Frame(panel, bg="white", width=120, height=90)
        button_panel.place(relx=0.5, y=240, anchor="n")
        button_panel.pack_propagate(False)

        controls = tk.Frame(button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        self.btn_forward = ttk.Button(controls, text="▴", style="Arrow.TButton")
        self.btn_backward = ttk.Button(controls, text="▾", style="Arrow.TButton")
        self.btn_left = ttk.Button(controls, text="◂", style="Arrow.TButton")
        self.btn_right = ttk.Button(controls, text="▸", style="Arrow.TButton")

        for btn in [self.btn_forward, self.btn_backward, self.btn_left, self.btn_right]:
            self.app.register_button(btn)

        self.btn_forward.bind("<ButtonPress-1>", self.on_press_forward)
        self.btn_forward.bind("<ButtonRelease-1>", self.on_release_forward)

        self.btn_backward.bind("<ButtonPress-1>", self.on_press_backward)
        self.btn_backward.bind("<ButtonRelease-1>", self.on_release_backward)

        self.btn_left.bind("<ButtonPress-1>", self.on_press_left)
        self.btn_left.bind("<ButtonRelease-1>", self.on_release_left)

        self.btn_right.bind("<ButtonPress-1>", self.on_press_right)
        self.btn_right.bind("<ButtonRelease-1>", self.on_release_right)

        for r in [0, 1]:
            controls.rowconfigure(r, weight=1)
        for c in [0, 1, 2]:
            controls.columnconfigure(c, weight=1)

        self.btn_forward.grid(row=0, column=1, sticky="nsew")
        self.btn_left.grid(row=1, column=0, sticky="nsew")
        self.btn_right.grid(row=1, column=2, sticky="nsew")
        self.btn_backward.grid(row=1, column=1, sticky="nsew")

        # Z section
        tk.Label(
            panel,
            text="Z Step Size (µm):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        ).place(x=label_x, y=345)

        self.z_step_var = tk.StringVar(value=str(self.stage.z_step_size))
        self.z_step_entry = ttk.Entry(panel, textvariable=self.z_step_var, width=8)
        self.z_step_entry.place(x=entry_x, y=345)

        tk.Label(
            panel,
            text="Z Speed (µm/s):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        ).place(x=label_x, y=375)

        self.z_speed_var = tk.StringVar(value=str(self.stage.z_velocity))
        self.z_speed_var.trace_add("write", self.on_speed_change_z)

        self.z_speed_entry = ttk.Entry(panel, textvariable=self.z_speed_var, width=8)
        self.z_speed_entry.place(x=entry_x, y=375)

        tk.Label(
            panel,
            text="Z (µm):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        ).place(x=label_x, y=405)

        self.z_coord_var = tk.StringVar()
        self.z_coord_entry = ttk.Entry(panel, textvariable=self.z_coord_var, width=8)
        self.z_coord_entry.place(x=entry_x, y=405)

        self.z_reset_button = ttk.Button(
            panel,
            text="Set Z = 0",
            style="Normal.TButton",
            command=self.set_z_zero
        )
        self.z_reset_button.place(relx=0.5, y=440, anchor="n")
        self.app.register_button(self.z_reset_button)

        self.z_move_to_button = ttk.Button(
            panel,
            text="Move to Z",
            style="Normal.TButton",
            command=self.go_to_z_position
        )
        self.z_move_to_button.place(relx=0.5, y=475, anchor="n")
        self.app.register_button(self.z_move_to_button)

        z_button_panel = tk.Frame(panel, bg="white", width=80, height=45)
        z_button_panel.place(relx=0.5, y=510, anchor="n")
        z_button_panel.pack_propagate(False)

        z_controls = tk.Frame(z_button_panel, bg="white")
        z_controls.pack(expand=True, fill="both")

        self.btn_up = ttk.Button(z_controls, text="▴", style="Arrow.TButton")
        self.btn_down = ttk.Button(z_controls, text="▾", style="Arrow.TButton")

        self.btn_up.bind("<ButtonPress-1>", self.on_press_up)
        self.btn_up.bind("<ButtonRelease-1>", self.on_release_up)

        self.btn_down.bind("<ButtonPress-1>", self.on_press_down)
        self.btn_down.bind("<ButtonRelease-1>", self.on_release_down)

        self.app.register_button(self.btn_up)
        self.app.register_button(self.btn_down)

        z_controls.rowconfigure(0, weight=1)
        z_controls.columnconfigure(0, weight=1)
        z_controls.columnconfigure(1, weight=1)

        self.btn_up.grid(row=0, column=0, sticky="nsew")
        self.btn_down.grid(row=0, column=1, sticky="nsew")

        return panel
    
    # ---------------- Position display ----------------

    def update_position_display(self):
        x, y, z = self.stage.get_position()

        self.x_coord_var.set(str(x))
        self.y_coord_var.set(str(y))
        self.z_coord_var.set(str(z))

    # ---------------- Absolute movement ----------------

    def set_origin(self):
        self.stage.set_origin()
        self.update_position_display()

    def set_z_zero(self):
        self.stage.set_z_zero()
        self.update_position_display()

    def go_to_position(self, x=None, y=None):
        self.app.disable_buttons()

        try:
            if x is None:
                x = int(self.x_coord_var.get())

            if y is None:
                y = int(self.y_coord_var.get())

            self.stage.move_to_xy(x, y)
            self.update_position_display()

        finally:
            self.app.enable_buttons()

    def go_to_z_position(self, z=None):
        self.app.disable_buttons()

        try:
            if z is None:
                z = float(self.z_coord_var.get())

            self.stage.move_to_z(z)
            self.update_position_display()

        finally:
            self.app.enable_buttons()

    # ---------------- Hold movement helpers ----------------

    def start_hold_motion(self, start_func):
        self.is_hold = True
        start_func()

    def on_press_motion(self, start_func):
        self.is_hold = False
        self.hold_job = self.root.after(
            200,
            lambda: self.start_hold_motion(start_func)
        )

    def on_release_motion(self, stop_func, step_func):
        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)
            self.hold_job = None

        if self.is_hold:
            stop_func()
        else:
            self.update_step_sizes()
            step_func()

        self.update_position_display()

    def update_step_sizes(self):
        try:
            self.stage.set_xy_step_size(int(self.xy_step_var.get()))
        except ValueError:
            pass

        try:
            self.stage.set_z_step_size(int(self.z_step_var.get()))
        except ValueError:
            pass

    # ---------------- XY hold callbacks ----------------

    def on_press_forward(self, event):
        self.on_press_motion(self.stage.start_forward)

    def on_release_forward(self, event):
        self.on_release_motion(self.stage.stop_y, self.stage.step_forward)

    def on_press_backward(self, event):
        self.on_press_motion(self.stage.start_backward)

    def on_release_backward(self, event):
        self.on_release_motion(self.stage.stop_y, self.stage.step_backward)

    def on_press_left(self, event):
        self.on_press_motion(self.stage.start_left)

    def on_release_left(self, event):
        self.on_release_motion(self.stage.stop_x, self.stage.step_left)

    def on_press_right(self, event):
        self.on_press_motion(self.stage.start_right)

    def on_release_right(self, event):
        self.on_release_motion(self.stage.stop_x, self.stage.step_right)

    # ---------------- Z hold callbacks ----------------

    def on_press_up(self, event):
        self.on_press_motion(self.stage.start_up)

    def on_release_up(self, event):
        self.on_release_motion(self.stage.stop_z, self.stage.step_up)

    def on_press_down(self, event):
        self.on_press_motion(self.stage.start_down)

    def on_release_down(self, event):
        self.on_release_motion(self.stage.stop_z, self.stage.step_down)

    # ---------------- Speed callbacks ----------------

    def on_speed_change_xy(self, *args):
        try:
            self.stage.set_velocity(int(self.xy_speed_var.get()))
        except ValueError:
            pass

    def on_speed_change_z(self, *args):
        try:
            self.stage.set_z_velocity(int(self.z_speed_var.get()))
        except ValueError:
            pass
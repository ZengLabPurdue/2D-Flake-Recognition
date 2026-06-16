import tkinter as tk
from tkinter import Frame, Label, StringVar
from tkinter import ttk

class FocusPanel:
    def __init__(
        self,
        parent,
        app,
        focus_controller,
        default_range=1000,
        default_accuracy=10,
        default_steps=20,
    ):
        self.parent = parent
        self.app = app
        self.focus_controller = focus_controller

        self.default_range = default_range
        self.default_accuracy = default_accuracy
        self.default_steps = default_steps

        self.sharpness_var = tk.StringVar(value="Sharpness: Unknown")

        self.frame = self._build_panel()

    def _build_panel(self):
        panel = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=205
        )
        panel.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            panel,
            bg="white",
            width=200,
            height=203
        )
        background.place(x=2, y=0)

        title = Label(
            panel,
            text="Focus Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title.place(relx=0.5, y=5, anchor="n")

        sharpness_label = Label(
            panel,
            textvariable=self.sharpness_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        sharpness_label.place(relx=0.5, y=35, anchor="n")

        label_x = 10
        entry_x = 130

        range_label = Label(
            panel,
            text="Range:",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        range_label.place(relx=0.0, rely=0.0, x=label_x, y=65)

        self.range_var = StringVar(value=str(self.default_range))
        self.range_entry = ttk.Entry(
            panel,
            textvariable=self.range_var,
            width=8
        )
        self.range_entry.place(relx=0.0, rely=0.0, x=entry_x, y=65)

        accuracy_label = Label(
            panel,
            text="Accuracy:",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        accuracy_label.place(relx=0.0, rely=0.0, x=label_x, y=95)

        self.accuracy_var = StringVar(value=str(self.default_accuracy))
        self.accuracy_entry = ttk.Entry(
            panel,
            textvariable=self.accuracy_var,
            width=8
        )
        self.accuracy_entry.place(relx=0.0, rely=0.0, x=entry_x, y=95)

        steps_label = Label(
            panel,
            text="Num Steps:",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        steps_label.place(relx=0.0, rely=0.0, x=label_x, y=125)

        self.steps_var = StringVar(value=str(self.default_steps))
        self.steps_entry = ttk.Entry(
            panel,
            textvariable=self.steps_var,
            width=8
        )
        self.steps_entry.place(relx=0.0, rely=0.0, x=entry_x, y=125)

        self.auto_focus_btn = ttk.Button(
            panel,
            text="Auto Focus",
            style="Normal.TButton",
            command=self.run_auto_focus
        )
        self.auto_focus_btn.place(relx=0.5, y=160, anchor="n")

        self.app.buttons.append(self.auto_focus_btn)

        return panel

    def update_sharpness(self, sharpness):
        self.sharpness_var.set(f"Sharpness: {sharpness:.3f}")

    def run_auto_focus(self):
        try:
            start_range = int(self.range_var.get())
            accuracy = int(self.accuracy_var.get())
            steps = int(self.steps_var.get())

        except ValueError:
            self.sharpness_var.set("Invalid focus values")
            return

        self.focus_controller.auto_focus(
            start_range=start_range,
            accuracy=accuracy,
            steps=steps
        )
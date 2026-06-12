import tkinter as tk
from tkinter import Frame, Label
from tkinter import ttk

import config

class ExposurePanel:
    def __init__(
        self,
        parent,
        get_camera,
    ):
        self.parent = parent
        self.get_camera = get_camera

        self.frame = self._build_panel()

    def _build_panel(self):
        frame = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=100
        )
        frame.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            frame,
            bg="white",
            width=200,
            height=98
        )
        background.place(x=2, y=0)

        title = Label(
            frame,
            text="Adjust Exposure",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title.place(relx=0.5, y=5, anchor="n")

        style = ttk.Style()
        style.configure("Custom.Horizontal.TScale", background="white")

        self.exposure_var = tk.DoubleVar(value=config.DEFAULT_EXPOSURE)

        self.slider = ttk.Scale(
            background,
            from_=30,
            to=120,
            orient="horizontal",
            variable=self.exposure_var,
            command=self.adjust_exposure,
            style="Custom.Horizontal.TScale"
        )
        self.slider.place(relx=0.5, y=50, anchor="center")

        self.value_label = Label(
            background,
            text=f"Exposure: {config.DEFAULT_EXPOSURE}",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 8)
        )
        self.value_label.place(relx=0.5, y=70, anchor="n")

        return frame

    def adjust_exposure(self, exposure):
        camera = self.get_camera()

        if camera is None:
            return

        camera.put_AutoExpoTarget(int(float(exposure)))
        current_exposure = int(float(camera.get_AutoExpoTarget()))

        self.value_label.config(text=f"Exposure: {current_exposure}")
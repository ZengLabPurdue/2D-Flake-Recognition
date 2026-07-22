import tkinter as tk
from tkinter import Frame, Label
from tkinter import ttk

import config

class CameraSettingsPanel:
    def __init__(
        self,
        parent,
        get_camera,
        resolution_options=None,
        get_resolution=None,
        change_resolution_callback=None,
        chip_filter_var=None,
        vignette_filter_var=None,
        chip_filter_callback=None,
        vignette_filter_callback=None,
        operation_allowed=None,
    ):
        self.parent = parent
        self.get_camera = get_camera
        self.get_resolution = get_resolution
        self.change_resolution_callback = change_resolution_callback
        self.chip_filter_var = (
            chip_filter_var
            if chip_filter_var is not None
            else tk.BooleanVar(value=False)
        )
        self.vignette_filter_var = (
            vignette_filter_var
            if vignette_filter_var is not None
            else tk.BooleanVar(value=False)
        )
        self.chip_filter_callback = chip_filter_callback
        self.vignette_filter_callback = vignette_filter_callback
        self.operation_allowed = operation_allowed or (lambda: True)
        self._hardware_enabled = True

        self.resolution_value_to_label = dict(resolution_options)
        self.resolution_label_to_value = {
            label: value for value, label in self.resolution_value_to_label.items()
        }

        self.frame = self._build_panel()

    def _build_panel(self):
        frame = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=308,
        )
        frame.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            frame,
            bg="white",
            width=200,
            height=306,
        )
        background.place(x=2, y=0)

        title = Label(
            background,
            text="Camera Settings",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13),
        )
        title.place(relx=0.5, y=10, anchor="n")

        exposure_header = Label(
            background,
            text="Exposure",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 10, "bold"),
        )
        exposure_header.place(relx=0.5, y=40, anchor="n")

        self.auto_exposure_var = tk.BooleanVar(value=False)

        style = ttk.Style()
        style.configure(
            "CameraSettings.TCheckbutton",
            background="white",
            foreground="black",
            font=("TkDefaultFont", 8),
        )

        self.auto_exposure_checkbox = ttk.Checkbutton(
            background,
            text="Auto Exposure",
            variable=self.auto_exposure_var,
            command=self.toggle_auto_exposure,
            style="CameraSettings.TCheckbutton",
        )
        self.auto_exposure_checkbox.place(relx=0.5, y=75, anchor="center")

        style.configure("Custom.Horizontal.TScale", background="white")

        self.exposure_var = tk.DoubleVar(value=config.DEFAULT_EXPOSURE)

        self.slider = ttk.Scale(
            background,
            from_=30,
            to=120,
            orient="horizontal",
            variable=self.exposure_var,
            command=self.adjust_exposure,
            style="Custom.Horizontal.TScale",
            length=150,
        )
        self.slider.place(relx=0.5, y=110, anchor="center")

        self.value_label = Label(
            background,
            text=f"Target: {config.DEFAULT_EXPOSURE}",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 8),
        )
        self.value_label.place(relx=0.5, y=125, anchor="n")

        self.set_slider_enabled(False)

        resolution_header = Label(
            background,
            text="Resolution",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 10, "bold"),
        )
        resolution_header.place(relx=0.5, y=150, anchor="n")

        if self.get_resolution is not None:
            current_resolution = self.get_resolution()
        else:
            current_resolution = list(self.resolution_value_to_label.keys())[0]

        current_resolution_label = self.resolution_value_to_label.get(
            current_resolution,
            str(current_resolution),
        )

        self.resolution_var = tk.StringVar(value=current_resolution_label)

        self.resolution_dropdown = ttk.Combobox(
            background,
            textvariable=self.resolution_var,
            values=list(self.resolution_label_to_value.keys()),
            state="readonly",
            width=18,
            font=("TkDefaultFont", 8),
        )
        self.resolution_dropdown.place(relx=0.5, y=190, anchor="center", width=184)
        self.resolution_dropdown.bind(
            "<<ComboboxSelected>>",
            self.change_resolution,
        )

        filter_header = Label(
            background,
            text="Image Filters",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 10, "bold"),
        )
        filter_header.place(relx=0.5, y=220, anchor="n")

        self.chip_filter_checkbox = ttk.Checkbutton(
            background,
            text="Chip Filter",
            variable=self.chip_filter_var,
            command=self.chip_filter_callback,
            style="CameraSettings.TCheckbutton",
        )
        self.chip_filter_checkbox.place(relx=0.5, y=255, anchor="center")

        self.vignette_filter_checkbox = ttk.Checkbutton(
            background,
            text="Vignette Filter",
            variable=self.vignette_filter_var,
            command=self.vignette_filter_callback,
            style="CameraSettings.TCheckbutton",
        )
        self.vignette_filter_checkbox.place(relx=0.5, y=282, anchor="center")

        return frame

    def set_slider_enabled(self, enabled: bool):
        if enabled and self._hardware_enabled:
            self.slider.state(["!disabled"])
        else:
            self.slider.state(["disabled"])

    def set_hardware_enabled(self, enabled: bool):
        """Lock camera settings while the scan worker owns the camera."""
        self._hardware_enabled = bool(enabled)
        if self._hardware_enabled:
            self.auto_exposure_checkbox.state(["!disabled"])
            self.resolution_dropdown.state(["!disabled", "readonly"])
        else:
            self.auto_exposure_checkbox.state(["disabled"])
            self.resolution_dropdown.state(["disabled"])
        self.set_slider_enabled(
            self._hardware_enabled and self.auto_exposure_var.get()
        )

    def toggle_auto_exposure(self):
        if not self.operation_allowed():
            return
        camera = self.get_camera()
        active = self.auto_exposure_var.get()

        self.set_slider_enabled(active)

        if camera is None:
            return

        camera.put_AutoExpoEnable(1 if active else 0)

        if active:
            self.adjust_exposure(self.exposure_var.get())

        if not self.auto_exposure_var.get():
            camera.put_ExpoTime(1500)
            camera.put_ExpoAGain(100)

    def adjust_exposure(self, exposure):
        if not self.operation_allowed() or not self.auto_exposure_var.get():
            return

        camera = self.get_camera()

        if camera is None:
            return

        exposure = int(float(exposure))

        camera.put_AutoExpoTarget(exposure)
        current_exposure = int(float(camera.get_AutoExpoTarget()))

        self.value_label.config(text=f"Target: {current_exposure}")

    def change_resolution(self, event=None):
        if not self.operation_allowed():
            return
        print("Resolution changed")
        selected_label = self.resolution_var.get()

        selected_resolution = self.resolution_label_to_value[selected_label]

        if self.change_resolution_callback is None:
            print(f"Selected resolution: {selected_resolution}")
            return

        self.change_resolution_callback(selected_resolution)

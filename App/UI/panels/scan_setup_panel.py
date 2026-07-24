from pathlib import Path

import tkinter as tk
from tkinter import Frame, Label, filedialog, messagebox
from tkinter import ttk

from Scanning.scan_profile import PROFILE_FILENAME, ScanProfile, ScanProfileError


class ScanSetupPanel:
    SCAN_TYPES = (
        "Complete Scan (1 Chip)",
        "Full Stage Scan",
        "2x Scan",
        "10x Scan",
        "20x Scan",
        "100x Scan",
        "Vignette Filter",
    )
    FULL_SCAN_TYPES = {"Complete Scan (1 Chip)", "Full Stage Scan"}
    DETECTION_SCAN_TYPES = FULL_SCAN_TYPES | {"100x Scan"}
    MATERIALS = ("Gr", "hBN", "MoS2", "MoTe2", "WS2", "WSe2")
    SUBSTRATE_THICKNESSES = ("100nm", "285nm")
    FULL_SCAN_MAGNIFICATIONS = ("10x", "20x")
    DETECTION_MODELS = ("Region Detection", "Flake Detection")

    DEFAULT_MATERIAL = "Gr"
    DEFAULT_SUBSTRATE = "285nm"

    def __init__(
        self,
        parent,
        root,
        app,
        scan_manager,
        ui_dispatch=None,
    ):
        self.parent = parent
        self.root = root
        self.app = app
        self.scan_manager = scan_manager
        self.ui_dispatch = ui_dispatch or (
            lambda callback, *args, **kwargs: callback(*args, **kwargs)
        )
        self._default_profile_checked = False

        self.scan_type_var = tk.StringVar(value=self.SCAN_TYPES[0])
        self.window_width_var = tk.StringVar()
        self.window_height_var = tk.StringVar()
        self.material_var = tk.StringVar(value=self.DEFAULT_MATERIAL)
        self.substrate_var = tk.StringVar(value=self.DEFAULT_SUBSTRATE)
        self.full_scan_magnification_var = tk.StringVar(
            value=self.FULL_SCAN_MAGNIFICATIONS[0]
        )
        self.detection_model_var = tk.StringVar(value="Region Detection")
        self.profile_name_var = tk.StringVar(value="Profile: loading default...")
        self.class_legend_var = tk.BooleanVar(value=True)

        self.frame = self._build_panel()
        self._show_controls_for_scan_type()

    def _build_panel(self):
        panel = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=480,
        )
        panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.background = Frame(panel, bg="white", width=200, height=478)
        self.background.place(x=2, y=0)

        style = ttk.Style()
        style.configure(
            "Normal.TButton",
            font="TkDefaultFont",
            background="white",
            relief="flat",
        )
        style.configure(
            "ScanSetup.TCheckbutton",
            background="white",
            foreground="black",
        )

        Label(
            self.background,
            text="Scan Setup",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13),
        ).place(relx=0.5, y=10, anchor="n")

        Label(
            self.background,
            text="Scan type",
            bg="white",
            fg="black",
        ).place(x=8, y=43)
        self.scan_type_dropdown = ttk.Combobox(
            self.background,
            textvariable=self.scan_type_var,
            values=self.SCAN_TYPES,
            state="readonly",
            width=24,
            font=("TkDefaultFont", 8),
        )
        self.scan_type_dropdown.place(relx=0.5, y=66, anchor="n", width=184)
        self.scan_type_dropdown.bind(
            "<<ComboboxSelected>>",
            self._show_controls_for_scan_type,
        )

        self.window_controls = Frame(
            self.background,
            bg="white",
            width=184,
            height=100,
        )
        Label(
            self.window_controls,
            text="Window size (tiles)",
            bg="white",
            fg="black",
        ).place(x=0, y=0)

        Label(
            self.window_controls,
            text="Width",
            bg="white",
            fg="black",
        ).place(x=15, y=31)
        self.window_width_entry = ttk.Spinbox(
            self.window_controls,
            from_=1,
            to=999,
            textvariable=self.window_width_var,
            width=10,
        )
        self.window_width_entry.place(x=85, y=28, width=84)

        Label(
            self.window_controls,
            text="Height",
            bg="white",
            fg="black",
        ).place(x=15, y=65)
        self.window_height_entry = ttk.Spinbox(
            self.window_controls,
            from_=1,
            to=999,
            textvariable=self.window_height_var,
            width=10,
        )
        self.window_height_entry.place(x=85, y=62, width=84)

        self.full_scan_controls = Frame(
            self.background,
            bg="white",
            width=184,
            height=340,
        )
        self._build_full_scan_controls()

        self.run_button = ttk.Button(
            self.background,
            text="Run Scan",
            style="Normal.TButton",
            command=self._run_scan,
        )
        return panel

    def _build_full_scan_controls(self):
        Label(
            self.full_scan_controls,
            text="Material",
            bg="white",
            fg="black",
        ).place(x=0, y=0)
        self.material_dropdown = ttk.Combobox(
            self.full_scan_controls,
            textvariable=self.material_var,
            values=self.MATERIALS,
            state="readonly",
            width=21,
            font=("TkDefaultFont", 8),
        )
        self.material_dropdown.place(relx=0.5, y=22, anchor="n", width=184)

        Label(
            self.full_scan_controls,
            text="Substrate thickness",
            bg="white",
            fg="black",
        ).place(x=0, y=54)
        self.substrate_dropdown = ttk.Combobox(
            self.full_scan_controls,
            textvariable=self.substrate_var,
            values=self.SUBSTRATE_THICKNESSES,
            state="readonly",
            width=21,
            font=("TkDefaultFont", 8),
        )
        self.substrate_dropdown.place(relx=0.5, y=76, anchor="n", width=184)

        Label(
            self.full_scan_controls,
            text="Full scan magnification",
            bg="white",
            fg="black",
        ).place(x=0, y=108)
        self.magnification_dropdown = ttk.Combobox(
            self.full_scan_controls,
            textvariable=self.full_scan_magnification_var,
            values=self.FULL_SCAN_MAGNIFICATIONS,
            state="readonly",
            width=21,
            font=("TkDefaultFont", 8),
        )
        self.magnification_dropdown.place(relx=0.5, y=130, anchor="n", width=184)

        Label(
            self.full_scan_controls,
            text="Detection model",
            bg="white",
            fg="black",
        ).place(x=0, y=162)
        self.model_dropdown = ttk.Combobox(
            self.full_scan_controls,
            textvariable=self.detection_model_var,
            values=self.DETECTION_MODELS,
            state="readonly",
            width=21,
            font=("TkDefaultFont", 8),
        )
        self.model_dropdown.place(relx=0.5, y=184, anchor="n", width=184)
        self.model_dropdown.bind(
            "<<ComboboxSelected>>",
            self._on_detection_model_changed,
        )

        self.profile_label = Label(
            self.full_scan_controls,
            textvariable=self.profile_name_var,
            bg="white",
            fg="#333333",
            justify="center",
            wraplength=184,
        )
        self.load_profile_button = ttk.Button(
            self.full_scan_controls,
            text="Load Profile",
            style="Normal.TButton",
            command=self._choose_and_load_profile,
        )
        self.class_legend_checkbox = ttk.Checkbutton(
            self.full_scan_controls,
            text="Show legend in flake results",
            variable=self.class_legend_var,
            style="ScanSetup.TCheckbutton",
        )

    def show(self):
        self.app.set_view("Camera View")

        if self.scan_manager.is_scan_running():
            self.app.close_all_panels()
            self.app.open_panel("Scan Info Panel")
            return

        self._sync_active_profile()
        self._ensure_default_profile()
        self.app.close_all_panels()
        self.app.open_panel("Scan Setup Panel")

    def _show_controls_for_scan_type(self, event=None):
        scan_type = self.scan_type_var.get()
        self.window_controls.place_forget()
        self.full_scan_controls.place_forget()

        if scan_type in self.DETECTION_SCAN_TYPES:
            self.full_scan_controls.place(relx=0.5, y=102, anchor="n")
        else:
            default_window = self.scan_manager.DEFAULT_WINDOWS[scan_type]
            self.window_width_var.set(str(default_window[0]))
            self.window_height_var.set(str(default_window[1]))
            self.window_controls.place(relx=0.5, y=105, anchor="n")
            self.frame.configure(height=257)
            self.background.configure(height=255)
            self.run_button.place(relx=0.5, y=214, anchor="n")

        self._on_detection_model_changed()
        self.app.update_panels()

    def _on_detection_model_changed(self, event=None):
        region_selected = self.detection_model_var.get() == "Region Detection"
        full_scan_selected = self.scan_type_var.get() in self.DETECTION_SCAN_TYPES
        self.profile_label.place_forget()
        self.load_profile_button.place_forget()
        self.class_legend_checkbox.place_forget()

        if region_selected and full_scan_selected:
            self.profile_label.place(relx=0.5, y=221, anchor="n", width=184)
            self.load_profile_button.place(relx=0.5, y=255, anchor="n")
            self.class_legend_checkbox.place(relx=0.5, y=295, anchor="n")
            self.run_button.state(["!disabled"])
            self.full_scan_controls.configure(height=330)
            self.frame.configure(height=480)
            self.background.configure(height=478)
            self.run_button.place(relx=0.5, y=435, anchor="n")
        else:
            self.run_button.state(["!disabled"])
            if full_scan_selected:
                self.full_scan_controls.configure(height=230)
                self.frame.configure(height=367)
                self.background.configure(height=365)
                self.run_button.place(relx=0.5, y=324, anchor="n")

        if event is not None:
            self.app.update_panels()

    def _run_scan(self):
        scan_type = self.scan_type_var.get()
        options = {"scan_type": scan_type}

        if scan_type in self.DETECTION_SCAN_TYPES:
            if (
                scan_type in self.FULL_SCAN_TYPES
                and
                self.material_var.get() == self.DEFAULT_MATERIAL
                and self.substrate_var.get() == self.DEFAULT_SUBSTRATE
            ):
                confirmed = messagebox.askyesno(
                    "Confirm Scan",
                    "The scan parameters are still set to their defaults.\n\n"
                    f"Material: {self.material_var.get()}\n"
                    f"Substrate thickness: {self.substrate_var.get()}\n\n"
                    "Are these parameters correct?",
                    parent=self.root,
                )
                if not confirmed:
                    return

            profile = self.app.get_active_scan_profile()
            if (
                self.detection_model_var.get() == "Region Detection"
                and (profile is None or getattr(profile, "path", None) is None)
            ):
                messagebox.showwarning(
                    "Scan Profile Required",
                    "Load a scan profile before running Region Detection.",
                    parent=self.root,
                )
                return
            options.update({
                "detection_model": self.detection_model_var.get(),
                "scan_profile": profile,
                "display_class_legend": self.class_legend_var.get(),
            })
            if scan_type in self.FULL_SCAN_TYPES:
                options.update({
                    "material": self.material_var.get(),
                    "substrate_thickness": self.substrate_var.get(),
                    "full_scan_magnification": self.full_scan_magnification_var.get(),
                })
        else:
            try:
                options["window"] = self._get_window()
            except ValueError as exc:
                messagebox.showwarning(
                    "Invalid Window Size",
                    str(exc),
                    parent=self.root,
                )
                return

        self.app.close_all_panels()
        self.app.open_panel("Scan Info Panel")
        self.root.update_idletasks()
        self.root.after_idle(lambda: self._execute_scan(options))

    def _execute_scan(self, options):
        try:
            self.scan_manager.start_scan(
                on_error=self._show_scan_error,
                **options,
            )
        except Exception as exc:
            self._show_scan_error(exc)

    def _show_scan_error(self, error):
        self.ui_dispatch(
            messagebox.showerror,
            "Scan Error",
            str(error),
            parent=self.root,
        )

    def _get_window(self):
        try:
            width = int(self.window_width_var.get())
            height = int(self.window_height_var.get())
        except ValueError as exc:
            raise ValueError("Window width and height must be whole numbers.") from exc
        if width < 1 or height < 1:
            raise ValueError("Window width and height must both be at least 1.")
        return width, height

    def _sync_active_profile(self):
        profile = self.app.get_active_scan_profile()
        if profile is not None and getattr(profile, "name", ""):
            self.profile_name_var.set(f"Profile: {profile.name}")

    def _ensure_default_profile(self):
        if self.app.get_active_scan_profile() is not None:
            self._default_profile_checked = True
            return
        if self._default_profile_checked:
            return

        self._default_profile_checked = True
        profile_paths = sorted(
            ScanProfile().profiles_dir.glob(f"*/{PROFILE_FILENAME}"),
            key=lambda path: str(path).casefold(),
        )
        if not profile_paths:
            self.profile_name_var.set("Profile: none loaded")
            return

        for profile_path in profile_paths:
            try:
                profile = ScanProfile().load_profile(profile_path)
            except (OSError, ScanProfileError):
                continue
            self.app.set_active_scan_profile(profile)
            self.profile_name_var.set(f"Profile: {profile.name} (default)")
            return
        self.profile_name_var.set("Profile: default could not be loaded")

    def _choose_and_load_profile(self):
        profiles_dir = ScanProfile().profiles_dir
        profiles_dir.mkdir(parents=True, exist_ok=True)
        load_folder = messagebox.askyesnocancel(
            "Load Scan Search Profile",
            "Load a profile folder?\n\n"
            "Yes: select the folder containing profile.json\n"
            "No: select profile.json directly",
            parent=self.root,
        )
        if load_folder is None:
            return

        if load_folder:
            selected = filedialog.askdirectory(
                title="Select Scan Search Profile Folder",
                initialdir=profiles_dir,
                parent=self.root,
            )
        else:
            selected = filedialog.askopenfilename(
                title="Select Scan Search Profile JSON",
                initialdir=profiles_dir,
                filetypes=[("Scan search profile", "profile.json"), ("JSON", "*.json")],
                parent=self.root,
            )
        if not selected:
            return

        try:
            profile = ScanProfile().load_profile(Path(selected))
        except (OSError, ScanProfileError) as exc:
            messagebox.showerror(
                "Invalid Scan Search Profile",
                str(exc),
                parent=self.root,
            )
            return

        self.app.set_active_scan_profile(profile)
        self.profile_name_var.set(f"Profile: {profile.name}")
        messagebox.showinfo(
            "Profile Loaded",
            f"Loaded '{profile.name}' with {len(profile.classes)} class(es).",
            parent=self.root,
        )

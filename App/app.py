import os
import sys
import threading

import cv2
import numpy as np

from tkinter import *
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

from config import CROP_RATIO, RESOLUTION_DISPLAY

from Mapping.mapper import Mapper
from Scanning.scan_manager import ScanManager
from Scanning.scan_profile import ScanProfile
from Hardware.stage_api import stage
from Hardware.turret_controller import TurretController
from UI.panels.info_panel import InfoPanel
from UI.panels.stage_control_panel import StageControlPanel
from UI.panels.objective_control_panel import ObjectiveControlPanel
from UI.panels.focus_panel import FocusPanel
from UI.panels.scan_status_panel import ScanStatusPanel
from UI.panels.scan_setup_panel import ScanSetupPanel
from UI.panels.view_scans_panel import ViewScansPanel
from UI.panels.scan_profile_panel import ScanProfilePanel
from UI.panels.capture_panel import CapturePanel
from UI.panels.camera_settings_panel import CameraSettingsPanel
from Imaging.frame_processing import FrameProcessor
from Imaging.focus import FocusController

DLL_PATH = os.getcwd() + r"\Hardware\PriorSDK1.9.2\x64\PriorScientificSDK.dll"
PRIOR_COM_PORT = sys.argv[1]
TURRET_COM_PORT = sys.argv[2]

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Scanning App")
        self.main_frame = Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=BOTH, expand=True)

        self.map_canvas = Canvas(self.main_frame)
        self.map_canvas.pack(fill=BOTH, expand=True)

        self.true_map = np.zeros((6000, 6000, 3), dtype=np.uint8)
        self.filter_map = np.zeros((6000, 6000), dtype=np.uint8)
        self.region_map = np.zeros((6000, 6000, 3), dtype=np.uint8)
        self.region_map_id = None
        self._region_map_lock = threading.Lock()

        self.img_label = Label(self.main_frame, bg="#f0f0f0")
        self.img_label.pack(fill=BOTH, expand=True)

        self.live_mapping_var = tk.BooleanVar(value=False)
        self.map_chip_filter_var = tk.BooleanVar(value=False)
        self.region_map_view_var = tk.BooleanVar(value=False)
        self.region_map_available = False
        self.camera_chip_filter_var = tk.BooleanVar(value=False)
        self.camera_vignette_filter_var = tk.BooleanVar(value=False)
        self.region_map_toggle = tk.Checkbutton(
            self.main_frame,
            text="Segmented Map",
            variable=self.region_map_view_var,
            command=self.toggle_region_map_view,
            bg="white",
            activebackground="white",
            selectcolor="white",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=4,
        )
        self.view_mode = None
        self.set_view("Camera View")

        self.hcam = None
        self.width = 0
        self.height = 0
        self._displayed_image_bounds = None
        
        self.magnification = "2X"
        self.resolution = "HIGH"

        self.buttons = []
        self.panels = []
        self.scan_profile = ScanProfile()
        self.active_scan_profile = None

        self.stage = stage(PRIOR_COM_PORT, DLL_PATH)
        
        self.frame_processor = FrameProcessor(
            root=self.root,
            app=self,
            stage=self.stage,
            get_live_mapping=self.get_live_mapping,
        )

        self.focus_controller = FocusController(
            app=self,
            stage=self.stage,
            frame_processor=self.frame_processor,
        )

        self.turret_controller = TurretController( 
            app=self,
            stage=self.stage,
            turret_port=TURRET_COM_PORT,
            start_auto_focus_thread=self.focus_controller.start_auto_focus_thread,
        )

        self.info_panel = InfoPanel(parent=self.main_frame)

        self.scan_status_panel = ScanStatusPanel(
            parent=self.main_frame,
            root=self.root,
        )

        self.mapper = Mapper(
            root=self.root,
            app=self,
            stage=self.stage,
            turret_controller=self.turret_controller,
            frame_processor=self.frame_processor,
            update_scan_status=self.scan_status_panel.update_status,
        )
        self.frame_processor.place_frame_on_map = self.mapper.place_frame_on_map

        self.frame_processor.run_camera()

        self.scan_manager = ScanManager(
            root=self.root,
            app=self,
            stage=self.stage,
            turret_controller=self.turret_controller,
            camera=self.frame_processor.get_camera(),
            frame_processor=self.frame_processor,
            mapper=self.mapper,
            update_scan_status=self.scan_status_panel.update_status,
            set_scan_running=self.scan_status_panel.set_scan_running,
        )
        self.scan_status_panel.set_stop_callback(self.scan_manager.stop_scan)

        self.stage_control_panel = StageControlPanel(
            parent=self.main_frame,
            root=self.root,
            app=self,
            stage=self.stage,
        )

        self.objective_control_panel = ObjectiveControlPanel(
            parent=self.main_frame,
            app=self,
            stage=self.stage,
            turret_controller=self.turret_controller,
        )

        self.capture_panel = CapturePanel(
            parent=self.main_frame,
            app=self,
            save_image=self.frame_processor.save_image,
        )

        self.camera_settings_panel = CameraSettingsPanel(
            parent=self.root,
            get_camera=self.frame_processor.get_camera,
            resolution_options=RESOLUTION_DISPLAY,
            get_resolution=self.get_resolution,
            change_resolution_callback=self.frame_processor.change_resolution,
            chip_filter_var=self.camera_chip_filter_var,
            vignette_filter_var=self.camera_vignette_filter_var,
            chip_filter_callback=lambda: self.set_view("Camera View"),
            vignette_filter_callback=self.toggle_camera_vignette_filter,
        )

        self.focus_panel = FocusPanel(
            parent=self.main_frame,
            app=self,
            focus_controller=self.focus_controller,
        )
        
        self.view_scans_panel = ViewScansPanel(
            parent=self.main_frame,
            root=self.root,
            app=self,
        )

        self.scan_profile_panel = ScanProfilePanel(
            parent=self.main_frame,
            root=self.root,
            app=self,
            scan_profile=self.scan_profile,
        )

        self.scan_setup_panel = ScanSetupPanel(
            parent=self.main_frame,
            root=self.root,
            app=self,
            scan_manager=self.scan_manager,
        )

        self.focus_controller.sharpness_callback = self.focus_panel.update_sharpness

        self.panels.append({
            "name": "Info Panel",
            "frame": self.info_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Scan Info Panel",
            "frame": self.scan_status_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Stage Control Panel",
            "frame": self.stage_control_panel.frame,
            "var": BooleanVar(value=False)
        })
        
        self.panels.append({
            "name": "Capture Panel",
            "frame": self.capture_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Camera Settings Panel",
            "frame": self.camera_settings_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Objective Control Panel",
            "frame": self.objective_control_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Focus Panel",
            "frame": self.focus_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Scan Setup Panel",
            "frame": self.scan_setup_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.update_panels()

        self.init_menu_bar()

        self.root.bind_all("<Button-1>", self.clear_focus, add="+")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)        

    # ------------- Initialization -------------

    def init_menu_bar(self):
        menu_bar = Menu(self.root)
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Quit", command=self.on_close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        view_menu = Menu(menu_bar, tearoff=0)

        view_menu.add_command(
            label="Camera View",
            command=lambda: self.set_view("Camera View"),
        )
        view_menu.add_command(
            label="Map View",
            command=lambda: self.set_view("Map"),
        )
        view_menu.add_separator()

        profile_menu = Menu(view_menu, tearoff=0)
        profile_menu.add_command(
            label="Create Profile",
            command=self.scan_profile_panel.start_create,
        )
        profile_menu.add_command(
            label="Load Profile",
            command=self.scan_profile_panel.choose_and_load_profile,
        )
        view_menu.add_cascade(label="Scan Profiles", menu=profile_menu)
        self.view_scans_panel.add_to_menu(view_menu)
        view_menu.add_separator()

        map_menu = Menu(view_menu, tearoff=0)
        map_menu.add_checkbutton(
            label="Chip Filter",
            variable=self.map_chip_filter_var,
            command=self.toggle_map_chip_filter,
        )

        camera_menu = Menu(view_menu, tearoff=0)
        camera_menu.add_checkbutton(
            label="Chip Filter",
            variable=self.camera_chip_filter_var,
            command=lambda: self.set_view("Camera View"),
        )
        camera_menu.add_checkbutton(
            label="Vignette Filter",
            variable=self.camera_vignette_filter_var,
            command=self.toggle_camera_vignette_filter,
        )

        view_menu.add_cascade(label="Map Filters", menu=map_menu)
        view_menu.add_cascade(label="Camera Filters", menu=camera_menu)

        menu_bar.add_cascade(label="View", menu=view_menu)

        panel_menu = Menu(menu_bar, tearoff=0)

        for panel in self.panels:
            panel_menu.add_checkbutton(
                label=panel["name"],
                variable=panel["var"],
                command=self.update_panels
            )

        menu_bar.add_cascade(label="Panels", menu=panel_menu)

        map_menu = Menu(menu_bar, tearoff=0)
        map_menu.add_radiobutton(label="Live Map 2x", variable=self.live_mapping_var, value=True, command=lambda: self.mapper.set_live_map_2x())
        map_menu.add_command(label="Auto Map 2x", command=self.mapper.auto_map_2x)
        menu_bar.add_cascade(label="Map", menu=map_menu)

        menu_bar.add_command(label="Scan", command=self.scan_setup_panel.show)

        self.root.config(menu=menu_bar)

    def display_chip_dropdown(self, display=True):

        shift = 0 if display else -30

        if display:
            self.chip_dropdown.place(relx=0.5, y=self.pos_chip, anchor="n")
        else:
            self.chip_dropdown.place_forget()

        self.image_label.place(relx=0.5, y=self.pos_image + shift, anchor="n")
        self.views_scans_button_panel.place(relx=0.5, y=self.pos_buttons + shift, anchor="n")

        base_height = 205
        new_height = base_height + shift

        self.view_scans_panel.config(height=new_height)
        self.view_scans_background.config(height=new_height - 2)

    # ------------- Display Functions -------------

    def display_map(self):
        if self.region_map_available and self.region_map_view_var.get():
            with self._region_map_lock:
                map_data = self.region_map.copy()
            self.map_image = Image.fromarray(map_data)
        elif self.map_chip_filter_var.get():
            self.map_image = Image.fromarray(self.filter_map.astype(np.uint8))
        else:
            self.map_image = Image.fromarray(self.true_map.astype(np.uint8))

        canvas_width = self.map_canvas.winfo_width()
        canvas_height = self.map_canvas.winfo_height()

        if canvas_width == 1 or canvas_height == 1:
            self.map_canvas.bind("<Configure>", lambda e: self.display_map())
            return

        img_ratio = self.map_image.width / self.map_image.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)

        img_resized = self.map_image.resize((new_width, new_height), Image.NEAREST)
        self.tk_map_image = ImageTk.PhotoImage(img_resized)

        self.map_canvas.delete("all")

        x_center = canvas_width // 2
        y_center = canvas_height // 2
        self.map_canvas.create_image(x_center, y_center, image=self.tk_map_image, anchor="center")

    def display_image(self, img_rgb):
        h, w = img_rgb.shape[:2]

        cx = w // 2
        cy = h // 2

        if self.view_mode == "Camera View":
            if self.magnification == "2X":
                crop_w = int(w * CROP_RATIO["2X"]["x"])
                crop_h = int(h * CROP_RATIO["2X"]["y"])
            elif self.magnification == "10X":
                crop_w = int(w * CROP_RATIO["10X"]["x"])
                crop_h = int(h * CROP_RATIO["10X"]["y"])
            elif self.magnification == "20X":
                crop_w = int(w * CROP_RATIO["20X"]["x"])
                crop_h = int(h * CROP_RATIO["20X"]["y"])
            elif self.magnification == "100X":
                crop_w = int(w * CROP_RATIO["100X"]["x"])
                crop_h = int(h * CROP_RATIO["100X"]["y"])
            else:
                crop_w = w
                crop_h = h

            x1 = cx - crop_w // 2
            y1 = cy - crop_h // 2
            x2 = cx + crop_w // 2
            y2 = cy + crop_h // 2

            img_rgb = img_rgb.copy()
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 5)

            cv2.circle(img_rgb, (cx, cy), radius=5, color=(255, 0, 0), thickness=-1)

        img_pil = Image.fromarray(img_rgb)

        lbl_w = self.img_label.winfo_width() or self.width
        lbl_h = self.img_label.winfo_height() or self.height

        if lbl_w < 10 or lbl_h < 10:
            return

        img_pil_copy = img_pil.copy()
        img_pil_copy.thumbnail((lbl_w, lbl_h), Image.Resampling.LANCZOS)

        display_img = Image.new("RGB", (lbl_w, lbl_h), "#f0f0f0")
        x_offset = (lbl_w - img_pil_copy.width) // 2
        y_offset = (lbl_h - img_pil_copy.height) // 2
        display_img.paste(img_pil_copy, (x_offset, y_offset))

        self._displayed_image_bounds = (
            x_offset,
            y_offset,
            img_pil_copy.width,
            img_pil_copy.height,
            w,
            h,
        )

        img_tk = ImageTk.PhotoImage(display_img)
        self.img_label.configure(image=img_tk, text="")
        self.img_label.image = img_tk

    def display_image_message(self, message):
        """Clear the current image and show a centered instruction instead."""
        self._displayed_image_bounds = None
        self.img_label.configure(
            image="",
            text=message,
            fg="#555555",
            font=("TkDefaultFont", 14),
            anchor="center",
        )
        self.img_label.image = None

    # ------------- Util Functions -------------

    def update_panels(self):
        y_position = -2

        for panel in self.panels:
            frame = panel["frame"]
            frame.place_forget()

            if panel["var"].get():
                frame.place(
                    relx=1.0,
                    rely=0.0,
                    anchor="ne",
                    y=y_position
                )

                frame.update_idletasks()

                y_position += frame.winfo_height()

    def close_all_panels(self):
        for panel in self.panels:
            panel["var"].set(False)
        self.update_panels()
        self.view_scans_panel.hide()
        self.scan_profile_panel.hide()

    def open_panel(self, name):
        for panel in self.panels:
            if panel["name"] == name:
                panel["var"].set(True)
                self.update_panels()
                return

    def enable_buttons(self):
        for btn in self.buttons:
            btn.state(["!disabled"])
        self.root.update_idletasks()

    def disable_buttons(self):
        for btn in self.buttons:
            btn.state(["disabled"])
        self.root.update_idletasks()

    def register_button(self, button):
        self.buttons.append(button)

    def clear_focus(self, event):
        widget = event.widget
        widget_path = str(widget).lower()

        if isinstance(widget, (ttk.Combobox, ttk.Entry, tk.Entry)):
            return
        
        if "popdown" in widget_path or "combobox" in widget_path:
            return

        self.root.focus_set()

    def on_close(self):
        self.frame_processor.close()
        self.stage.disconnect()
        self.root.destroy()

    def get_view(self):
        return self.view_mode

    def display_to_image_point(self, display_x, display_y):
        """Map a click in ``img_label`` back to the current source image."""
        if self._displayed_image_bounds is None:
            return None

        x_offset, y_offset, display_width, display_height, image_width, image_height = (
            self._displayed_image_bounds
        )
        relative_x = display_x - x_offset
        relative_y = display_y - y_offset
        if not (0 <= relative_x < display_width and 0 <= relative_y < display_height):
            return None

        image_x = min(image_width - 1, int(relative_x * image_width / display_width))
        image_y = min(image_height - 1, int(relative_y * image_height / display_height))
        return image_x, image_y

    def set_view(self, mode, filter_status=None):
        self.view_mode = mode

        if hasattr(self, "scan_profile_panel") and mode not in (
            "Create Search Profile",
            "Load Search Profile",
        ):
            self.scan_profile_panel.hide()
        if hasattr(self, "view_scans_panel") and mode != "Scan Results":
            self.view_scans_panel.hide()

        if filter_status is not None:
            if mode == "Map":
                self.map_chip_filter_var.set(filter_status)
            elif mode == "Camera View":
                self.camera_chip_filter_var.set(filter_status)

        if mode == "Map":
            self.display_map()
            self.map_canvas.pack(fill=BOTH, expand=True) # Show map canvas
            self.img_label.pack_forget() # Hide image label
            pass
        elif mode == "Camera View":
            self.img_label.pack(fill=BOTH, expand=True)
            self.map_canvas.pack_forget() # Hide map canvas
        elif mode == "Create Search Profile" or mode == "Load Search Profile":
            self.img_label.pack(fill=BOTH, expand=True)
            self.map_canvas.pack_forget()
        elif mode == "Scan Results":
            self.img_label.pack(fill=BOTH, expand=True)
            self.map_canvas.pack_forget() # Hide map canvas

        self._update_region_map_toggle_visibility()
        self.root.update_idletasks()

    def get_live_mapping(self):
        return self.live_mapping_var.get()
    
    def set_live_mapping(self, live_mapping_status):
        self.live_mapping_var.set(live_mapping_status)

    def get_filter(self):
        if self.view_mode == "Map":
            return self.get_map_chip_filter()
        if self.view_mode == "Camera View":
            return self.get_camera_chip_filter()
        return False
    
    def set_filter(self, status : bool):
        if self.view_mode == "Map":
            self.map_chip_filter_var.set(status)
        elif self.view_mode == "Camera View":
            self.camera_chip_filter_var.set(status)

    def get_map_chip_filter(self):
        return self.map_chip_filter_var.get()

    def toggle_map_chip_filter(self):
        if self.map_chip_filter_var.get():
            self.region_map_view_var.set(False)
        self.set_view("Map")

    def toggle_region_map_view(self):
        if self.region_map_view_var.get():
            self.map_chip_filter_var.set(False)
        self.set_view("Map")

    def _update_region_map_toggle_visibility(self):
        if self.region_map_available and self.view_mode == "Map":
            self.region_map_toggle.place(
                x=12,
                y=12,
                anchor="nw",
            )
            self.region_map_toggle.lift()
        else:
            self.region_map_toggle.place_forget()

    def set_region_map_available(self, available):
        self.region_map_available = bool(available)
        if not self.region_map_available:
            self.region_map_view_var.set(False)
        self._update_region_map_toggle_visibility()
        if self.view_mode == "Map":
            self.display_map()

    def get_region_map_view(self):
        return self.region_map_available and self.region_map_view_var.get()

    def reset_region_map(self, map_id, shape):
        with self._region_map_lock:
            self.region_map = np.zeros(shape, dtype=np.uint8)
            self.region_map_id = map_id
        if self.get_region_map_view() and self.view_mode == "Map":
            self.display_map()

    def place_region_map_frame(self, map_id, image_rgb, map_x, map_y, zoom):
        if (
            map_id is None
            or image_rgb is None
            or map_x is None
            or map_y is None
            or zoom is None
        ):
            return

        step = max(1, int(round(float(zoom))))
        image_small = image_rgb[::step, ::step]
        map_x = int(map_x)
        map_y = int(map_y)

        with self._region_map_lock:
            if map_id != self.region_map_id:
                return
            map_height, map_width = self.region_map.shape[:2]
            x_start = max(0, map_x)
            y_start = max(0, map_y)
            source_x = max(0, -map_x)
            source_y = max(0, -map_y)
            width = min(image_small.shape[1] - source_x, map_width - x_start)
            height = min(image_small.shape[0] - source_y, map_height - y_start)
            if width <= 0 or height <= 0:
                return
            self.region_map[
                y_start:y_start + height,
                x_start:x_start + width,
            ] = image_small[
                source_y:source_y + height,
                source_x:source_x + width,
            ]

    def get_camera_chip_filter(self):
        return self.camera_chip_filter_var.get()

    def get_camera_vignette_filter(self):
        return self.camera_vignette_filter_var.get()

    def toggle_camera_vignette_filter(self):
        if self.camera_vignette_filter_var.get():
            try:
                self.frame_processor.load_vignette_filter()
            except (FileNotFoundError, ValueError) as exc:
                self.disable_camera_vignette_filter(str(exc))
                return
        self.set_view("Camera View")

    def disable_camera_vignette_filter(self, message=None):
        self.camera_vignette_filter_var.set(False)
        if message:
            messagebox.showwarning(
                "Vignette Filter Unavailable",
                message,
                parent=self.root,
            )

    def get_magnification(self):
        return self.magnification

    def set_magnification(self, magnification):
        self.magnification = magnification

    def get_resolution(self):
        return self.resolution
    
    def set_resolution(self, resolution):
        self.resolution = resolution

    def get_true_map(self):
        return self.true_map
    
    def set_true_map(self, map):
        self.true_map = map

    def get_filter_map(self):
        return self.filter_map
    
    def set_filter_map(self, filter_map):
        self.filter_map = filter_map

    def set_active_scan_profile(self, profile):
        self.active_scan_profile = profile

    def get_active_scan_profile(self):
        return self.active_scan_profile

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()

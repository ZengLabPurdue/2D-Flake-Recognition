import os
import sys

import cv2
import numpy as np

from tkinter import *
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from config import CROP_RATIO

from Mapping.mapper import Mapper
from Scanning.scan_manager import ScanManager
from Hardware.stage_controller import StageController
from Hardware.turret_controller import TurretController
from UI.panels.stage_control_panel import StageControlPanel
from UI.panels.objective_control_panel import ObjectiveControlPanel
from UI.panels.focus_panel import FocusPanel
from UI.panels.scan_info_panel import ScanInfoPanel
from UI.panels.view_scans_panel import ViewScansPanel
from UI.panels.capture_panel import CapturePanel
from UI.panels.exposure_panel import ExposurePanel
from Imaging.frame_processing import FrameProcessor
from Imaging.focus import FocusController

DLL_PATH = os.getcwd() + r"\APIs\PriorSDK1.9.2\x64\PriorScientificSDK.dll"
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

        self.true_map = np.zeros((3000, 3000, 3), dtype=np.uint8)
        self.filter_map = np.zeros((3000, 3000), dtype=np.int8)

        self.img_label = Label(self.main_frame, bg="#f0f0f0")
        self.img_label.pack(fill=BOTH, expand=True)

        self.live_mapping_var = tk.BooleanVar(value=False)
        self.filter_var = tk.BooleanVar(value=False)
        self.view_mode = None
        self.set_view("Camera View", False)

        self.hcam = None
        self.width = 0
        self.height = 0
        
        self.magnification = "2X"
        self.resolution = "MED"

        self.buttons = []
        self.panels = []

        self.stage_controller = StageController(PRIOR_COM_PORT, DLL_PATH)
        
        self.frame_processor = FrameProcessor(
            root=self.root,
            app=self,
            stage=self.stage_controller,
            get_live_mapping_status=self.get_live_mapping,
            place_live_frame_on_map=self.mapper.place_live_frame_on_map,
        )

        self.focus_controller = FocusController(
            app=self,
            stage=self.stage_controller,
            frame_processor=self.frame_processor,
        )

        self.turret_controller = TurretController( 
            app=self,
            stage=self.stage_controller,
            turret_port=TURRET_COM_PORT,
            auto_focus=self.focus_controller.auto_focus,
        )

        self.scan_info_panel = ScanInfoPanel(parent=self.main_frame)

        self.mapper = Mapper(
            root=self.root,
            app=self,
            stage=self.stage_controller,
            turret_controller=self.turret_controller,
            frame_processor=self.frame_processor,
            update_scan_status=self.scan_info_panel.update_status,
        )

        self.scan_manager = ScanManager(
            root=self.root,
            app=self,
            stage=self.stage_controller,
            turret_controller=self.turret_controller,
            camera=self.frame_processor.get_camera(),
            frame_processor=self.frame_processor,
            update_scan_status=self.scan_info_panel.update_status,
        )

        self.stage_controller_control_panel = StageControlPanel(
            parent=self.main_frame,
            root=self.root,
            app=self,
            stage=self.stage_controller,
        )

        self.objective_control_panel = ObjectiveControlPanel(
            parent=self.main_frame,
            app=self,
            stage_controller=self.stage_controller,
            turret_controller=self.turret_controller,
        )

        self.capture_panel = CapturePanel(
            parent=self.main_frame,
            app=self,
            save_image=self.frame_processor.save_image,
        )

        self.exposure_panel = ExposurePanel(
            parent=self.main_frame,
            get_camera=self.frame_processor.get_camera,
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

        self.focus_controller.sharpness_callback = self.focus_panel.update_sharpness

        self.panels.append({
            "name": "Info Panel",
            "frame": self.scan_info_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Stage Control Panel",
            "frame": self.stage_controller_control_panel.frame,
            "var": BooleanVar(value=False)
        })
        
        self.panels.append({
            "name": "Capture Panel",
            "frame": self.capture_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Adjust Exposure Panel",
            "frame": self.exposure_panel.frame,
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

        self.update_panels()

        self.init_menu_bar()

        self.root.bind_all("<Button-1>", self.clear_focus, add="+")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.frame_processor.run_camera()        

    # ------------- Initialization -------------

    def init_menu_bar(self):
        menu_bar = Menu(self.root)
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Quit", command=self.on_close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        view_menu = Menu(menu_bar, tearoff=0)

        map_menu = Menu(view_menu, tearoff=0)
        map_menu.add_radiobutton(label="No Filter", variable=self.filter_var, value=False, command=lambda: self.set_view("Map", False))
        map_menu.add_radiobutton(label="Filter", variable=self.filter_var, value=True, command=lambda: self.set_view("Map", True))

        camera_menu = Menu(view_menu, tearoff=0)
        camera_menu.add_radiobutton(label="No Filter", variable=self.filter_var, value=False, command=lambda: self.set_view("Camera View", False))
        camera_menu.add_radiobutton(label="Filter", variable=self.filter_var, value=True, command=lambda: self.set_view("Camera View", True))

        view_menu.add_cascade(label="Map", menu=map_menu)
        view_menu.add_cascade(label="Camera View", menu=camera_menu)

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
        map_menu.add_command(label="Capture Area", command=self.mapper.capture_area)
        menu_bar.add_cascade(label="Map", menu=map_menu)

        scan_menu = Menu(menu_bar, tearoff=0)
        scan_menu.add_command(label="Run Complete Scan (1 Chip)", command=self.scan_manager.run_complete_scan)
        scan_menu.add_command(label="Run Complete Scan", command=lambda: self.scan_manager.run_complete_scan(window=(11, 3)))
        scan_menu.add_command(label="Run 2x Scan", command=lambda: self.scan_manager.run_2x_scan(full_zoom=True))
        scan_menu.add_command(label="Run 10x Scan", command=self.scan_manager.run_10x_scan)
        menu_bar.add_cascade(label="Scan", menu=scan_menu)

        self.view_scans_panel.add_to_menu(menu_bar)

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
        if self.filter_var.get():
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

        img_tk = ImageTk.PhotoImage(display_img)
        self.img_label.configure(image=img_tk)
        self.img_label.image = img_tk

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

        if isinstance(widget, (ttk.Combobox, ttk.Entry)):
            return

        self.root.focus_set()

    def on_close(self):
        self.frame_processor.close()
        self.stage_controller.disconnect()
        self.root.destroy()

    def get_view(self):
        return self.view_mode

    def set_view(self, mode, filter_status):
        self.view_mode = mode

        self.filter_var.set(filter_status)

        if mode == "Map":
            self.display_map()
            self.map_canvas.pack(fill=BOTH, expand=True) # Show map canvas
            self.img_label.pack_forget() # Hide image label
            pass
        elif mode == "Camera View":
            self.img_label.pack(fill=BOTH, expand=True)
            self.map_canvas.pack_forget() # Hide map canvas
        elif mode == "Scan Results":
            self.img_label.pack(fill=BOTH, expand=True)
            self.map_canvas.pack_forget() # Hide map canvas

        self.root.update_idletasks()

    def get_live_mapping(self):
        return self.live_mapping_var.get()
    
    def set_live_mapping(self, live_mapping_status):
        self.live_mapping_var.set(live_mapping_status)

    def get_filter(self):
        return self.filter_var.get()
    
    def set_filter(self, status : bool):
        self.filter_var.set(status)

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

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
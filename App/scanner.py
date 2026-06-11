from collections import deque
import os
import sys
import re
import threading
from queue import Queue

import time
from datetime import datetime

import math
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np

from tkinter import *
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from PIL import Image, ImageTk

from pathlib import Path

import config
from Scanning.scan_manager import ScanManager
from Hardware.stage_controller import StageController
from UI.panels.stage_control_panel import StageControlPanel
from UI.panels.objective_panel import ObjectiveControlPanel
from UI.panels.focus_panel import FocusPanel
from UI.panels.scan_info_panel import ScanInfoPanel
from Imaging.frame_processing import FrameProcessor
from Imaging.focus import FocusController

home_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = home_dir.parent

sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "Flake Recognition"))

from App.Hardware.turret_api import turret
from App.Hardware.turret_controller import TurretController
import flake_identifier

DLL_PATH = os.getcwd() + r"\APIs\PriorSDK1.9.2\x64\PriorScientificSDK.dll"
PRIOR_COM_PORT = sys.argv[1]
TURRET_COM_PORT = sys.argv[2]

try:
    fi = flake_identifier.Flake_Identifier()
except Exception as e:
    print("Failed to connect to Prior Controller:", e)
    sys.exit(1)

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

        self.filter_var = tk.BooleanVar(value=False)
        self.view_mode = None
        self.set_view("Camera View", False)

        self.hcam = None
        self.width = 0
        self.height = 0
        
        self.magnification = "2x"

        self.view_chip_index = 0
        self.view_image_index = 0
        self.view_scan_path = None
        self.view_folder = None

        self.buttons = []
        self.panels = []

        self.scan_info_panel = ScanInfoPanel(parent=self.main_frame)

        self.panels.append({
            "name": "Info Panel",
            "frame": self.scan_info_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.stage_controller = StageController(PRIOR_COM_PORT, DLL_PATH)
        self.turret = turret(TURRET_COM_PORT)

        self.frame_processor = FrameProcessor(
            root=self.root,
            stage=self.stage_controller,
            home_dir=home_dir,
            get_view_mode=lambda: self.view_mode,
            get_filter_status=lambda: self.filter_var.get(),
            get_magnification=lambda: self.magnification,
            #get_live_mapping_status=lambda: self.live_mapping_var.get(),
            display_image=self.display_image,
            display_map=self.display_map,
            #place_live_frame_on_map=self.place_live_frame_on_map,
            disable_buttons=self.disable_buttons,
            enable_buttons=self.enable_buttons,
        )
        self.frame_processor.run_camera()

        self.scan_manager = ScanManager(
            root=self.root,
            home_dir=home_dir,
            stage=self.stage_controller,
            turret=self.turret,
            camera=self.frame_processor.get_camera,
            frame_processor=self.frame_processor,
            get_view_mode=lambda: self.view_mode,
            get_filter_status=self.filter_var.get,
            set_filter_status=self.filter_var.set,
            set_view=self.set_view,
            display_image=self.display_image,
            display_map=self.display_map,
            update_scan_status=self.scan_info_panel.update_status,
            open_panel=self.open_panel,
            get_true_map=self.get_true_map,
            set_true_map=self.set_true_map,
            get_filter_map=self.get_filter_map,
            set_filter_map=self.set_filter_map,
        )

        self.focus_controller = FocusController(
            stage=self.stage_controller,
            frame_processor=self.frame_processor,
            disable_buttons=self.disable_buttons,
            enable_buttons=self.enable_buttons,
        )

        self.stage_controller_control_panel = StageControlPanel(
            parent=self.main_frame,
            root=self.root,
            stage=self.stage_controller,
            disable_buttons=self.disable_buttons,
            enable_buttons=self.enable_buttons,
            register_button=self.buttons.append
        )

        self.panels.append({
            "name": "Stage Control Panel",
            "frame": self.stage_controller_control_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Capture Panel",
            "frame": self.init_capture_panel(),
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Adjust Exposure Panel",
            "frame": self.init_adjust_exposure_panel(),
            "var": BooleanVar(value=False)
        })

        self.objective_control_panel = ObjectiveControlPanel(
            parent=self.main_frame,
            stage=self.stage_controller,
            turret=self.turret,
            get_magnification=lambda: self.magnification,
            set_magnification=self.set_magnification,
            auto_focus=self.focus_controller.auto_focus,
        )

        self.panels.append({
            "name": "Objective Control Panel",
            "frame": self.objective_control_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.focus_panel = FocusPanel(
            parent=self.main_frame,
            focus_controller=self.focus_controller,
            register_button=self.buttons.append,
        )

        self.focus_controller.sharpness_callback = self.focus_panel.update_sharpness

        self.panels.append({
            "name": "Focus Panel",
            "frame": self.focus_panel.frame,
            "var": BooleanVar(value=False)
        })

        self.update_panels()

        self.init_view_scans_panel()
        self.view_scans_panel.place_forget()

        self.init_menu_bar()

        self.root.bind_all("<Button-1>", self.clear_focus, add="+")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------- Initialization -------------

    def init_menu_bar(self):
        menu_bar = Menu(root)
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

        scan_menu = Menu(menu_bar, tearoff=0)
        scan_menu.add_command(label="Run Complete Scan (1 Chip)", command=self.scan_manager.run_complete_scan)
        scan_menu.add_command(label="Run Complete Scan", command=lambda: self.scan_manager.run_complete_scan(window=(11, 3)))
        scan_menu.add_command(label="Run 2x Scan", command=lambda: self.scan_manager.run_2x_scan(full_zoom=True))
        scan_menu.add_command(label="Run 10x Scan", command=self.scan_manager.run_10x_scan)
        menu_bar.add_cascade(label="Scan", menu=scan_menu)

        self.results_menu = Menu(menu_bar, tearoff=0)

        self.results_menu.add_command(label="Open Scan...", command=self.open_scan)
        self.results_menu.add_separator()

        self.open_scan_menu = Menu(self.results_menu, tearoff=0)
        self.open_scan_menu.add_command(label="Raw Images (2x)", command=lambda: self.set_view_folder("Raw 2x"))
        self.open_scan_menu.add_command(label="Raw Images (10x)", command=lambda: self.set_view_folder("Raw 10x"))
        self.open_scan_menu.add_command(label="Processed Images (10x)", command=lambda: self.set_view_folder("Processed 10x"))
        self.open_scan_menu.add_command(label="Detected Flakes", command=lambda: self.set_view_folder("Flakes Found"))
        self.results_menu.add_cascade(label="View Scan", state="disabled", menu=self.open_scan_menu)
        self.results_menu.add_command(label="Classify Flakes", state="disabled", command=None)
        menu_bar.add_cascade(label="Results", menu=self.results_menu)

        root.config(menu=menu_bar)

    # Panel Initialization 

    def init_capture_panel(self):

        self.capture_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=120
        )
        self.capture_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.capture_background = Frame(
            self.capture_panel,
            bg="white",
            width=200,
            height=118
        )
        self.capture_background.place(x=2, y=0)

        capture_title = Label(
            self.capture_panel,
            text="Capture",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        capture_title.place(relx=0.5, y=10, anchor="n")

        style = ttk.Style()
        style.configure("Save.TButton", background="white")
        style.configure("Save.TButton", relief="flat")

        self.capture_image_button = ttk.Button(
            self.capture_background,
            text="Save Image",
            style="Save.TButton",
            command=self.frame_processor.save_image
        )
        self.capture_image_button.place(relx=0.5, y=55, anchor="center")
        self.buttons.append(self.capture_image_button)

        self.capture_map_button = ttk.Button(
            self.capture_background,
            text="Save Map",
            style="Save.TButton",
            command=lambda: self.frame_processor.save_image(image=cv2.cvtColor(self.true_map, cv2.COLOR_RGB2BGR))
        )
        self.capture_map_button.place(relx=0.5, y=90, anchor="center")
        self.buttons.append(self.capture_map_button)

        return self.capture_panel
    
    def init_adjust_exposure_panel(self):
        self.adjust_exposure_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=100
        )
        self.adjust_exposure_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.adjust_exposure_background = Frame(
            self.adjust_exposure_panel,
            bg="white",
            width=200,
            height=98
        )
        self.adjust_exposure_background.place(x=2, y=0)

        adjust_exposure_title = Label(
            self.adjust_exposure_panel,
            text="Adjust Exposure",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        adjust_exposure_title.place(relx=0.5, y=5, anchor="n")

        style = ttk.Style()
        style.configure("Custom.Horizontal.TScale", background="white")

        self.exposure_var = DoubleVar(value=config.DEFAULT_EXPOSURE)
        self.adjust_exposure_slider = ttk.Scale(
            self.adjust_exposure_background,
            from_=30,
            to=120,
            orient="horizontal",
            variable=self.exposure_var,
            command=self.adjust_exposure,
            style="Custom.Horizontal.TScale"
        )
        self.adjust_exposure_slider.place(relx=0.5, y=50, anchor="center")

        self.exposure_value_label = Label(
            self.adjust_exposure_background,
            text=f"Exposure: {config.DEFAULT_EXPOSURE}",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 8)
        )
        self.exposure_value_label.place(relx=0.5, y=70, anchor="n")

        return self.adjust_exposure_panel
 
    def init_view_scans_panel(self):

        self.pos_scan_name = 40
        self.pos_chip = 70
        self.pos_image = 100
        self.pos_buttons = 140

        self.view_scans_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=205
        )
        self.view_scans_panel.place(relx=0.0, rely=0.0, anchor="nw")

        self.view_scans_background = Frame(
            self.view_scans_panel,
            bg="white",
            width=200,
            height=203
        )
        self.view_scans_background.place(x=2, y=0)

        view_scans_title = Label(
            self.view_scans_panel,
            text="Scan Results",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        view_scans_title.place(relx=0.5, y=5, anchor="n")

        self.scan_name_var = tk.StringVar()
        self.scan_name_var.set("Scan: Not Selected")

        self.scan_name_label = Label(
            self.view_scans_panel,
            textvariable=self.scan_name_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        self.scan_name_label.place(relx=0.5, y=self.pos_scan_name, anchor="n")

        self.chip_var = tk.StringVar()

        self.chip_dropdown = ttk.Combobox(
            self.view_scans_panel,
            textvariable=self.chip_var,
            state="readonly"
        )

        self.chip_dropdown.place(relx=0.5, y=self.pos_chip, anchor="n")

        self.image_var = tk.StringVar()
        self.image_var.set("Image: None")

        self.image_label = Label(
            self.view_scans_panel,
            textvariable=self.image_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        self.image_label.place(relx=0.5, y=self.pos_image, anchor="n")

        self.views_scans_button_panel = Frame(self.view_scans_panel, bg="white", width=80, height=45)
        self.views_scans_button_panel.place(relx=0.5, x=0, y=self.pos_buttons, anchor="n")
        self.views_scans_button_panel.pack_propagate(False)

        view_scans_controls = Frame(self.views_scans_button_panel, bg="white")
        view_scans_controls.pack(expand=True, fill="both")

        self.btn_next = ttk.Button(view_scans_controls, text="▸", style="Arrow.TButton")
        self.btn_previous = ttk.Button(view_scans_controls, text="◂", style="Arrow.TButton")

        self.btn_next.bind("<ButtonPress-1>", self.next_image)
        self.btn_previous.bind("<ButtonPress-1>", self.previous_image)

        self.root.bind("<Left>", self.previous_image)
        self.root.bind("<Right>", self.next_image)

        view_scans_controls.rowconfigure(0, weight=1)
        view_scans_controls.columnconfigure(0, weight=1)
        view_scans_controls.columnconfigure(1, weight=1)

        self.btn_next.grid(row=0, column=1, sticky="nsew")
        self.btn_previous.grid(row=0, column=0, sticky="nsew")

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

    def set_magnification(self, magnification):
        self.magnification = magnification

    # ------------- View Scan Functions -------------

    def previous_image(self, event=None):

        if self.view_mode != "Scan Results":
            return

        if self.image_files is None:
            return

        self.view_image_index = (self.view_image_index - 1) % len(self.image_files)

        self.image_var.set(f"Image: {self.image_files[self.view_image_index].name}")

        img_path = self.image_files[self.view_image_index]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.display_image(img)

        self.root.focus_set()

    def next_image(self, event=None):
        
        if self.view_mode != "Scan Results":
            return

        if self.image_files is None:
            return

        self.view_image_index = (self.view_image_index + 1) % len(self.image_files)

        self.image_var.set(f"Image: {self.image_files[self.view_image_index].name}")

        img_path = self.image_files[self.view_image_index]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.display_image(img)

        self.root.focus_set()

    # ------------- Display Functions -------------

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

        self.root.update()

    def draw_map(self):
        map_img = Image.fromarray(self.filter_map, mode="L")

        canvas_w = self.map_canvas.winfo_width()
        canvas_h = self.map_canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            self.root.after(100, self.draw_map)
            return

        scale = min(canvas_w / map_img.width, canvas_h / map_img.height, 1.0)
        new_w = int(map_img.width * scale)
        new_h = int(map_img.height * scale)
        map_img_resized = map_img.resize((new_w, new_h), Image.Resampling.NEAREST)

        self.map_tk = ImageTk.PhotoImage(map_img_resized)

        x_center = canvas_w // 2
        y_center = canvas_h // 2

        self.map_canvas.delete("all")
        self.map_canvas.create_image(
            x_center, y_center,
            anchor="center",
            image=self.map_tk
        )

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
            if self.magnification == "2x":
                crop_w = int(w * config.CENTER_CROP_WIDTH_RATIO_2X)
                crop_h = int(h * config.CENTER_CROP_HEIGHT_RATIO_2X)
            elif self.magnification == "10x":
                crop_w = int(w * config.CENTER_CROP_WIDTH_RATIO_10X)
                crop_h = int(h * config.CENTER_CROP_HEIGHT_RATIO_10X)
            elif self.magnification == "20x":
                crop_w = int(w * config.CENTER_CROP_WIDTH_RATIO_20X)
                crop_h = int(h * config.CENTER_CROP_HEIGHT_RATIO_20X)
            elif self.magnification == "100x":
                crop_w = int(w * config.CENTER_CROP_WIDTH_RATIO_100X)
                crop_h = int(h * config.CENTER_CROP_HEIGHT_RATIO_100X)
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

    def adjust_exposure(self, exposure):
        self.hcam.put_AutoExpoTarget(int(float(exposure)))
        self.exposure_value_label.config(text=f"Exposure: {int(float(self.hcam.get_AutoExpoTarget()))}")

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

    def open_scan(self):
        folder = filedialog.askdirectory(title="Select Scan Folder")
        if not folder:
            return

        path = Path(folder)
        folder_name = path.name

        pattern = r"^Full Scan \(\d{4}-\d{2}-\d{2}\) \(\d{2}-\d{2}-\d{2}\)$"

        if not re.match(pattern, folder_name):
            messagebox.showwarning(
                "Invalid Folder",
                "Selected folder is not a valid scan folder."
            )
            return

        self.view_scan_path = path

        self.scan_name_var.set(f"{folder_name}")

        self.results_menu.entryconfig("View Scan", state="normal")
        self.results_menu.entryconfig("Classify Flakes", state="normal") 

        messagebox.showinfo(
            "Scan Loaded",
            f"Scan loaded successfully!"
        )
    
    def set_view_folder(self, selected_view):

        self.set_view("Scan Results", False)
        
        self.view_scans_panel.place(relx=0.0, rely=0.0, anchor="nw")
        
        self.view_chip_index = 0
        self.view_image_index = 0
        
        base_path = self.view_scan_path / "All Images"

        if selected_view == "Raw 2x":
            self.view_folder = base_path / "2x" / "Raw"
            self.display_chip_dropdown(False)

        elif selected_view == "Raw 10x":
            chip_folder = self.get_subfolder(base_path / "10x", self.view_chip_index).name
            self.view_folder = base_path / "10x" / chip_folder / "Raw"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown()            

        elif selected_view == "Processed 10x":
            chip_folder = self.get_subfolder(base_path / "10x", self.view_chip_index).name
            self.view_folder = base_path / "10x" / chip_folder / "Processed"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown() 

        elif selected_view == "Flakes Found":
            chip_folder = self.get_subfolder(self.view_scan_path / "Flakes Found", self.view_chip_index).name
            self.view_folder = self.view_scan_path / "Flakes Found" / chip_folder
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown()

        self.image_files = sorted([p for p in self.view_folder.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]], key=self.image_sort_key)

        img_path = self.image_files[self.view_image_index]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.display_image(img)
        self.root.update()

    def populate_chips_dropdown(self):
        chip_root = self.view_scan_path / "All Images" / "10x"

        if not chip_root.exists():
            self.chip_dropdown["values"] = []
            self.chip_var.set("No chips found")
            return

        chips = sorted([p.name for p in chip_root.iterdir() if p.is_dir()])

        self.chip_dropdown["values"] = chips

        if chips:
            self.chip_var.set(chips[0])
        else:
            self.chip_var.set("No chips found")

    def image_sort_key(self, p):
        match = re.search(r'_(\d+)\.', p.name)
        return int(match.group(1)) if match else -1

    def get_subfolder(self, path, index):
        subfolders = [p for p in path.iterdir() if p.is_dir()]
        subfolders = sorted(subfolders)
        return subfolders[index] if 0 <= index < len(subfolders) else None

    def clear_focus(self, event):
        widget = event.widget

        if isinstance(widget, (ttk.Combobox, ttk.Entry)):
            return

        self.root.focus_set()

    def on_close(self):
        self.frame_processor.close()
        self.stage_controller.disconnect()
        self.root.destroy()

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
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

from Hardware.stage_controller import StageController
from UI.panels.stage_control_panel import StageControlPanel
from Imaging.frame_processing import FrameProcessor

home_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = home_dir.parent

sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "Flake Recognition"))

from APIs.prior_api import Prior_Controller
from APIs.turret_api import Turret_Controller
import chip_edge_classifier
import flake_identifier
import flake_identifier_yolo
import vignetting_corrector

DLL_PATH = os.getcwd() + r"\APIs\PriorSDK1.9.2\x64\PriorScientificSDK.dll"
PRIOR_COM_PORT = sys.argv[1]
TURRET_COM_PORT = sys.argv[2]
DEFAULT_EXPOSURE = 60

CENTER_CROP_WIDTH_RATIO_2X = 0.7
CENTER_CROP_HEIGHT_RATIO_2X = 0.7

CENTER_CROP_WIDTH_RATIO_10X = 0.9
CENTER_CROP_HEIGHT_RATIO_10X = 0.9

CENTER_CROP_WIDTH_RATIO_20X = 1
CENTER_CROP_HEIGHT_RATIO_20X = 1

CENTER_CROP_WIDTH_RATIO_100X = 1
CENTER_CROP_HEIGHT_RATIO_100X = 1

RELATIVE_2X_Z = 0
RELATIVE_10X_Z = 1250
RELATIVE_20X_Z = 4300
RELATIVE_100X_Z = 4300

X_SIZE_2 = 10642
Y_SIZE_2 = 7027

X_SIZE_10 = 2142 
Y_SIZE_10 = 1359

MAGNIFICATION = 2

IMAGE_UM_PER_PIXEL_2X_MED = 3.8569 # um
IMAGE_UM_PER_PIXEL_10X_MED = 0.76609 # um

FLATFIELD_IMG = cv2.imread(str(home_dir / "Flatfields" / "flatfield_2x_med_smoothed.png"))

try:
    tc = Turret_Controller(TURRET_COM_PORT)
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

        self.scan_running = False

        self.hold_job = None
        self.is_hold = False

        self.hcam = None
        self.buf = None
        self.prevImg = None
        self.width = 0
        self.height = 0
        
        self.magnification = "2x"

        self.auto_focus_range = 1000
        self.auto_focus_accuracy = 10
        self.auto_focus_steps = 20

        self.view_chip_index = 0
        self.view_image_index = 0
        self.view_scan_path = None
        self.view_folder = None

        self.buttons = []
        self.panels = []

        self.panels.append({
            "name": "Info Panel",
            "frame": self.init_scan_info_panel(),
            "var": BooleanVar(value=False)
        })

        self.stage = StageController(PRIOR_COM_PORT, DLL_PATH)
        self.frame_processor = FrameProcessor(
            root=self.root,
            stage=self.stage,
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

        self.stage_control_panel = StageControlPanel(
            parent=self.main_frame,
            root=self.root,
            stage=self.stage,
            disable_buttons=self.disable_buttons,
            enable_buttons=self.enable_buttons,
            register_button=self.buttons.append
        )

        self.panels.append({
            "name": "Stage Control Panel",
            "frame": self.stage_control_panel.frame,
            "var": BooleanVar(value=False)
        })

        '''
        self.panels.append({
            "name": "Stage Control Panel",
            "frame": self.init_stage_control_panel(),
            "var": BooleanVar(value=False)
        })
        '''

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

        self.panels.append({
            "name": "Objective Control Panel",
            "frame": self.init_objective_control_panel(),
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Focus Panel",
            "frame": self.init_focus_panel(),
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
        scan_menu.add_command(label="Run Complete Scan (1 Chip)", command=self.run_complete_scan)
        scan_menu.add_command(label="Run Complete Scan", command=lambda: self.run_complete_scan(window=(11, 3)))
        scan_menu.add_command(label="Run 2x Scan", command=lambda: self.run_2x_scan(full_zoom=True))
        scan_menu.add_command(label="Run 10x Scan", command=self.run_10x_scan)
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

    def init_scan_info_panel(self):

        self.scan_info_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=147
        )
        self.scan_info_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.info_background = Frame(
            self.scan_info_panel,
            bg="white",
            width=200,
            height=145
        )
        self.info_background.place(x=2, y=0)

        title_label = Label(
            self.scan_info_panel,
            text="Info",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title_label.place(relx=0.5, y=10, anchor="n")

        self.scan_info_type_label = Label(
            self.scan_info_panel,
            text=f"Scan: None",
            bg="white",
            fg="black"
        )
        self.scan_info_type_label.place(relx=0.5, y=35, anchor="n")

        self.scan_info_stage_label = Label(
            self.scan_info_panel,
            text="Stage: Not Started",
            bg="white",
            fg="black"
        )
        self.scan_info_stage_label.place(relx=0.5, y=55, anchor="n")

        self.scan_info_progress_label = Label(
            self.scan_info_panel,
            text="Progress: Not Started",
            bg="white",
            fg="black"
        )
        self.scan_info_progress_label.place(relx=0.5, y=75, anchor="n")

        self.scan_info_stage_time_label = Label(
            self.scan_info_panel,
            text="Stage Time Elapsed: Not Started",
            bg="white",
            fg="black"
        )
        self.scan_info_stage_time_label.place(relx=0.5, y=95, anchor="n")

        self.scan_info_total_time_label = Label(
            self.scan_info_panel,
            text="Total Time Elapsed: Not Started",
            bg="white",
            fg="black"
        )
        self.scan_info_total_time_label.place(relx=0.5, y=115, anchor="n")

        return self.scan_info_panel
    
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

        self.exposure_var = DoubleVar(value=DEFAULT_EXPOSURE)
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
            text=f"Exposure: {DEFAULT_EXPOSURE}",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 8)
        )
        self.exposure_value_label.place(relx=0.5, y=70, anchor="n")

        return self.adjust_exposure_panel

    def init_objective_control_panel(self):
        self.objective_control_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=240
        )
        self.objective_control_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.objective_control_background = Frame(
            self.objective_control_panel,
            bg="white",
            width=200,
            height=238
        )
        self.objective_control_background.place(x=2, y=0)

        objective_control_title = Label(
            self.objective_control_panel,
            text="Objective Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        objective_control_title.place(relx=0.5, y=5, anchor="n")

        self.objective_var = tk.StringVar()
        self.objective_var.set("Objective: Unknown")

        self.objective_label = Label(
            self.objective_control_panel,
            textvariable=self.objective_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )

        self.objective_label.place(relx=0.5, y=40, anchor="n")

        style = ttk.Style()
        style.configure("Custom.Horizontal.TScale", background="white")

        self.objective_control_button_panel = Frame(
            self.objective_control_panel,
            bg="white",
            width=150,
            height=150
        )
        self.objective_control_button_panel.place(x=26, y=70)
        self.objective_control_button_panel.pack_propagate(False)

        controls = Frame(self.objective_control_button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        style = ttk.Style()
        style.configure("Custom.TButton", font=("TkDefaultFont", 10), padding=5)
        style.configure("Custom.TButton", background="white", relief="flat")
        
        self.btn1 = ttk.Button(controls, text="1", style="Custom.TButton")
        self.btn2 = ttk.Button(controls, text="2", style="Custom.TButton")
        self.btn3 = ttk.Button(controls, text="3", style="Custom.TButton")
        self.btn4 = ttk.Button(controls, text="4", style="Custom.TButton")
        self.btn5 = ttk.Button(controls, text="5", style="Custom.TButton")

        self.objective_buttons = [self.btn1, self.btn2, self.btn3, self.btn4, self.btn5]

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

        #self.change_objective(1)

        tc.turn_to_position(1)
        self.objective_var.set(f"Objective: {1}")

        return self.objective_control_panel

    def init_focus_panel(self):
        self.focus_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=205
        )
        self.focus_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.focus_background = Frame(
            self.focus_panel,
            bg="white",
            width=200,
            height=203
        )
        self.focus_background.place(x=2, y=0)

        focus_title = Label(
            self.focus_panel,
            text="Focus Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        focus_title.place(relx=0.5, y=5, anchor="n")

        self.sharpness_var = tk.StringVar()
        self.sharpness_var.set("Sharpness: Unknown")

        self.sharpness_label = Label(
            self.focus_panel,
            textvariable=self.sharpness_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )

        self.sharpness_label.place(relx=0.5, y=35, anchor="n")

        label_x = 10
        entry_x = 130

        range_label = Label(
            self.focus_panel,
            text="Range:",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        range_label.place(relx=0.0, rely=0.0, x=label_x, y=65)

        self.auto_focus_range_var = StringVar(value=str(self.auto_focus_range))
        self.auto_focus_range_entry = ttk.Entry(
            self.focus_panel,
            textvariable=self.auto_focus_range_var,
            width=8
        )
        self.auto_focus_range_entry.place(relx=0.0, rely=0.0, x=entry_x, y=65)

        accuracy_label = Label(
            self.focus_panel,
            text="Accuracy:",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        accuracy_label.place(relx=0.0, rely=0.0, x=label_x, y=95)

        self.auto_focus_accuracy_var = StringVar(value=str(self.auto_focus_accuracy))
        self.auto_focus_accuracy_entry = ttk.Entry(
            self.focus_panel,
            textvariable=self.auto_focus_accuracy_var,
            width=8
        )
        self.auto_focus_accuracy_entry.place(relx=0.0, rely=0.0, x=entry_x, y=95)

        step_label = Label(
            self.focus_panel,
            text="Num Steps:",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        step_label.place(relx=0.0, rely=0.0, x=label_x, y=125)

        self.auto_focus_steps_var = StringVar(value=str(self.auto_focus_steps))
        self.auto_focus_steps_entry = ttk.Entry(
            self.focus_panel,
            textvariable=self.auto_focus_steps_var,
            width=8
        )
        self.auto_focus_steps_entry.place(relx=0.0, rely=0.0, x=entry_x, y=125)

        self.auto_focus_btn = ttk.Button(self.focus_panel, text="Auto Focus", style="Normal.TButton", command=lambda: self.auto_focus(start_range=int(self.auto_focus_range_var.get()), accuracy=int(self.auto_focus_accuracy_var.get()), steps=int(self.auto_focus_steps_var.get())))
        self.auto_focus_btn.place(relx=0.5, y=160, anchor="n")
        self.buttons.append(self.auto_focus_btn)

        return self.focus_panel
    
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

    # ------------- Objective Control Functions -------------

    def change_objective(self, position):

        objective_map = {
            1: ("2x", RELATIVE_2X_Z),
            2: ("10x", RELATIVE_10X_Z),
            3: ("20x", RELATIVE_20X_Z),
            4: (None, RELATIVE_20X_Z),
            5: ("100x", RELATIVE_100X_Z),
        }

        current_position = tc.check_position()

        if position == current_position:
            return

        current_z = pc.get_curr_z_pos()

        _, current_rel_z = objective_map.get(current_position, (None, 0))
        magnification, target_rel_z = objective_map.get(position, (None, 0))

        change_z = target_rel_z - current_rel_z

        pc.go_to_z_pos(current_z + change_z)
        tc.turn_to_position(position)

        if position == 1:
            self.auto_focus()
        elif position == 2:
            #self.auto_focus(start_range=500, accuracy=10, steps=20)
            pass
        elif position == 3:
            self.auto_focus(start_range=200, accuracy=5, steps=20)
        elif position == 4:
            #self.auto_focus(start_range=50, accuracy=2, steps=10)
            pass

        self.magnification = magnification
        self.objective_var.set(f"Objective: {position}")

        self.enable_buttons()

    def find_sharpness(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3,3), 0)

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        self.sharpness_var.set(f"Sharpness: {sharpness:.3f}")

        return sharpness

    def get_raw_sharpness(self, num_images=2):
        self.wait_until_new_frame()

        sharpness = 0

        for _ in range(num_images):
            self.wait_until_new_frame()
            sharpness += self.find_sharpness(self.current_frame)

        sharpness /= num_images

        return sharpness

    def discard_initial_frame(self, position):
        discard_z = position
        pc.go_to_z_pos(discard_z)
        self.get_position()
        self.get_raw_sharpness(num_images=3)

    def find_best_focus(self, z_start, z_end, steps, tolerance=0.2):

        best_sharpness = -1
        best_z = z_start

        z_positions = [
            z_start + i * (z_end - z_start) / steps
            for i in range(steps + 1)
        ]

        print(f"Speed: {pc.get_z_velocity()}, Step: {int((z_end - z_start) / steps)}")
        #pc.set_velocity(int((z_end - z_start) / steps))
        
        curr_z = pc.get_curr_z_pos()

        if abs(curr_z - z_positions[0]) < abs(curr_z - z_positions[-1]):
            z_positions.reverse()

        self.discard_initial_frame(z_positions[0])

        for z in z_positions:

            pc.go_to_z_pos(z)
            self.get_position()

            sharpness = self.get_raw_sharpness(num_images=3)

            print(f"Z: {z:>12.1f} | Sharpness: {sharpness:>8.3f} | Best Sharpness: {best_sharpness:>8.3f} | Best Z: {best_z:>12.1f}")

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_z = z
                drops = 0
            else:
                if sharpness < best_sharpness - tolerance:
                    drops += 1

            if drops >= 2:
                print("Focus peak passed")
                break

        pc.go_to_z_pos(best_z)

        return best_z

    def auto_focus(self, start_range=3000, accuracy=50, steps=20):
        start_time = time.time()
        self.disable_buttons()
        _range = start_range
        best_z = pc.get_curr_z_pos()
        while _range >= accuracy:
            best_z = self.find_best_focus(best_z-_range, best_z+_range, steps)
            pc.go_to_z_pos(best_z)
            self.discard_initial_frame(best_z)
            print(f"Best Z: {best_z:>12.1f} | Sharpness: {self.get_raw_sharpness(num_images=3):>8.3f} | Range: {_range}")
            print("-----------------------------------")
            _range = int(_range / (steps / 2))
        print(f"Time taken: {time.time() - start_time:.2f}s")
        self.enable_buttons()

    # ------------- Scanning Functions -------------

    def run_complete_scan(self, window=(3, 3)):

        self.open_panel("Info Panel")

        start_time = time.time()

        scan_path = home_dir / "Scans" / datetime.now().strftime("Full Scan (%Y-%m-%d) (%H-%M-%S)")

        self.update_scan_status(scan_type="Full Scan")

        center_x, center_y, scale_2x = self.run_2x_scan(scan_path=scan_path, full_scan=True, full_scan_start_time=start_time, window=window, full_zoom=True)
        chips = self.find_chips(self.filter_map)
        scan_coordinates = self.generate_10x_scan_coordinates(chips, center_x, center_y, scale_2x)

        image_queue = Queue(maxsize=200)

        flake_detection_thread = threading.Thread(
            target=self.run_10x_flake_detection,
            kwargs={"image_queue": image_queue},
            daemon=True
        )

        flake_detection_thread.start()

        self.run_10x_scan(scan_coordinates, scan_path=scan_path, full_scan=True, full_scan_start_time=start_time, image_queue=image_queue)
        image_queue.put(None)

        flake_detection_thread.join()

        print("Full scan finished!")
        print(f"Time taken: {time.time() - start_time:.2f}s")

        self.go_to_position(x = 0,  y = 0)
        self.change_objective(1)

    def run_2x_scan(self, window=(3, 3), scan_path=None, zoom=6, full_scan=False, full_scan_start_time = None, full_zoom=False):

        self.open_panel("Info Panel")

        print("2x scan running...")

        self.change_objective(1)

        self.set_view("Map", True)

        start_time = time.time()
        if scan_path is None:
            path = home_dir / "Scans" / datetime.now().strftime("2x (%Y-%m-%d) (%H-%M-%S)")
        else:
            path = scan_path / "All Images" / "2x"

        self.true_map = np.zeros((3000, 3000, 3), dtype=np.uint8)
        self.filter_map = np.zeros((3000, 3000), dtype=np.uint8)
        self.scan_running = True

        global x_pos, y_pos

        center_x = x_pos
        center_y = y_pos

        #coords, total_frames = self.generate_spiral_coords(max(num_steps_x, num_steps_y))
        coords, total_frames = self.generate_rect_coords(window[1], window[0])

        zoom = max(zoom, int(self.hcam.get_Size()[1] / (self.true_map.shape[0] / window[1])), int(self.hcam.get_Size()[0] / (self.true_map.shape[1] / window[0])))
        
        if full_zoom:
            zoom = max(int(self.hcam.get_Size()[1] / (self.true_map.shape[0] / window[1])), int(self.hcam.get_Size()[0] / (self.true_map.shape[1] / window[0])))

        if full_scan:
            self.update_scan_status(stage="2x Scan", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")
        else:
            self.update_scan_status(scan_type="2x Scan", stage="2x Scan", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")

        i = 1
        for offset_x, offset_y in coords:

            '''
            target_x = center_x + offset_x * self.hcam.get_Size()[0] * IMAGE_UM_PER_PIXEL_2X_MED * CENTER_CROP_WIDTH_RATIO_2X
            target_y = center_y - offset_y * self.hcam.get_Size()[1] * IMAGE_UM_PER_PIXEL_2X_MED * CENTER_CROP_HEIGHT_RATIO_2X
            '''
            
            target_x = center_x + offset_x * X_SIZE_2 * CENTER_CROP_WIDTH_RATIO_2X
            target_y = center_y - offset_y * Y_SIZE_2 * CENTER_CROP_HEIGHT_RATIO_2X

            pc.go_to_pos(target_x, target_y)
            x_pos, y_pos = target_x, target_y

            pc.wait_until_not_busy()

            img = self.frame_processor.capture_frame()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image_path = path / "Raw" / f"img_2x_{i}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            self.frame_processor.save_image(image=img, filename=image_path)

            binary = chip_edge_classifier.chip_filter(img, display=False)
            img_binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

            if self.view_mode == "Camera View":
                if self.filter_var.get():
                    self.display_image(cv2.cvtColor(chip_edge_classifier.chip_filter(img), cv2.COLOR_GRAY2RGB))
                else:
                    self.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            map_x = int(self.filter_map.shape[1] / 2 - (offset_x + 0.5) * img_binary_rgb.shape[1] / zoom)
            map_y = int(self.filter_map.shape[0] / 2 + (offset_y - 0.5) * img_binary_rgb.shape[0] / zoom)

            h, w = img_binary_rgb.shape[:2]
            img_small = img_rgb[::zoom, ::zoom]
            img_binary_small = img_binary_rgb[::zoom, ::zoom, 0]

            x_start = max(0, map_x)
            y_start = max(0, map_y)
            x_end = min(self.filter_map.shape[1], x_start + img_binary_small.shape[1])
            y_end = min(self.filter_map.shape[0], y_start + img_binary_small.shape[0])

            self.true_map[y_start:y_end, x_start:x_end] = img_small[:y_end - y_start, :x_end - x_start]
            self.filter_map[y_start:y_end, x_start:x_end] = img_binary_small[:y_end - y_start, :x_end - x_start]

            stage_elapsed = time.time() - start_time
            if full_scan_start_time is not None:
                total_elapsed = time.time() - full_scan_start_time
                total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
            stage_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stage_elapsed))
            progress_percent = f"{(i)}/{total_frames} ({(i)*100//total_frames}%)"
            i = i + 1
    
            if full_scan:
                self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=total_elapsed_str)
            else:
                self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=stage_elapsed_str)

        self.scan_running = False
        pc.go_to_pos(center_x, center_y)
        print("2x scan imaging finished!")
        print("Time taken: {:.2f}s".format(time.time() - start_time))

        return center_x, center_y, zoom

    def run_10x_scan(self, scan_coordinates_10x=None, scan_path=None, image_queue=None, zoom=4, full_scan=False, full_scan_start_time=None):

        self.open_panel("Info Panel")

        start_time = time.time()

        self.set_view("Map", False)

        self.change_objective(2)

        input("Press Enter to start 10x scan...")

        if scan_path is None:
            path = home_dir / "Scans" / datetime.now().strftime("10x (%Y-%m-%d) (%H-%M-%S)")
        else:
            path = scan_path / "All Images" / "10x"

        if scan_coordinates_10x is None:
            x, y, _ = pc.get_curr_pos()
            scan_coordinates_10x = [[x, y, 10, 10]]

        cropped_flatfield = self.frame_processor.crop_frame(FLATFIELD_IMG)

        i = 0
        for coordinates in scan_coordinates_10x:

            chip_time = time.time()

            i += 1
            self.true_map = np.zeros((3000, 3000, 3), dtype=np.uint8)
            self.scan_running = True

            if full_scan:
                self.update_scan_status(stage=f"10x Scan - Chip {i} / {len(scan_coordinates_10x)}", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")
            else:
                self.update_scan_status(scan_type="10x Scan", stage="10x Scan", progress="0%", stage_elapsed_time="00:00:00", total_elapsed_time="00:00:00")

            global x_pos, y_pos

            center_x = coordinates[0]
            center_y = coordinates[1]

            coords, total_frames = self.generate_rect_coords(coordinates[2], coordinates[3])

            pc.go_to_pos(center_x, center_y)

            max_zoom = max(zoom, int(self.hcam.get_Size()[1] / (self.true_map.shape[0] / coordinates[2])), int(self.hcam.get_Size()[0] / (self.true_map.shape[1] / coordinates[3])))

            j = 0
            for offset_x, offset_y in coords:
                '''
                target_x = center_x + offset_x * self.hcam.get_Size()[0] * IMAGE_UM_PER_PIXEL_10X_MED * CENTER_CROP_WIDTH_RATIO_10X
                target_y = center_y - offset_y * self.hcam.get_Size()[1] * IMAGE_UM_PER_PIXEL_10X_MED * CENTER_CROP_HEIGHT_RATIO_10X
                '''
                
                target_x = center_x + offset_x * X_SIZE_10 * CENTER_CROP_WIDTH_RATIO_10X
                target_y = center_y - offset_y * Y_SIZE_10 * CENTER_CROP_HEIGHT_RATIO_10X

                pc.go_to_pos(target_x, target_y)
                x_pos, y_pos = target_x, target_y

                pc.wait_until_not_busy()

                img = self.frame_processor.capture_frame()
                img = vignetting_corrector.vignetting_correction_direct_single_channel(img, cropped_flatfield, reference_point=(img.shape[1]//2, img.shape[0]//2))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                image_path = path / f"Chip {i} ({center_x}, {center_y})" / "Raw" / f"img_10x_{j}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                self.frame_processor.save_image(image=img, filename=image_path)

                if image_queue is not None:
                    image_queue.put(image_path)

                if self.view_mode == "Camera View":
                    if self.filter_var.get():
                        self.filter_var.set(False)
                    self.root.after(0, lambda img=img: self.display_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

                map_x = int(self.filter_map.shape[1] / 2 - (offset_x + 0.5) * img_rgb.shape[1] / max_zoom)
                map_y = int(self.filter_map.shape[0] / 2 + (offset_y - 0.5) * img_rgb.shape[0] / max_zoom)

                h, w = img_rgb.shape[:2]
                img_small = img_rgb[::max_zoom, ::max_zoom]

                x_start = max(0, map_x)
                y_start = max(0, map_y)
                x_end = min(self.filter_map.shape[1], x_start + img_small.shape[1])
                y_end = min(self.filter_map.shape[0], y_start + img_small.shape[0])

                j += 1

                self.true_map[y_start:y_end, x_start:x_end] = img_small[:y_end - y_start, :x_end - x_start]

                self.display_map()

                stage_elapsed = time.time() - chip_time
                if full_scan_start_time is not None:
                    total_elapsed = time.time() - full_scan_start_time
                    total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
                stage_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stage_elapsed))
                progress_percent = f"{(j)}/{total_frames} ({(j)*100//total_frames}%)"
    
                if full_scan:
                    self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=total_elapsed_str)
                else:
                    self.update_scan_status(progress=progress_percent, stage_elapsed_time=stage_elapsed_str, total_elapsed_time=stage_elapsed_str)

            print("Chip {} imaging finished!".format(i))
            print("Time taken: {:.2f}s".format(time.time() - chip_time))

        if image_queue is not None:
            image_queue.put(None)
        self.scan_running = False

        print("10x scan imaging finished!")
        print("Time taken: {:.2f}s".format(time.time() - start_time))

    def find_chips(self, binary_map):
        binary_map = (self.filter_map > 0).astype("uint8") * 255
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = [c for c in contours if cv2.contourArea(c) >= 1000]

        chips = []

        for i, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            chips.append((x,y,w,h))
            cv2.rectangle(self.true_map, (x, y), (x+w, y+h), (255,255,0), 5)

        return chips
    
    def generate_10x_scan_coordinates(self, chips, scan_center_x, scan_center_y, scale):

        window_w = int(self.hcam.get_Size()[0] * CENTER_CROP_WIDTH_RATIO_2X / scale / (IMAGE_UM_PER_PIXEL_2X_MED / IMAGE_UM_PER_PIXEL_10X_MED) / CENTER_CROP_WIDTH_RATIO_10X)
        window_h = int(self.hcam.get_Size()[1] * CENTER_CROP_HEIGHT_RATIO_2X / scale / (IMAGE_UM_PER_PIXEL_2X_MED / IMAGE_UM_PER_PIXEL_10X_MED) / CENTER_CROP_HEIGHT_RATIO_10X)
        
        #window_w = int(self.hcam.get_Size()[0] / scale / (IMAGE_UM_PER_PIXEL_2X_MED / IMAGE_UM_PER_PIXEL_10X_MED))
        #window_h = int(self.hcam.get_Size()[1] / scale / (IMAGE_UM_PER_PIXEL_2X_MED / IMAGE_UM_PER_PIXEL_10X_MED))

        scan_coordinates_10x = []

        for chip in chips:
            x, y, w, h = chip

            num_windows_x = math.ceil(w / window_w)
            num_windows_y = math.ceil(h / window_h)

            grid_w = num_windows_x * window_w
            grid_h = num_windows_y * window_h

            chip_center_x = x + w // 2
            chip_center_y = y + h // 2

            start_x = max(0, chip_center_x - grid_w // 2)
            start_y = max(0, chip_center_y - grid_h // 2)

            start_pos_x = - (chip_center_x - self.true_map.shape[1] / 2) * (X_SIZE_2 * CENTER_CROP_WIDTH_RATIO_2X) / (self.hcam.get_Size()[0] / scale * CENTER_CROP_WIDTH_RATIO_2X) + scan_center_x
            start_pos_y = - (chip_center_y - self.true_map.shape[0] / 2) * (Y_SIZE_2 * CENTER_CROP_WIDTH_RATIO_2X) / (self.hcam.get_Size()[1] / scale * CENTER_CROP_WIDTH_RATIO_2X) + scan_center_y
            scan_coordinates_10x.append([round(start_pos_x), round(start_pos_y), num_windows_x, num_windows_y])

            for i in range(num_windows_x):
                for j in range(num_windows_y):
                    wx = start_x + i * window_w
                    wy = start_y + j * window_h

                    cv2.rectangle(self.true_map, (wx, wy), (wx + window_w, wy + window_h), (0, 255, 0), 5, cv2.LINE_AA)

            cv2.circle(self.true_map, (chip_center_x, chip_center_y), 8, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.circle(self.true_map, (int(self.true_map.shape[1] / 2), int(self.true_map.shape[0] / 2)), 8, (255, 0, 0), -1, cv2.LINE_AA)

        self.root.after(0, self.display_map)

        return scan_coordinates_10x

    def generate_rect_coords(self, x, y):

        rect_coords = []
        total_frames = x * y

        for i in range(x):

            if i % 2 == 0:
                y_range = range(y)
            else:
                y_range = range(y - 1, -1, -1)

            for j in y_range:
                rect_coords.append((i - x // 2, j - y // 2))

        return rect_coords, total_frames

    def generate_spiral_coords(self, length):

        spiral_coords = []
        total_frames = length ** 2

        dx, dy = 0, 0
        step = 1
        direction = 0

        while len(spiral_coords) < total_frames:
            for _ in range(2):
                for _ in range(step):
                    if len(spiral_coords) >= total_frames:
                        break
                    spiral_coords.append((dx, dy))
                    if direction == 0:
                        dx += 1
                    elif direction == 1:
                        dy += 1
                    elif direction == 2:
                        dx -= 1
                    else:
                        dy -= 1
                direction = (direction + 1) % 4
            step += 1

        return spiral_coords, total_frames

    def update_scan_status(self, scan_type=None, stage=None, progress=None, stage_elapsed_time=None, total_elapsed_time=None):
        if scan_type is not None:
            self.scan_info_type_label.config(text=f"Scan: {scan_type}")
    
        if stage is not None:
            self.scan_info_stage_label.config(text=f"Stage: {stage}")

        if progress is not None:
            self.scan_info_progress_label.config(text=f"Stage Progress: {progress}")

        if stage_elapsed_time is not None:
            self.scan_info_stage_time_label.config(text=f"Stage Time Elapsed: {stage_elapsed_time}")

        if total_elapsed_time is not None:
            self.scan_info_total_time_label.config(text=f"Total Time Elapsed: {total_elapsed_time}")

        self.display_map()
        self.scan_info_panel.update()

    # ------------- Flake Detection -------------

    def run_10x_flake_detection(self, image_queue=None):
        while True:
            img_path = image_queue.get()
            if img_path is None:
                break
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            scanned_img, _, save = fi.identify_flakes_flake_model(img)
            out_path = img_path.parent.parent / "Processed" / img_path.name
            self.frame_processor.save_image(cv2.cvtColor(scanned_img, cv2.COLOR_RGB2BGR), save_dir=out_path.parent, filename=out_path.name)
            if save:
                chip_folder = img_path.parent.parent
                scan_root = chip_folder.parent.parent.parent        
                flakes_dir = scan_root / "Flakes Found" / chip_folder.name
                flakes_dir.mkdir(parents=True, exist_ok=True)       
                self.frame_processor.save_image(cv2.cvtColor(scanned_img, cv2.COLOR_RGB2BGR), save_dir=flakes_dir, filename=img_path.name)
            image_queue.task_done()

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
                crop_w = int(w * CENTER_CROP_WIDTH_RATIO_2X)
                crop_h = int(h * CENTER_CROP_HEIGHT_RATIO_2X)
            elif self.magnification == "10x":
                crop_w = int(w * CENTER_CROP_WIDTH_RATIO_10X)
                crop_h = int(h * CENTER_CROP_HEIGHT_RATIO_10X)
            elif self.magnification == "20x":
                crop_w = int(w * CENTER_CROP_WIDTH_RATIO_20X)
                crop_h = int(h * CENTER_CROP_HEIGHT_RATIO_20X)
            elif self.magnification == "100x":
                crop_w = int(w * CENTER_CROP_WIDTH_RATIO_100X)
                crop_h = int(h * CENTER_CROP_HEIGHT_RATIO_100X)
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

    # ------------- Setting and Saving Functions -------------

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
        self.stage.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
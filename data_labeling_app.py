import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import re
from pathlib import Path

import time
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import math

from tensorflow.keras.models import load_model
from tkinter import messagebox

import cv2
import csv

home_dir = Path(os.path.dirname(os.path.abspath(__file__)))
flake_finder_dir = home_dir / "Flake Recognition"
sys.path.insert(0, str(flake_finder_dir))

import flake_finder

CROP_SCALE = 1.2

class DataLabelingApp:
    def __init__(self, root):

        self.root = root
        self.root.title("Data Labeling App")
        self.root.geometry("1125x750")
        
        self.container = tk.Frame(root)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.container, bg="white", width=200)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        self.left_panel.pack_propagate(False)

        self.right_panel = tk.Frame(self.container)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_placeholder = tk.Label(
            self.right_panel,
            text="Select Image",
            fg="gray",
            font=("TkDefaultFont", 12)
        )
        self.right_placeholder.pack(expand=True)

        self.image_label = tk.Label(self.right_panel, bg="black")

        self.init_button_display()

        self.save_folder_path = home_dir / "Labeled Data"
        self.save_folder_path.mkdir(parents=True, exist_ok=True)
        self.images = []

        self.display_image = None
        self.current_image = None
        self.image_index = -1
        self.contour_index = 0

        self.len_contours = 0
        self.contours = []

        self.create_menu()

        self.right_panel.bind("<Configure>", self.on_resize)

    def init_button_display(self):

        title_label = tk.Label(
            self.left_panel,
            text="Classify Flakes",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title_label.place(relx=0.5, y=15, anchor="n")

        style = ttk.Style()
        style.configure("Normal.TButton", font="TkDefaultFont")
        style.configure("Normal.TButton", background="white")
        style.configure("Normal.TButton", relief="flat")
        style.configure("Normal.TButton", padding=(20, 10))

        labels = ["Thick Flake", "Med Flake", "Thin Flake", "Mixed Flake", "Glue", "Dust"]

        self.buttons = {}

        self.image_name_var = tk.StringVar()
        self.image_name_var.set("No Image Selected")

        self.image_name_label = tk.Label(
            self.left_panel,
            textvariable=self.image_name_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        self.image_name_label.place(relx=0.5, y=50, anchor="n")

        self.contour_count_var = tk.StringVar()
        self.contour_count_var.set("Contours Found: None")

        self.contour_count_label = tk.Label(
            self.left_panel,
            textvariable=self.contour_count_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        self.contour_count_label.place(relx=0.5, y=80, anchor="n")

        self.contour_index_var = tk.StringVar()
        self.contour_index_var.set("Contour Index: None")

        self.contour_index_label = tk.Label(
            self.left_panel,
            textvariable=self.contour_index_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        self.contour_index_label.place(relx=0.5, y=110, anchor="n")

        style = ttk.Style()
        style.configure("Image.TButton", font="TkDefaultFont")
        style.configure("Image.TButton", background="white")
        style.configure("Image.TButton", relief="flat")

        self.next_image_button = ttk.Button(
            self.left_panel,
            text="Next Image",
            style="Image.TButton",
            command=self.next_image
        )
        self.next_image_button.place(relx=0.5, x=40, y=150, anchor="n")
        self.next_image_button.state(["disabled"])

        self.previous_image_button = ttk.Button(
            self.left_panel,
            text="Prev Image",
            style="Image.TButton",
            command=self.previous_image
        )
        self.previous_image_button.place(relx=0.5, x=-40, y=150, anchor="n")
        self.previous_image_button.state(["disabled"])

        self.root.bind("<Up>", self.previous_image)
        self.root.bind("<Down>", self.next_image)

        y_start = 200
        y_step = 60

        for i, text in enumerate(labels):
            key = str(i + 1)

            btn = ttk.Button(
                self.left_panel,
                text=f"{key}: {text}", 
                style="Normal.TButton"
            )

            def make_callback(t=text, b=btn):
                def callback(event=None):
                    b.state(["pressed"])
                    self.left_panel.after(100, lambda: b.state(["!pressed"]))
                    self.save_contour(t.lower().replace(" ", "_"), output=True)
                return callback

            cb = make_callback()

            btn.config(command=cb)

            self.root.bind(f"<KeyPress-{key}>", cb)

            btn.place(relx=0.5, y=y_start + i * y_step, anchor="n")
            self.buttons[text] = btn

        style = ttk.Style()
        style.configure("Arrow.TButton", font=("TkDefaultFont", 15), padding=5)
        style.configure("Arrow.TButton", background="white")
        style.configure("Arrow.TButton", relief="flat")

        self.contour_button_panel = tk.Frame(self.left_panel, bg="white", width=80, height=45)
        self.contour_button_panel.place(relx=0.5, x=0, y=565, anchor="n")
        self.contour_button_panel.pack_propagate(False)

        image_controls = tk.Frame(self.contour_button_panel, bg="white")
        image_controls.pack(expand=True, fill="both")

        self.btn_next_contour = ttk.Button(image_controls, text="▸", style="Arrow.TButton")
        self.btn_previous_contour = ttk.Button(image_controls, text="◂", style="Arrow.TButton")

        self.btn_next_contour.bind("<ButtonPress-1>", self.next_contour)
        self.btn_previous_contour.bind("<ButtonPress-1>", self.previous_contour)

        self.root.bind("<Left>", self.previous_contour)
        self.root.bind("<Right>", self.next_contour)

        image_controls.rowconfigure(0, weight=1)
        image_controls.columnconfigure(0, weight=1)
        image_controls.columnconfigure(1, weight=1)

        self.btn_next_contour.grid(row=0, column=1, sticky="nsew")
        self.btn_previous_contour.grid(row=0, column=0, sticky="nsew")

        self.btn_next_contour.state(["disabled"])
        self.btn_previous_contour.state(["disabled"])

    def create_menu(self):
        self.menubar = tk.Menu(self.root)

        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Open Folder", command=self.open_folder)
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=file_menu)

        self.menubar.add_command(label="Save Folder")

        self.root.config(menu=self.menubar) 

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )

        if not file_path:
            return

        self.image_name_var.set(f"{os.path.basename(file_path)}")
        self.images.append(file_path)
        self.image_index = len(self.images) - 1
        self.current_image = Image.open(self.images[self.image_index])
        self.find_contours(self.current_image)
        self.update_image_display(self.current_image)
        self.contour_index = -1

        if len(self.images) > 1:
            self.next_image_button.state(["!disabled"])
            self.previous_image_button.state(["!disabled"])

    def open_folder(self):

        folder_path = filedialog.askdirectory()

        if not folder_path:
            return

        valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        file_list = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(valid_ext)
        ]

        file_list = sorted(
            [Path(f) for f in file_list],
            key=self.image_sort_key
        )

        if not file_list:
            return

        start_index = len(self.images)

        for file_path in file_list:
            try:
                self.images.append(file_path)

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        self.image_name_var.set(f"{os.path.basename(file_path)}")
        self.current_image = Image.open(self.images[start_index])
        self.find_contours(self.current_image)
        self.update_image_display(self.current_image)
        self.image_index = start_index
        self.contour_index = -1

        self.next_image_button.state(["!disabled"])
        self.previous_image_button.state(["!disabled"])

    def image_sort_key(self, p):
        match = re.search(r'_(\d+)\.', p.name)
        return int(match.group(1)) if match else -1

    def update_image_display(self, img):
        if img is None:
            return

        if self.right_placeholder.winfo_ismapped():
            self.right_placeholder.pack_forget()
            self.image_label.pack(expand=True, fill="both")

        self.right_panel.update_idletasks()

        w = self.right_panel.winfo_width()
        h = self.right_panel.winfo_height()

        if w < 10 or h < 10:
            w, h = 600, 600

        img_copy = img.copy()
        img_copy.thumbnail((w, h), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(img_copy)

        self.image_label.config(image=self.tk_image)

    def find_contours(self, img):
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        self.masked_background, self.contours = flake_finder.find_flakes(img_bgr, display=False)

        self.len_contours = len(self.contours)
        self.contour_count_var.set(f"Contours Found: {self.len_contours}")

        self.contour_index = 0
        self.contour_index_var.set(f"Contour Index: None/{self.len_contours}")

        self.btn_next_contour.state(["!disabled"])
        self.btn_previous_contour.state(["!disabled"])

        return self.contours

    def draw_contour(self, contour):

        img = np.array(self.current_image).copy()

        contour = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)

        cv2.drawContours(img, [contour], -1, (255, 255, 255), 2)
        
        x, y, x2, y2 = self.find_contour_bounded_box(img, contour)

        cv2.rectangle(img, (x, y), (x2, y2), color=(255, 255, 255), thickness=2)

        pil_img = Image.fromarray(img)

        self.display_image = pil_img
        self.update_image_display(self.display_image)

    def find_contour_bounded_box(self, img, contour):
        x, y, w, h = cv2.boundingRect(contour)

        scale = 1.2
        cx, cy = x + w / 2, y + h / 2

        new_w, new_h = w * scale, h * scale
        new_x = int(cx - new_w / 2)
        new_y = int(cy - new_h / 2)
        new_x2 = int(cx + new_w / 2)
        new_y2 = int(cy + new_h / 2)

        h_img, w_img = img.shape[:2]
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_x2 = min(w_img, new_x2)
        new_y2 = min(h_img, new_y2)

        return new_x, new_y, new_x2, new_y2

    def next_contour(self, event=None):
        if self.len_contours == 0:
            return
        self.contour_index = (self.contour_index + 1) % self.len_contours
        self.draw_contour(self.contours[self.contour_index])
        self.contour_index_var.set(f"Contour Index: {self.contour_index + 1}/{self.len_contours}")

    def previous_contour(self, event=None):
        if self.len_contours == 0:
            return
        self.contour_index = (self.contour_index - 1) % self.len_contours
        self.draw_contour(self.contours[self.contour_index])
        self.contour_index_var.set(f"Contour Index: {self.contour_index + 1}/{self.len_contours}")

    def next_image(self, event=None):
        self.image_index = (self.image_index + 1) % len(self.images)
        
        self.current_image = Image.open(self.images[self.image_index])        
        self.find_contours(self.current_image)
        self.update_image_display(self.current_image)
        
        self.contour_index = -1
        self.image_name_var.set(f"{os.path.basename(self.images[self.image_index])}")

    def previous_image(self, event=None):
        self.image_index = (self.image_index - 1) % len(self.images)
        
        self.current_image = Image.open(self.images[self.image_index])
        self.find_contours(self.current_image)
        self.update_image_display(self.current_image)
        
        self.contour_index = -1
        self.image_name_var.set(f"{os.path.basename(self.images[self.image_index])}")

    def save_contour(self, label, output=False):

        if self.contour_index < 0 or self.contour_index >= len(self.contours):
            return
        
        self.save_folder_path.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"
        filepath = self.save_folder_path / f"{label}_{timestamp}.png"

        img = np.array(self.current_image).copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, x2, y2 = self.find_contour_bounded_box(img, self.contours[self.contour_index])
        cropped = img[y:y2, x:x2]

        cv2.imwrite(str(filepath), cropped)

        if output:
            print(f"Image saved to {filepath}")

        self.next_contour()

    def choose_save_folder(self):

        self.save_folder_path = filedialog.askdirectory()

        if not self.save_folder_path:
            return

    def on_resize(self, event):
        if self.display_image is not None:
            self.update_image_display(self.display_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataLabelingApp(root)
    root.mainloop()
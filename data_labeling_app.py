import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
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

class DataLabelingApp:
    def __init__(self, root):

        self.root = root
        self.root.title("Data Labeling App")
        self.root.geometry("900x600")
        
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
        self.right_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        self.init_button_display()

        self.save_folder_path = home_dir / "Labeled Data"
        self.save_folder_path.mkdir(parents=True, exist_ok=True)
        self.images = []

        self.current_image = None
        self.image_index = -1
        self.contour_index = 0

        self.create_menu()

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

        labels = ["Labels", "Thick Flake", "Med Flake", "Thin Flake", "Mixed Flake", "Glue", "Dust"]

        self.buttons = {}

        y_start = 50
        y_step = 60

        for i, text in enumerate(labels):
            btn = ttk.Button(
                self.left_panel,
                text=text,
                style="Normal.TButton",
            )

            btn.place(relx=0.5, y=y_start + i * y_step, anchor="n")
            self.buttons[text] = btn

        style = ttk.Style()
        style.configure("Arrow.TButton", font=("TkDefaultFont", 15), padding=5)
        style.configure("Arrow.TButton", background="white")
        style.configure("Arrow.TButton", relief="flat")

        self.image_button_panel = tk.Frame(self.left_panel, bg="white", width=80, height=45)
        self.image_button_panel.place(relx=0.5, x=0, y=475, anchor="n")
        self.image_button_panel.pack_propagate(False)

        image_controls = tk.Frame(self.image_button_panel, bg="white")
        image_controls.pack(expand=True, fill="both")

        self.btn_next = ttk.Button(image_controls, text="▸", style="Arrow.TButton")
        self.btn_previous = ttk.Button(image_controls, text="◂", style="Arrow.TButton")

        self.btn_next.bind("<ButtonPress-1>", self.next_contour)
        self.btn_previous.bind("<ButtonPress-1>", self.previous_contour)

        self.root.bind("<Left>", self.previous_contour)
        self.root.bind("<Right>", self.next_contour)

        image_controls.rowconfigure(0, weight=1)
        image_controls.columnconfigure(0, weight=1)
        image_controls.columnconfigure(1, weight=1)

        self.btn_next.grid(row=0, column=1, sticky="nsew")
        self.btn_previous.grid(row=0, column=0, sticky="nsew")

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

        image = Image.open(file_path)

        self.images.append(file_path)

        self.current_image = image
        self.find_contours(self.current_image)
        self.update_image_display(self.current_image)
        self.image_index += 1
        self.contour_index = -1

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

        if not file_list:
            return

        self.images = []

        for file_path in file_list:
            try:
                image = Image.open(file_path)

                self.images.append(file_path)

            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        self.current_image = Image.open(self.images[0])
        self.find_contours(self.current_image)
        self.update_image_display(self.current_image)
        self.image_index += 1
        self.contour_index = -1

    def update_image_display(self, img):

        if img is None:
            return

        self.right_panel.update_idletasks()

        w = self.right_panel.winfo_width()
        h = self.right_panel.winfo_height()

        if w < 10 or h < 10:
            w, h = 600, 600

        img.thumbnail((w, h), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(img)

        for widget in self.right_panel.winfo_children():
            widget.destroy()

        label = tk.Label(self.right_panel, image=self.tk_image, bg="black")
        label.pack(expand=True)

    def find_contours(self, img):
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        self.masked_background, self.contours = flake_finder.find_flakes(img_bgr, display=False)

        self.len_contours = len(self.contours)

        return self.contours

    def draw_contour(self, contour):

        img = np.array(self.current_image).copy()

        contour = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)

        cv2.drawContours(img, [contour], -1, (255, 255, 255), 2)
        
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

        cv2.rectangle(img, (new_x, new_y), (new_x2, new_y2), color=(255, 255, 255), thickness=2)

        pil_img = Image.fromarray(img)

        self.update_image_display(pil_img)

    def next_contour(self, event=None):
        self.contour_index = (self.contour_index + 1) % self.len_contours
        self.draw_contour(self.contours[self.contour_index])

    def previous_contour(self, event=None):
        self.contour_index = (self.contour_index - 1) % self.len_contours
        self.draw_contour(self.contours[self.contour_index])

    def save_contour(self, image, label, output=False):

        self.save_folder_path.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"
        filepath = self.save_folder_path / f"{timestamp}_{label}.png"

        cv2.imwrite(str(filepath), image)

        if output:
            print(f"Image saved to {filepath}")

    def choose_save_folder(self):

        self.save_folder_path = filedialog.askdirectory()

        if not self.save_folder_path:
            return

if __name__ == "__main__":
    root = tk.Tk()
    app = DataLabelingApp(root)
    root.mainloop()
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model
from tkinter import messagebox

import os
import csv

class ModelTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Reader App")

        self.paned = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.paned)
        self.right_panel = tk.Frame(self.paned)

        self.paned.add(self.left_panel, stretch="always")
        self.paned.add(self.right_panel, stretch="always")

        self.paned.paneconfigure(self.left_panel, minsize=300)
        self.paned.paneconfigure(self.right_panel, minsize=250)

        self.root.update_idletasks()
        self.paned.sash_place(0, self.paned.winfo_width() // 2, 0)

        self.canvas = tk.Canvas(self.left_panel, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.left_placeholder = tk.Label(
            self.left_panel,
            text="Select Image",
            fg="gray",
            font=("TkDefaultFont", 12)
        )
        self.left_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        self.right_top = tk.Frame(self.right_panel, bg="white")
        self.right_bottom = tk.Frame(self.right_panel, bg="white", height=100)

        self.right_top.pack(fill=tk.BOTH, expand=True)
        self.right_bottom.pack(fill=tk.X, side=tk.BOTTOM, pady=(2, 2))

        self.init_model_display()
        self.init_color_display()

        self.placeholder = tk.Label(
            self.right_top,
            text="Select picker",
            bg="white",
            fg="gray",
            font=("TkDefaultFont", 12)
        )

        self.file_path = None
        self.image = None
        self.tk_image = None
        self.prev_image_dim = None
        self.current_line = None

        self.path = []
        self.samples = []

        self.avg_rect = None
        self.avg_start = None
        self.avg_end = None

        self.current_picker = None
        self.currently_drawing = False

        self.model = None

        self.create_menu()

        self.canvas.bind("<Button-1>", lambda e: self.on_mouse_down_avg_picker(e, target="background"))
        self.canvas.bind("<B1-Motion>", lambda e: self.on_mouse_drag_avg_picker(e, button="left"))
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up_avg_picker)

        self.canvas.bind("<Button-3>", lambda e: self.on_mouse_down_avg_picker(e, target="compare"))
        self.canvas.bind("<B3-Motion>", lambda e: self.on_mouse_drag_avg_picker(e, button="right"))
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up_avg_picker)

        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def init_color_display(self):
        self.color_display_frame = tk.Frame(self.right_top, bg="white")
        self.color_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.color_display_frame.rowconfigure(0, weight=1)
        self.color_display_frame.rowconfigure(1, weight=1)
        self.color_display_frame.columnconfigure(0, weight=1)

        top_frame = tk.Frame(self.color_display_frame, bg="white")
        top_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tk.Label(top_frame, text="Compared Color", bg="white", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.left_square = tk.Label(top_frame, bg="white", relief=tk.SUNKEN)
        self.left_square.pack(fill=tk.BOTH, expand=True, pady=(5, 5))

        self.left_label = tk.Label(
            top_frame,
            text="R: -  G: -  B: -",
            bg="white",
            font=("TkDefaultFont", 11)
        )
        self.left_label.pack(anchor="w")

        bottom_frame = tk.Frame(self.color_display_frame, bg="white")
        bottom_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        tk.Label(bottom_frame, text="Background Color", bg="white", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.right_square = tk.Label(bottom_frame, bg="white", relief=tk.SUNKEN)
        self.right_square.pack(fill=tk.BOTH, expand=True, pady=(5, 5))

        self.right_label = tk.Label(bottom_frame, text="R: -  G: -  B: -", bg="white", font=("TkDefaultFont", 11))
        self.right_label.pack(anchor="w")

    def init_model_display(self):
        style = ttk.Style()
        style.configure("Custom.TButton", background="white")
        style.configure("Custom.TButton", relief="flat")

        self.run_model_button = ttk.Button(
            self.right_bottom,
            text="Run Model",
            command=self.run_model,
            style="Custom.TButton"
        )
        self.run_model_button.pack(pady=10)

        self.model_output_label = tk.Label(
            self.right_bottom,
            text="Model output will appear here.",
            bg="white",
            anchor="w",
            justify="left"
        )
        self.model_output_label.pack(fill=tk.X, padx=10)

    def create_menu(self):
        self.menubar = tk.Menu(self.root)

        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=file_menu)

        picker_menu = tk.Menu(self.menubar, tearoff=0)
        picker_menu.add_command(label="Average Region", command=lambda: self.set_mode("r"))
        picker_menu.add_command(label="Point", command=lambda: self.set_mode("p"))
        self.menubar.add_cascade(label="Picker", menu=picker_menu)

        model_menu = tk.Menu(self.menubar, tearoff=0)
        model_menu.add_command(label="Load Keras Model", command=self.load_model)
        self.menubar.add_cascade(label="Model", menu=model_menu)

        self.root.config(menu=self.menubar) 

    def set_mode(self, picker):
        if picker == self.current_picker: return

        self.current_picker = picker

        if self.avg_rect:
            self.canvas.delete(self.avg_rect)

        if picker == "r":
            self.color_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        if picker == "p":
            self.color_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def set_color_display(self, r, g, b, target):
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        if target == "background":
            self.right_square.config(bg=hex_color)
            self.right_label.config(text=f"R: {r}  G: {g}  B: {b}")
        elif target == "compare":
            self.left_square.config(bg=hex_color)
            self.left_label.config(text=f"R: {r}  G: {g}  B: {b}")

    def open_image(self, file_path=None):
        if file_path is None:
            self.file_path = filedialog.askopenfilename(
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
            )
            if not self.file_path:
                return
        else:
            self.file_path = file_path    

        try:
            self.image = Image.open(self.file_path).convert("RGB")
        except FileNotFoundError:
             tk.messagebox.showinfo("Error", f"File Not Found!")
             return

        if hasattr(self, "left_placeholder"):
            self.left_placeholder.place_forget()

        canvas_width = self.canvas.winfo_width() or self.image.width
        canvas_height = self.canvas.winfo_height() or self.image.height
        self.display_image = self.image.copy()
        self.display_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def on_canvas_resize(self, event):

        if self.image is not None:
            self.prev_image_dim = (self.display_image.width, self.display_image.height)

            self.display_image = self.image.copy()
            self.display_image.thumbnail((event.width, event.height), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(self.display_image)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

            scale_x = self.display_image.width / self.prev_image_dim[0]
            scale_y = self.display_image.height / self.prev_image_dim[1]

    def on_canvas_left_click(self, event):
        if self.current_picker == "p":
            self.pick_point(event, target="background")
        elif self.current_picker == "r":
            self.on_mouse_down_avg_picker(event, target="background")

    def on_canvas_right_click(self, event):
        if self.current_picker == "p":
            self.pick_point(event, target="compare")
        elif self.current_picker == "r":
            self.on_mouse_down_avg_picker(event, target="compare")

    def on_mouse_drag_avg_picker(self, event, button="left"):
        if self.image is None or not self.avg_start:
            return

        if self.avg_rect:
            self.canvas.delete(self.avg_rect)

        self.avg_end = (event.x, event.y)

        outline_color = "red" if button == "left" else "yellow"

        self.avg_rect = self.canvas.create_rectangle(
            self.avg_start[0], self.avg_start[1],
            self.avg_end[0], self.avg_end[1],
            outline=outline_color,
            width=2
        )

    def on_mouse_up_avg_picker(self, event):
        if self.image is None or not self.avg_start:
            return

        scale_x = self.display_image.width / self.image.width
        scale_y = self.display_image.height / self.image.height

        x0 = int(self.avg_start[0] / scale_x)
        y0 = int(self.avg_start[1] / scale_y)
        x1 = int(event.x / scale_x)
        y1 = int(event.y / scale_y)

        x_start, x_end = sorted([x0, x1])
        y_start, y_end = sorted([y0, y1])

        region = np.array(self.image.crop((x_start, y_start, x_end + 1, y_end + 1)))
        if region.size == 0:
            return

        r = int(region[:, :, 0].mean())
        g = int(region[:, :, 1].mean())
        b = int(region[:, :, 2].mean())

        self.set_color_display(r, g, b, self.avg_target)

        self.avg_start = None

    def on_mouse_down_avg_picker(self, event, target):
        self.avg_target = target
        self.avg_start = (event.x, event.y)

        if self.avg_rect:
            self.canvas.delete(self.avg_rect)
            self.avg_rect = None

    def pick_point(self, event, target):
        if self.image is None:
            return

        scale_x = self.display_image.width / self.image.width
        scale_y = self.display_image.height / self.image.height

        img_x = int(event.x / scale_x)
        img_y = int(event.y / scale_y)

        r, g, b = self.image.getpixel((img_x, img_y))
        self.set_color_display(r, g, b, target)

    def load_model(self):
        file_path = tk.filedialog.askopenfilename(
            filetypes=[("Models", "*.keras *.h5")]
        )
        if not file_path:
            return

        try:
            self.model = load_model(file_path)
            messagebox.showinfo("Success", f"Model loaded from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.model = None

    def run_model(self):
        if self.model is None:
            self.model_output_label.config(text="No model loaded.")
            return
    
        try:
            # Compared color
            comp_bg = self.left_square.cget("bg")  # hex color, e.g. "#ff00aa"
            comp_r = int(comp_bg[1:3], 16)
            comp_g = int(comp_bg[3:5], 16)
            comp_b = int(comp_bg[5:7], 16)
    
            # Background color
            back_bg = self.right_square.cget("bg")
            back_r = int(back_bg[1:3], 16)
            back_g = int(back_bg[3:5], 16)
            back_b = int(back_bg[5:7], 16)
        except Exception as e:
            self.model_output_label.config(text=f"Failed to read colors:\n{e}")
            return
    
        # Create input array
        input_array = np.array([[comp_r, comp_g, comp_b, back_r, back_g, back_b]], dtype=np.float32) / 255.0
    
        try:
            result = self.model.predict(input_array)
            predicted_class = np.argmax(result, axis=1)[0]
            class_names = ["Background", "Glue", "Thin Flake", "Med Flake", "Thick Flake"]
            self.model_output_label.config(
                text=f"Model output: {predicted_class} ({class_names[predicted_class]})"
            )
        except Exception as e:
            self.model_output_label.config(text=f"Model prediction failed:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelTesterApp(root)
    root.mainloop()
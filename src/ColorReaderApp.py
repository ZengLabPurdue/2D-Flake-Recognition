import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import math

import os
import csv

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class ColorReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Reader App - Straight Line Mode")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame, cursor="cross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize = (4,4), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)

        self.channels = ["intensity","red", "green", "blue"]

        self.channel_vars = {
            "intensity": tk.BooleanVar(value=True), 
            "red": tk.BooleanVar(value=True),
            "green": tk.BooleanVar(value=True),
            "blue": tk.BooleanVar(value=True),
        }

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.plot_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.file_path = None
        self.image = None
        self.tk_image = None
        self.prev_image_dim = None
        self.current_line = None
        
        self.path = []
        self.samples = []
        self.curved_mode = False

        self.create_menu()
        self.update_plot()

        self.currently_drawing = False

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Image", command=self.open_image)
        filemenu.add_command(label="Import", command=self.import_file)
        filemenu.add_command(label="Export", command=self.export_file)
        menubar.add_cascade(label="File", menu=filemenu)

        channelmenu = tk.Menu(menubar, tearoff=0)

        for channel, var in self.channel_vars.items():
            channelmenu.add_checkbutton(
                label=channel.capitalize(),
                variable=var,
                command=self.update_channels
            )

        menubar.add_cascade(label="Channel", menu=channelmenu)

        linestylemenu =  tk.Menu(menubar, tearoff=0)
        linestylemenu.add_command(label="Striaght", command=lambda: self.set_linestyle("Straight"))
        linestylemenu.add_command(label="Curved", command=lambda: self.set_linestyle("Curved"))
        menubar.add_cascade(label="Line Style", menu=linestylemenu)

        self.root.config(menu=menubar)  

    def update_channels(self):
        self.channels = [
            ch for ch, var in self.channel_vars.items()
            if var.get()
        ]
        print(self.channels)
        self.update_plot()

    def export_file(self):
        if len(self.samples) == 0:
            tk.messagebox.showwarning("Export Data", "No samples to export!")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return

        file_path = os.path.abspath(self.file_path)

        scale_x = self.display_image.width / self.image.width
        scale_y = self.display_image.height / self.image.height

        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["File Path", file_path])

            path_str = ";".join(f"{int(x / scale_x)},{int(y / scale_y)}" for x, y in self.path)
            writer.writerow(["Line Path", path_str])
            writer.writerow([])
            
            writer.writerow(["Intensity", "Red", "Green", "Blue"])
            for intensity, r, g, b in self.samples:
                    writer.writerow([
                        f"{intensity:.2f}",
                        f"{r:.2f}",
                        f"{g:.2f}",
                        f"{b:.2f}"
                    ])

        tk.messagebox.showinfo("Export", f"Curve exported to {filepath}")

    def import_file(self):
        path = filedialog.askopenfilename(
                   filetypes=[("CSV files", "*.csv")],
                   title="Select CSV file"
               )
        if not path:
            return

        with open(path, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

            for i, row in enumerate(rows):
                if row and row[0] == "File Path":
                    file_path = row[1]
                elif row and row[0] == "Line Path":
                    path_str = row[1]
                    if path_str:
                        coords = path_str.split(";")
                        self.path = [tuple(map(float, c.split(","))) for c in coords]
                elif row and row[0] == "Intensity":
                    sample_start = i + 1
                    break

            for row in rows[sample_start:]:
                if len(row) < 4:
                    continue
                intensity, r, g, b = map(float, row[:4])
                self.samples.append((intensity, r, g, b))

        self.open_image(file_path=file_path)

        scale_x = self.display_image.width / self.image.width
        scale_y = self.display_image.height / self.image.height

        if self.path:
            self.path = [(x * scale_x, y * scale_y) for x, y in self.path]
            flat = []
            flat = [v for pt in self.path for v in pt]
            self.current_line = self.canvas.create_line(
                *flat, fill="red", width=2, smooth=self.curved_mode
            )

        flat = [v for pt in self.path for v in pt]
        self.canvas.coords(self.current_line, *flat)

        self.sample_path()
        self.update_plot()

        tk.messagebox.showinfo("Import", f"Imported!")

    def set_linestyle(self, mode):
        if mode == "Curved": 
            self.curved_mode = True 
            self.root.title("Color Reader App - Curved Line Mode")
        else: 
            self.curved_mode = False 
            self.root.title("Color Reader App - Straight Line Mode")

    def open_image(self, file_path = None):
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

        canvas_width = self.canvas.winfo_width() or self.image.width
        canvas_height = self.canvas.winfo_height() or self.image.height
        self.display_image = self.image.copy()
        self.display_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.update_plot()
        self.plot_canvas.draw()

        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def on_canvas_resize(self, event):
        if self.image is None:
            return
        
        self.prev_image_dim = (self.display_image.width, self.display_image.height)

        self.display_image = self.image.copy()
        self.display_image.thumbnail((event.width, event.height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        scale_x = self.display_image.width / self.prev_image_dim[0]
        scale_y = self.display_image.height / self.prev_image_dim[1]

        if self.current_line and self.path:
            self.path = [(x * scale_x, y * scale_y) for x, y in self.path]
            flat = []
            flat = [v for pt in self.path for v in pt]
            self.current_line = self.canvas.create_line(
                *flat, fill="red", width=2, smooth=self.curved_mode
            )

        self.update_plot()
    
    def on_mouse_down(self, event):

        if self.image is None: return

        self.currently_drawing = True

        if self.current_line:
            self.canvas.delete(self.current_line)

        self.path = [(event.x, event.y)]
        self.current_line = self.canvas.create_line(
            event.x, event.y, event.x, event.y,
            fill="red", width=2, smooth=self.curved_mode
        )

    def on_mouse_drag(self, event):
        if self.image is None: return
        if self.curved_mode:
            self.path.append((event.x, event.y))
        else:
            self.path = [self.path[0], (event.x, event.y)]

        flat = [v for pt in self.path for v in pt]
        self.canvas.coords(self.current_line, *flat)

        self.update_plot()

    def on_mouse_up(self, event):
        if self.image is None: return
        self.currently_drawing = False
        self.update_plot()

    def sample_path(self):
        if self.image is None or len(self.path) < 2:
            return np.array([])

        pixels = np.array(self.image)
        self.samples = []

        scale_x = self.display_image.width / self.image.width
        scale_y = self.display_image.height / self.image.height

        for i in range(len(self.path) - 1):
            x1d, y1d = self.path[i]
            x2d, y2d = self.path[i + 1]

            x1 = int(x1d / scale_x)
            y1 = int(y1d / scale_y)
            x2 = int(x2d / scale_x)
            y2 = int(y2d / scale_y)

            dist = int(math.dist((x1, y1), (x2, y2)))
            if dist == 0:
                continue

            t = np.linspace(0, 1, dist)
            xs = (x1 + t * (x2 - x1)).astype(int)
            ys = (y1 + t * (y2 - y1)).astype(int)

            mask = (
                (xs >= 0) & (xs < pixels.shape[1]) &
                (ys >= 0) & (ys < pixels.shape[0])
            )

            rgb = pixels[ys[mask], xs[mask]]
            for r, g, b in rgb:
                intensity = (int(r) + int(g) + int(b)) / 3
                self.samples.append((intensity, r, g, b))

    def update_plot(self):
        self.sample_path()
        self.ax.clear()

        if len(self.samples) > 0:
            intensity, r, g, b = zip(*self.samples)
            x = np.arange(len(self.samples))

            if "intensity" in self.channels:
                self.ax.plot(x, intensity, color="black", label="Intensity")
            if "red" in self.channels:
                self.ax.plot(x, r, color="red", label="Red")
            if "green" in self.channels:
                self.ax.plot(x, g, color="green", label="Green")
            if "blue" in self.channels:
                self.ax.plot(x, b, color="blue", label="Blue")

            self.ax.legend()

        self.ax.set_title("Values Along Line")
        self.ax.set_xlabel("Distance (pixels)")
        self.ax.set_ylabel("Value")

        self.figure.tight_layout()
        self.plot_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorReaderApp(root)
    root.mainloop()
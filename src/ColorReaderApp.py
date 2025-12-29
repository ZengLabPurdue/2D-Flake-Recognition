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
        self.root.title("Color Reader App")

        self.paned = tk.PanedWindow(
            root,
            orient=tk.HORIZONTAL,
        )
        self.paned.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.paned)
        self.right_panel = tk.Frame(self.paned, bg="white")

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
            font=("Arial", 12)
        )
        self.left_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        self.right_placeholder = tk.Label(
            self.right_panel,
            text="Select Tool",
            bg="white",
            fg="gray",
            font=("Arial", 12)
        )
        self.right_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        self.figure = Figure(figsize = (4,4), dpi=100, constrained_layout=True)
        self.ax = self.figure.add_subplot(111)

        self.init_color_picker_tool()
        self.color_display_frame.pack_forget()

        self.init_average_color_tool()
        self.average_color_frame.pack_forget()

        self.channels = ["intensity","red", "green", "blue"]

        self.channel_vars = {
            "intensity": tk.BooleanVar(value=True), 
            "red": tk.BooleanVar(value=True),
            "green": tk.BooleanVar(value=True),
            "blue": tk.BooleanVar(value=True),
        }

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=self.right_panel)
        self.plot_widget = self.plot_canvas.get_tk_widget()
        self.plot_widget.pack(fill=tk.BOTH, expand=True)
        self.hide_line_plot()

        self.placeholder = tk.Label(
            self.right_panel,
            text="Select Tool",
            bg="white",
            fg="gray",
            font=("Arial", 12)
        )

        self.file_path = None
        self.image = None
        self.tk_image = None
        self.prev_image_dim = None
        self.current_line = None
        
        self.path = []
        self.samples = []
        self.curved_mode = False

        self.avg_rect = None
        self.avg_start = None

        self.create_menu()
        self.update_line_plot()

        self.current_tool = None
        self.currently_drawing = False

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def init_color_picker_tool(self):
        self.color_display_frame = tk.Frame(self.right_panel, bg="white")
        self.color_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.color_squares = []
        self.color_labels = []

        rows, cols = 3, 2

        for r in range(rows):
            self.color_display_frame.rowconfigure(r, weight=1, uniform="row")
        for c in range(cols):
            self.color_display_frame.columnconfigure(c, weight=1, uniform="col")

        for i in range(6):
            r = i // cols
            c = i % cols

            square_frame = tk.Frame(self.color_display_frame, bg="white")
            square_frame.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")

            square = tk.Label(square_frame, bg="white", relief=tk.SUNKEN)
            square.pack(fill=tk.BOTH, expand=True)

            label = tk.Label(square_frame, text="R: - G: - B: -", font=("Arial", 9), bg="white")
            label.pack(fill=tk.X)

            self.color_squares.append(square)
            self.color_labels.append(label)

        self.next_color_index = 0

    def init_average_color_tool(self):
        self.average_color_frame = tk.Frame(self.right_panel, bg="white")
        self.average_color_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.average_color_frame.rowconfigure(0, weight=1)
        self.average_color_frame.columnconfigure(0, weight=1)

        self.average_color_square = tk.Label(
            self.average_color_frame, bg="white", relief=tk.SUNKEN
        )
        self.average_color_square.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.average_color_label = tk.Label(
            self.average_color_frame, text="R: - G: - B: -", font=("Arial", 12), bg="white"
        )
        self.average_color_label.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.average_color_frame.pack_forget()

    def create_menu(self):
        self.menubar = tk.Menu(self.root)
        
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Open Image", command=self.open_image)
        filemenu.add_command(label="Import", command=self.import_file)
        filemenu.add_command(label="Export", command=self.export_file)
        self.menubar.add_cascade(label="File", menu=filemenu)

        self.channelmenu = tk.Menu(self.menubar, tearoff=0)

        for channel, var in self.channel_vars.items():
            self.channelmenu.add_checkbutton(
                label=channel.capitalize(),
                variable=var,
                command=self.update_channels
            )

        self.linetypemenu = tk.Menu(self.menubar, tearoff=0)
        self.linetypemenu.add_command(label="straight", command=lambda: self.set_linetype("straight"))
        self.linetypemenu.add_command(label="curved", command=lambda: self.set_linetype("curved"))

        self.averagetypemenu = tk.Menu(self.menubar, tearoff=0)
        self.averagetypemenu.add_command(label="Region", command=lambda: self.set_linetype("region"))
        self.averagetypemenu.add_command(label="Line", command=lambda: self.set_linetype("line"))

        toolmenu = tk.Menu(self.menubar, tearoff=0)
        toolmenu.add_command(label="Color Picker Tool", command=lambda: self.set_tool("picker"))
        toolmenu.add_command(label="Line Tool", command=lambda: self.set_tool("line"))
        toolmenu.add_command(label="Average Color Tool", command=lambda: self.set_tool("average"))
        self.menubar.add_cascade(label="Tools", menu=toolmenu)

        self.root.config(menu=self.menubar)  

    def set_tool(self, tool):
        self.current_tool = tool

        self.hide_linetype_menu()
        self.hide_channel_menu()
        self.hide_averagetype_menu()
        
        self.plot_widget.pack_forget()
        self.placeholder.pack_forget()

        self.color_display_frame.pack_forget()
        self.average_color_frame.pack_forget()

        if hasattr(self, "right_placeholder"):
            self.right_placeholder.place_forget()

        if tool == "line":
            self.root.title("Color Reader App - Line Tool")
            self.show_linetype_menu()
            self.show_channel_menu()
            self.plot_widget.pack(fill=tk.BOTH, expand=True)
        elif tool == "average":
            self.root.title("Color Reader App - Average Color Tool")
            self.show_averagetype_menu()
            self.average_color_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        elif tool == "picker":
            self.root.title("Color Reader App - Picker Tool")
            self.color_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def show_linetype_menu(self):
        if not self.linetype_menu_visible():
            self.menubar.add_cascade(label="Line Type", menu=self.linetypemenu)

    def hide_linetype_menu(self):
        for i in range(self.menubar.index("end") + 1):
            if self.menubar.type(i) == "cascade":
                if self.menubar.entrycget(i, "label") == "Line Type":
                    self.menubar.delete(i)
                    break

    def linetype_menu_visible(self):
        for i in range(self.menubar.index("end") + 1):
            if self.menubar.type(i) == "cascade":
                if self.menubar.entrycget(i, "label") == "Line Type":
                    return True
        return False
    
    def show_channel_menu(self):
        if not self.channel_menu_visible():
            self.menubar.add_cascade(label="Channels", menu=self.channelmenu)

    def hide_channel_menu(self):
        for i in range(self.menubar.index("end") + 1):
            if self.menubar.type(i) == "cascade":
                if self.menubar.entrycget(i, "label") == "Channels":
                    self.menubar.delete(i)
                    break

    def channel_menu_visible(self):
        for i in range(self.menubar.index("end") + 1):
            if self.menubar.type(i) == "cascade":
                if self.menubar.entrycget(i, "label") == "Channels":
                    return True
        return False
    
    def show_averagetype_menu(self):
        if not self.averagetype_menu_visible():
            self.menubar.add_cascade(label="Average Type", menu=self.averagetypemenu)

    def hide_averagetype_menu(self):
        for i in range(self.menubar.index("end") + 1):
            if self.menubar.type(i) == "cascade":
                if self.menubar.entrycget(i, "label") == "Average Type":
                    self.menubar.delete(i)
                    break

    def averagetype_menu_visible(self):
        for i in range(self.menubar.index("end") + 1):
            if self.menubar.type(i) == "cascade":
                if self.menubar.entrycget(i, "label") == "Average Type":
                    return True
        return False

    def update_channels(self):
        self.channels = [
            ch for ch, var in self.channel_vars.items()
            if var.get()
        ]
        self.update_line_plot()

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
        self.update_line_plot()

        tk.messagebox.showinfo("Import", f"Imported!")

    def set_linetype(self, mode):
        if mode == "curved": 
            self.curved_mode = True 
            self.root.title("Color Reader App - curved Line Mode")
        else: 
            self.curved_mode = False 
            self.root.title("Color Reader App - straight Line Mode")

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

        if hasattr(self, "left_placeholder"):
            self.left_placeholder.place_forget()

        canvas_width = self.canvas.winfo_width() or self.image.width
        canvas_height = self.canvas.winfo_height() or self.image.height
        self.display_image = self.image.copy()
        self.display_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.update_line_plot()
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

        self.update_line_plot()
    
    def on_canvas_click(self, event):
        if getattr(self, "current_tool", None) == "picker":
            self.pick_color(event)
        elif getattr(self, "current_tool", None) == "line":
            self.on_mouse_down_line_tool(event)
        elif getattr(self, "current_tool", None) == "average":
            self.avg_start = (event.x, event.y)
            if self.avg_rect:
                self.canvas.delete(self.avg_rect)
                self.avg_rect = None

    def on_mouse_drag(self, event):
        if getattr(self, "current_tool", None) == "line":
            self.on_mouse_drag_line_tool(event)
        elif getattr(self, "current_tool", None) == "average":
            self.on_mouse_drag_average_tool(event)

    def on_mouse_up(self, event):
        if getattr(self, "current_tool", None) == "line":
            self.on_mouse_up_line_tool(event)
        elif getattr(self, "current_tool", None) == "average":
            self.on_mouse_up_average_tool(event)

    def on_mouse_down_line_tool(self, event):

        if self.image is None or self.current_tool != "line": return

        self.currently_drawing = True

        if self.current_line:
            self.canvas.delete(self.current_line)

        self.path = [(event.x, event.y)]
        self.current_line = self.canvas.create_line(
            event.x, event.y, event.x, event.y,
            fill="red", width=2, smooth=self.curved_mode
        )

    def on_mouse_drag_line_tool(self, event):
        if self.image is None: return
        
        if self.curved_mode:
            self.path.append((event.x, event.y))
        else:
            self.path = [self.path[0], (event.x, event.y)]

        flat = [v for pt in self.path for v in pt]
        self.canvas.coords(self.current_line, *flat)

        self.update_line_plot()

    def on_mouse_up_line_tool(self, event):
        if self.image is None or self.current_tool != "line": return
        self.currently_drawing = False
        self.update_line_plot()

    def on_mouse_drag_average_tool(self, event):
        if self.image is None: return

        if self.avg_rect:
            self.canvas.delete(self.avg_rect)

        x0, y0 = self.avg_start
        x1, y1 = event.x, event.y
        
        self.avg_rect = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)

    def on_mouse_up_average_tool(self, event):
        if self.image is None or self.avg_start is None: return

        x0, y0 = self.avg_start
        x1, y1 = event.x, event.y
        self.avg_start = None

        scale_x = self.display_image.width / self.image.width
        scale_y = self.display_image.height / self.image.height
        img_x0 = max(0, min(self.image.width - 1, int(x0 / scale_x)))
        img_y0 = max(0, min(self.image.height - 1, int(y0 / scale_y)))
        img_x1 = max(0, min(self.image.width - 1, int(x1 / scale_x)))
        img_y1 = max(0, min(self.image.height - 1, int(y1 / scale_y)))

        x_start, x_end = sorted([img_x0, img_x1])
        y_start, y_end = sorted([img_y0, img_y1])

        region = np.array(self.image.crop((x_start, y_start, x_end+1, y_end+1)))
        if region.size == 0:
            return
        r_avg = int(np.mean(region[:,:,0]))
        g_avg = int(np.mean(region[:,:,1]))
        b_avg = int(np.mean(region[:,:,2]))

        hex_color = f"#{r_avg:02x}{g_avg:02x}{b_avg:02x}"
        self.average_color_square.config(bg=hex_color)
        self.average_color_label.config(text=f"R: {r_avg} G: {g_avg} B: {b_avg}")

        if self.avg_rect:
            self.canvas.delete(self.avg_rect)
            self.avg_rect = None

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

    def update_line_plot(self):
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

    def hide_line_plot(self):
        self.plot_widget.pack_forget()

    def pick_color(self, event):
        if self.image is None:
            return

        scale_x = self.display_image.width / self.image.width
        scale_y = self.display_image.height / self.image.height
        img_x = int(event.x / scale_x)
        img_y = int(event.y / scale_y)

        r, g, b = self.image.getpixel((img_x, img_y))

        square = self.color_squares[self.next_color_index]
        label = self.color_labels[self.next_color_index]

        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        square.config(bg=hex_color)
        label.config(text=f"R: {r} G: {g} B: {b}")

        self.next_color_index = (self.next_color_index + 1) % len(self.color_squares)

    def average_color(self, event):
        if self.image is None:
            return

        scale_x = self.display_image.width / self.image.width
        scale_y = self.display_image.height / self.image.height
        img_x = int(event.x / scale_x)
        img_y = int(event.y / scale_y)

        r, g, b = self.image.getpixel((img_x, img_y))

        square = self.color_squares[self.next_color_index]
        label = self.color_labels[self.next_color_index]

        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        square.config(bg=hex_color)
        label.config(text=f"R: {r} G: {g} B: {b}")

        self.next_color_index = (self.next_color_index + 1) % len(self.color_squares)

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorReaderApp(root)
    root.mainloop()
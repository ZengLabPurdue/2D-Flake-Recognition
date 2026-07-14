import re
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import Frame, Label, filedialog, messagebox
from tkinter import ttk

class ViewScansPanel:
    def __init__(
        self,
        parent,
        root,
        app,
    ):
        self.parent = parent
        self.root = root
        self.app = app

        self.view_chip_index = 0
        self.view_image_index = 0
        self.view_scan_path = None
        self.view_folder = None
        self.image_files = None

        self.results_menu = None
        self.open_scan_menu = None

        self._build_panel()
        self.frame.place_forget()

    def _build_panel(self):
        self.pos_scan_name = 50
        self.pos_chip = 85
        self.pos_image = 125
        self.pos_buttons = 160

        self.frame = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=225
        )
        self.frame.place(relx=0.0, rely=0.0, anchor="nw")

        self.background = Frame(
            self.frame,
            bg="white",
            width=200,
            height=223
        )
        self.background.place(x=2, y=0)

        title = Label(
            self.frame,
            text="Scan Results",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title.place(relx=0.5, y=10, anchor="n")

        self.scan_name_var = tk.StringVar(value="Scan: Not Selected")

        self.scan_name_label = Label(
            self.frame,
            textvariable=self.scan_name_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        self.scan_name_label.place(relx=0.5, y=self.pos_scan_name, anchor="n")

        self.chip_var = tk.StringVar()

        self.chip_dropdown = ttk.Combobox(
            self.frame,
            textvariable=self.chip_var,
            state="readonly"
        )
        self.chip_dropdown.place(relx=0.5, y=self.pos_chip, anchor="n")

        self.image_var = tk.StringVar(value="Image: None")

        self.image_label = Label(
            self.frame,
            textvariable=self.image_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        self.image_label.place(relx=0.5, y=self.pos_image, anchor="n")

        self.button_panel = Frame(
            self.frame,
            bg="white",
            width=80,
            height=45
        )
        self.button_panel.place(relx=0.5, x=0, y=self.pos_buttons, anchor="n")
        self.button_panel.pack_propagate(False)

        controls = Frame(self.button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        self.btn_next = ttk.Button(controls, text="▸", style="Arrow.TButton")
        self.btn_previous = ttk.Button(controls, text="◂", style="Arrow.TButton")

        self.btn_next.bind("<ButtonPress-1>", self.next_image)
        self.btn_previous.bind("<ButtonPress-1>", self.previous_image)

        self.root.bind("<Left>", self.previous_image)
        self.root.bind("<Right>", self.next_image)

        controls.rowconfigure(0, weight=1)
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        self.btn_previous.grid(row=0, column=0, sticky="nsew")
        self.btn_next.grid(row=0, column=1, sticky="nsew")

    def add_to_menu(self, parent_menu):
        self.results_menu = tk.Menu(parent_menu, tearoff=0)

        self.results_menu.add_command(
            label="Open Scan...",
            command=self.open_scan
        )
        self.results_menu.add_separator()

        self.open_scan_menu = tk.Menu(self.results_menu, tearoff=0)

        self.open_scan_menu.add_command(
            label="Raw Images (2x)",
            command=lambda: self.set_view_folder("Raw 2x")
        )
        self.open_scan_menu.add_command(
            label="Raw Images (10x)",
            command=lambda: self.set_view_folder("Raw 10x")
        )
        self.open_scan_menu.add_command(
            label="Processed Images (10x)",
            command=lambda: self.set_view_folder("Processed 10x")
        )
        self.open_scan_menu.add_command(
            label="Detected Flakes",
            command=lambda: self.set_view_folder("Flakes Found")
        )

        self.results_menu.add_cascade(
            label="View Scan",
            state="disabled",
            menu=self.open_scan_menu
        )

        self.results_menu.add_command(
            label="Classify Flakes",
            state="disabled",
            command=None
        )

        parent_menu.add_cascade(label="Results", menu=self.results_menu)

    def show(self):
        self.app.close_all_panels()
        self.frame.place(relx=0.0, rely=0.0, anchor="nw")

    def hide(self):
        self.frame.place_forget()

    def display_chip_dropdown(self, display=True):
        shift = 0 if display else -40

        if display:
            self.chip_dropdown.place(relx=0.5, y=self.pos_chip, anchor="n")
        else:
            self.chip_dropdown.place_forget()

        self.image_label.place(relx=0.5, y=self.pos_image + shift, anchor="n")
        self.button_panel.place(relx=0.5, y=self.pos_buttons + shift, anchor="n")

        base_height = 225
        new_height = base_height + shift

        self.frame.config(height=new_height)
        self.background.config(height=new_height - 2)

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
        self.scan_name_var.set(folder_name)

        if self.results_menu is not None:
            self.results_menu.entryconfig("View Scan", state="normal")
            self.results_menu.entryconfig("Classify Flakes", state="normal")

        messagebox.showinfo(
            "Scan Loaded",
            "Scan loaded successfully!"
        )

    def set_view_folder(self, selected_view):
        if self.view_scan_path is None:
            messagebox.showwarning(
                "No Scan Selected",
                "Please open a scan folder first."
            )
            return

        self.app.set_view("Scan Results")
        self.show()

        self.view_chip_index = 0
        self.view_image_index = 0

        base_path = self.view_scan_path / "All Images"

        if selected_view == "Raw 2x":
            self.view_folder = base_path / "2x" / "Raw"
            self.display_chip_dropdown(False)

        elif selected_view == "Raw 10x":
            chip_folder = self.get_subfolder(base_path / "10x", self.view_chip_index)
            if chip_folder is None:
                self._show_missing_folder("No 10x chip folders found.")
                return

            self.view_folder = base_path / "10x" / chip_folder.name / "Raw"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(base_path / "10x")

        elif selected_view == "Processed 10x":
            chip_folder = self.get_subfolder(base_path / "10x", self.view_chip_index)
            if chip_folder is None:
                self._show_missing_folder("No 10x chip folders found.")
                return

            self.view_folder = base_path / "10x" / chip_folder.name / "Processed"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(base_path / "10x")

        elif selected_view == "Flakes Found":
            chip_folder = self.get_subfolder(self.view_scan_path / "Flakes Found", self.view_chip_index)
            if chip_folder is None:
                self._show_missing_folder("No detected flake folders found.")
                return

            self.view_folder = self.view_scan_path / "Flakes Found" / chip_folder.name
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(self.view_scan_path / "Flakes Found")

        self.load_current_folder()

    def load_current_folder(self):
        if self.view_folder is None or not self.view_folder.exists():
            self._show_missing_folder(f"Folder does not exist:\n{self.view_folder}")
            return

        self.image_files = sorted(
            [
                p for p in self.view_folder.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
            ],
            key=self.image_sort_key
        )

        if not self.image_files:
            self.image_var.set("Image: None")
            messagebox.showwarning(
                "No Images Found",
                f"No images found in:\n{self.view_folder}"
            )
            return

        self.view_image_index = 0
        self.display_current_image()

    def display_current_image(self):
        if not self.image_files:
            return

        img_path = self.image_files[self.view_image_index]

        self.image_var.set(f"Image: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            messagebox.showwarning(
                "Image Error",
                f"Could not read image:\n{img_path}"
            )
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.app.display_image(img)
        self.root.update()

    def previous_image(self, event=None):
        if self.app.get_view() != "Scan Results":
            return

        if not self.image_files:
            return

        self.view_image_index = (self.view_image_index - 1) % len(self.image_files)
        self.display_current_image()
        self.root.focus_set()

    def next_image(self, event=None):
        if self.app.get_view() != "Scan Results":
            return

        if not self.image_files:
            return

        self.view_image_index = (self.view_image_index + 1) % len(self.image_files)
        self.display_current_image()
        self.root.focus_set()

    def populate_chips_dropdown(self, chip_root):
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
        match = re.search(r"_(\d+)\.", p.name)
        return int(match.group(1)) if match else -1

    def get_subfolder(self, path, index):
        if not path.exists():
            return None

        subfolders = sorted([p for p in path.iterdir() if p.is_dir()])
        return subfolders[index] if 0 <= index < len(subfolders) else None

    def _show_missing_folder(self, message):
        self.image_files = []
        self.image_var.set("Image: None")
        messagebox.showwarning("Missing Folder", message)

import re
from pathlib import Path

import tkinter as tk
from tkinter import Frame, Label, filedialog, messagebox
from tkinter import ttk

from UI.sparse_tile_viewer import SparseTileViewer


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
FILTERED_MAP_VIEW_PAIRS = {
    "Raw 2x": "Filtered 2x",
    "Raw 10x": "Processed 10x",
}


def _natural_sort_key(value):
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", str(value))
    )


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
        self.selected_view = None
        self.sparse_map_mode = False
        self.chip_root = None
        self.chip_source_folder = None
        self.magnification_roots = {}
        self.available_result_views = set()
        self.tile_viewer = SparseTileViewer(self.app.scan_results_canvas)

        self.results_menu = None
        self.open_scan_menu = None

        self._build_panel()
        self.frame.place_forget()

    def _build_panel(self):
        self.frame = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=225,
        )
        self.frame.place(relx=0.0, rely=0.0, anchor="nw")

        self.background = Frame(
            self.frame,
            bg="white",
            width=200,
            height=223,
        )
        self.background.place(x=2, y=0)

        self.content = Frame(self.background, bg="white", width=184)
        self.content.place(x=8, y=8, width=184)
        self.content.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.configure(
            "ScanResults.TCheckbutton",
            background="white",
            foreground="black",
        )

        self.title_label = Label(
            self.content,
            text="Scan Results",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13),
        )
        self.title_label.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.scan_name_var = tk.StringVar(value="Scan: Not Selected")

        self.scan_name_label = Label(
            self.content,
            textvariable=self.scan_name_var,
            bg="white",
            fg="black",
            font="TkDefaultFont",
            justify="center",
            wraplength=176,
        )
        self.scan_name_label.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )

        self.chip_var = tk.StringVar()

        self.chip_dropdown = ttk.Combobox(
            self.content,
            textvariable=self.chip_var,
            state="readonly",
            width=1,
        )
        self.chip_dropdown.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )
        self.chip_dropdown.bind("<<ComboboxSelected>>", self._on_chip_selected)

        self.image_var = tk.StringVar(value="Image: None")

        self.image_label = Label(
            self.content,
            textvariable=self.image_var,
            bg="white",
            fg="black",
            font="TkDefaultFont",
            justify="center",
            wraplength=176,
        )
        self.image_label.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )

        self.filtered_map_var = tk.BooleanVar(value=False)
        self.filtered_map_checkbox = ttk.Checkbutton(
            self.content,
            text="Filtered map",
            variable=self.filtered_map_var,
            command=self._toggle_filtered_map,
            style="ScanResults.TCheckbutton",
        )

        self.button_panel = Frame(
            self.content,
            bg="white",
            width=80,
            height=32,
        )
        self.button_panel.grid_propagate(False)

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

        self._show_chip = True
        self._show_navigation = True
        self._show_filter_toggle = False
        self._resize_job = None
        self.scan_name_var.trace_add("write", self._queue_panel_resize)
        self.image_var.trace_add("write", self._queue_panel_resize)
        self._layout_controls(
            show_chip=True,
            show_navigation=True,
            show_filter_toggle=False,
        )

    def _queue_panel_resize(self, *_args):
        if self._resize_job is None:
            self._resize_job = self.frame.after_idle(self._refresh_panel_height)

    def _refresh_panel_height(self):
        self._resize_job = None
        self.content.update_idletasks()
        visible_elements = (
            self.title_label,
            self.scan_name_label,
            self.chip_dropdown,
            self.image_label,
            self.filtered_map_checkbox,
            self.button_panel,
        )
        bottom = max(
            (
                element.winfo_y() + element.winfo_height()
                for element in visible_elements
                if element.winfo_manager()
            ),
            default=0,
        )
        height = max(96, self.content.winfo_y() + bottom + 15)
        self.frame.configure(height=height)
        self.background.configure(height=max(1, height - 2))

    def _layout_controls(
        self,
        show_chip=None,
        show_navigation=None,
        show_filter_toggle=None,
    ):
        if show_chip is not None:
            self._show_chip = bool(show_chip)
        if show_navigation is not None:
            self._show_navigation = bool(show_navigation)
        if show_filter_toggle is not None:
            self._show_filter_toggle = bool(show_filter_toggle)
        if self._show_chip:
            self.chip_dropdown.grid()
        else:
            self.chip_dropdown.grid_remove()

        if self._show_filter_toggle:
            self.filtered_map_checkbox.grid(
                row=4,
                column=0,
                sticky="w",
                pady=(0, 8),
            )
        else:
            self.filtered_map_checkbox.grid_remove()

        self.button_panel.grid_remove()
        if self._show_navigation:
            self.button_panel.grid(
                row=5,
                column=0,
                pady=(0, 2),
            )
        self._queue_panel_resize()

    def add_to_menu(self, parent_menu):
        self.results_menu = tk.Menu(parent_menu, tearoff=0)

        self.results_menu.add_command(
            label="Open Scan...",
            command=self.open_scan
        )
        self.results_menu.add_separator()

        self.open_scan_menu = tk.Menu(self.results_menu, tearoff=0)
        self._rebuild_open_scan_menu()

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

    def _rebuild_open_scan_menu(self):
        if self.open_scan_menu is None:
            return
        self.open_scan_menu.delete(0, "end")

        groups = []
        two_x_items = []
        if "Raw 2x" in self.available_result_views:
            two_x_items.append(("2x Map", "Raw 2x", "normal"))
        elif self.view_scan_path is None:
            two_x_items.append(("2x Map", "Raw 2x", "disabled"))
        if "Filtered 2x" in self.available_result_views:
            two_x_items.append(("Filtered 2x Map", "Filtered 2x", "normal"))
        if "Scan Windows" in self.available_result_views:
            two_x_items.append(("2x Scan Windows", "Scan Windows", "normal"))
        if two_x_items:
            groups.append(two_x_items)

        for magnification in ("10x", "20x"):
            items = []
            raw_view = f"Raw {magnification[:-1]}x"
            processed_view = f"Processed {magnification[:-1]}x"
            if raw_view in self.available_result_views:
                items.append((f"{magnification} Map", raw_view, "normal"))
            if processed_view in self.available_result_views:
                items.append(
                    (
                        f"Processed {magnification} Map",
                        processed_view,
                        "normal",
                    )
                )
            if items:
                groups.append(items)

        groups.append([
            (
                "Detected Flakes",
                "Flakes Found",
                (
                    "normal"
                    if "Flakes Found" in self.available_result_views
                    else "disabled"
                ),
            )
        ])

        for group_index, group in enumerate(groups):
            if group_index:
                self.open_scan_menu.add_separator()
            for label, selected_view, state in group:
                self.open_scan_menu.add_command(
                    label=label,
                    state=state,
                    command=lambda view=selected_view: self.set_view_folder(view),
                )

    def _discover_result_layers(self):
        roots = {}
        available = set()
        if self.view_scan_path is None:
            return roots, available

        all_images = self.view_scan_path / "All Images"
        try:
            candidates = [path for path in all_images.iterdir() if path.is_dir()]
        except OSError:
            candidates = []
        for candidate in candidates:
            normalized = re.sub(r"[\s_-]+", "", candidate.name).casefold()
            if normalized in {"2x", "10x", "20x"}:
                roots[normalized] = candidate

        maps = self.view_scan_path / "Maps"
        two_x_root = roots.get("2x")
        if (
            self._folder_has_images(self._named_child(two_x_root, "Raw"))
            or (maps / "map_2x.png").is_file()
        ):
            available.add("Raw 2x")
        if (
            self._folder_has_images(self._named_child(two_x_root, "Filtered"))
            or (maps / "map_2x_filtered.png").is_file()
        ):
            available.add("Filtered 2x")
        if (maps / "map_2x_scan_windows.png").is_file():
            available.add("Scan Windows")

        for magnification in ("10x", "20x"):
            root = roots.get(magnification)
            if self._chip_folders(root, source_name="Raw"):
                available.add(f"Raw {magnification[:-1]}x")
            if self._chip_folders(root, source_name="Processed"):
                available.add(f"Processed {magnification[:-1]}x")

        flakes_root = self.view_scan_path / "Flakes Found"
        if self._chip_folders(flakes_root, require_images=True):
            available.add("Flakes Found")
        return roots, available

    def _magnification_root(self, magnification, base_path):
        return (
            self.magnification_roots.get(magnification.casefold())
            or self._named_child(base_path, magnification)
            or base_path / magnification
        )

    @staticmethod
    def _named_child(parent, name):
        if parent is None or not parent.is_dir():
            return None
        try:
            for child in parent.iterdir():
                if child.is_dir() and child.name.casefold() == name.casefold():
                    return child
        except OSError:
            return None
        return None

    @staticmethod
    def _folder_has_images(folder):
        if folder is None or not folder.is_dir():
            return False
        try:
            return any(
                path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                for path in folder.iterdir()
            )
        except OSError:
            return False

    def _chip_folders(self, root, source_name=None, require_images=False):
        if root is None or not root.is_dir():
            return []
        try:
            folders = sorted(
                [path for path in root.iterdir() if path.is_dir()],
                key=lambda folder: _natural_sort_key(folder.name),
            )
        except OSError:
            return []

        matching = []
        for folder in folders:
            if source_name is not None:
                source_folder = self._named_child(folder, source_name)
                if not self._folder_has_images(source_folder):
                    continue
            elif require_images and not self._folder_has_images(folder):
                continue
            matching.append(folder)
        return matching

    def show(self):
        self.app.close_all_panels()
        self.frame.place(relx=0.0, rely=0.0, anchor="nw")

    def hide(self):
        self.frame.place_forget()

    def pause(self):
        self.tile_viewer.pause()

    def shutdown(self):
        self.tile_viewer.shutdown()

    def display_chip_dropdown(self, display=True):
        self._layout_controls(show_chip=display)

    @staticmethod
    def _map_layer_pair(selected_view):
        if selected_view in FILTERED_MAP_VIEW_PAIRS:
            return selected_view, FILTERED_MAP_VIEW_PAIRS[selected_view]
        for raw_view, filtered_view in FILTERED_MAP_VIEW_PAIRS.items():
            if selected_view == filtered_view:
                return raw_view, filtered_view
        return None

    def _configure_filtered_map_toggle(self, selected_view):
        pair = self._map_layer_pair(selected_view)
        available = bool(
            pair
            and pair[0] in self.available_result_views
            and pair[1] in self.available_result_views
        )
        self.filtered_map_var.set(
            bool(pair and selected_view == pair[1])
        )
        self._layout_controls(show_filter_toggle=available)

    def _toggle_filtered_map(self):
        pair = self._map_layer_pair(self.selected_view)
        if pair is None:
            self.filtered_map_var.set(False)
            return

        raw_view, filtered_view = pair
        target_view = (
            filtered_view
            if self.filtered_map_var.get()
            else raw_view
        )
        if target_view not in self.available_result_views:
            self.filtered_map_var.set(self.selected_view == filtered_view)
            return

        self.set_view_folder(
            target_view,
            view_state=self.tile_viewer.capture_view_state(),
            preserve_context=True,
        )

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
        self.selected_view = None
        self.chip_var.set("")
        self.filtered_map_var.set(False)
        self._layout_controls(show_filter_toggle=False)
        self.magnification_roots, self.available_result_views = (
            self._discover_result_layers()
        )
        self._rebuild_open_scan_menu()

        if self.results_menu is not None:
            self.results_menu.entryconfig("View Scan", state="normal")
            self.results_menu.entryconfig("Classify Flakes", state="normal")

        # Detected flakes are the landing page even for a zero-flake scan; the
        # latter gets a quiet empty state instead of an interrupting warning.
        self.set_view_folder("Flakes Found")

    def set_view_folder(
        self,
        selected_view,
        view_state=None,
        preserve_context=False,
    ):
        if self.view_scan_path is None:
            messagebox.showwarning(
                "No Scan Selected",
                "Please open a scan folder first."
            )
            return

        self.app.set_view("Scan Results")
        self.show()

        previous_chip_name = (
            self.chip_var.get()
            if preserve_context
            else None
        )
        if not preserve_context:
            self.view_chip_index = 0
            self.view_image_index = 0
        self.selected_view = selected_view
        self.sparse_map_mode = selected_view != "Flakes Found"
        self.chip_root = None
        self.chip_source_folder = None
        self._configure_filtered_map_toggle(selected_view)
        self._layout_controls(
            show_chip=False,
            show_navigation=False,
        )

        base_path = self.view_scan_path / "All Images"

        if selected_view == "Raw 2x":
            magnification_root = self._magnification_root("2x", base_path)
            self.chip_var.set("")
            self.view_folder = self._named_child(magnification_root, "Raw")
            self.display_chip_dropdown(False)
            if self.view_folder is None:
                self._layout_controls(
                    show_chip=False,
                    show_navigation=False,
                )
                self._load_flattened_map(
                    "map_2x.png",
                    "Raw 2x",
                    False,
                    view_state=view_state,
                )
                return

        elif selected_view == "Filtered 2x":
            magnification_root = self._magnification_root("2x", base_path)
            self.chip_var.set("")
            self.view_folder = self._named_child(magnification_root, "Filtered")
            self.display_chip_dropdown(False)
            self._layout_controls(
                show_chip=False,
                show_navigation=False,
            )
            if self.view_folder is not None:
                self.load_current_folder(view_state=view_state)
            else:
                self._load_flattened_map(
                    "map_2x_filtered.png",
                    "Filtered 2x Map",
                    True,
                    view_state=view_state,
                )
            return

        elif selected_view == "Scan Windows":
            self.chip_var.set("")
            self.view_folder = self.view_scan_path / "Maps"
            self.display_chip_dropdown(False)
            self._layout_controls(
                show_chip=False,
                show_navigation=False,
            )
            self._load_flattened_map(
                "map_2x_scan_windows.png",
                "2x Scan Windows",
                False,
                view_state=view_state,
            )
            return

        elif selected_view == "Raw 10x":
            magnification_root = self._magnification_root("10x", base_path)
            chip_folder = self.get_subfolder(
                magnification_root,
                self.view_chip_index,
                source_name="Raw",
                preferred_name=previous_chip_name,
            )
            if chip_folder is None:
                self._show_missing_folder("No 10x chip folders found.")
                return

            self.view_folder = self._named_child(chip_folder, "Raw")
            self.chip_root = magnification_root
            self.chip_source_folder = "Raw"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(
                magnification_root,
                source_name="Raw",
                selected_name=chip_folder.name,
            )

        elif selected_view == "Processed 10x":
            magnification_root = self._magnification_root("10x", base_path)
            chip_folder = self.get_subfolder(
                magnification_root,
                self.view_chip_index,
                source_name="Processed",
                preferred_name=previous_chip_name,
            )
            if chip_folder is None:
                self._show_missing_folder("No 10x chip folders found.")
                return

            self.view_folder = self._named_child(chip_folder, "Processed")
            self.chip_root = magnification_root
            self.chip_source_folder = "Processed"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(
                magnification_root,
                source_name="Processed",
                selected_name=chip_folder.name,
            )

        elif selected_view == "Raw 20x":
            magnification_root = self._magnification_root("20x", base_path)
            chip_folder = self.get_subfolder(
                magnification_root,
                self.view_chip_index,
                source_name="Raw",
            )
            if chip_folder is None:
                self._show_missing_folder("No 20x chip folders found.")
                return

            self.view_folder = self._named_child(chip_folder, "Raw")
            self.chip_root = magnification_root
            self.chip_source_folder = "Raw"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(magnification_root, source_name="Raw")

        elif selected_view == "Processed 20x":
            magnification_root = self._magnification_root("20x", base_path)
            chip_folder = self.get_subfolder(
                magnification_root,
                self.view_chip_index,
                source_name="Processed",
            )
            if chip_folder is None:
                self._show_missing_folder("No 20x chip folders found.")
                return

            self.view_folder = self._named_child(chip_folder, "Processed")
            self.chip_root = magnification_root
            self.chip_source_folder = "Processed"
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(
                magnification_root,
                source_name="Processed",
            )

        elif selected_view == "Flakes Found":
            flakes_root = self.view_scan_path / "Flakes Found"
            chip_folder = self.get_subfolder(
                flakes_root,
                self.view_chip_index,
                require_images=True,
            )
            if chip_folder is None:
                self.view_folder = flakes_root
                self.image_files = []
                self.chip_var.set("")
                self._layout_controls(
                    show_chip=False,
                    show_navigation=False,
                )
                self.image_var.set("Image: No flakes found")
                self.tile_viewer.clear("No flakes found in this scan")
                return

            self.view_folder = flakes_root / chip_folder.name
            self.chip_root = flakes_root
            self.display_chip_dropdown(True)
            self.populate_chips_dropdown(flakes_root, require_images=True)

        self._layout_controls(
            show_chip=self.chip_root is not None,
            show_navigation=not self.sparse_map_mode,
        )
        self.load_current_folder(view_state=view_state)

    def load_current_folder(self, view_state=None):
        if self.view_folder is None or not self.view_folder.exists():
            self._show_missing_folder(f"Folder does not exist:\n{self.view_folder}")
            return

        if self.sparse_map_mode:
            title = self.selected_view
            if self.chip_var.get():
                title = f"{title} - {self.chip_var.get()}"
            self.image_var.set(f"Map: {title}")
            use_nearest = self.selected_view.startswith(("Processed", "Filtered"))
            fallback_name = self._flattened_map_name()
            overview_path = (
                self.view_scan_path / "Maps" / fallback_name
                if fallback_name is not None
                else None
            )

            def load_flattened_fallback():
                if fallback_name is not None:
                    self._load_flattened_map(
                        fallback_name,
                        title,
                        use_nearest,
                        view_state=view_state,
                    )
                else:
                    self.tile_viewer.clear("No positioned scan tiles were found")

            if self.tile_viewer.load_tiles(
                self.view_folder,
                title=title,
                nearest=use_nearest,
                on_empty=load_flattened_fallback,
                overview_path=overview_path,
                view_state=view_state,
            ):
                return
            if fallback_name is not None:
                self._load_flattened_map(
                    fallback_name,
                    title,
                    use_nearest,
                    view_state=view_state,
                )
            return

        self.image_files = sorted(
            [
                p for p in self.view_folder.iterdir()
                if p.suffix.lower() in IMAGE_SUFFIXES
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

        if not self.tile_viewer.load_image(img_path, title=img_path.name):
            messagebox.showwarning(
                "Image Error",
                f"Could not read image:\n{img_path}"
            )
            return

    def previous_image(self, event=None):
        if self.app.get_view() != "Scan Results":
            return

        if self.sparse_map_mode:
            return

        if not self.image_files:
            return

        self.view_image_index = (self.view_image_index - 1) % len(self.image_files)
        self.display_current_image()
        self.root.focus_set()

    def next_image(self, event=None):
        if self.app.get_view() != "Scan Results":
            return

        if self.sparse_map_mode:
            return

        if not self.image_files:
            return

        self.view_image_index = (self.view_image_index + 1) % len(self.image_files)
        self.display_current_image()
        self.root.focus_set()

    def populate_chips_dropdown(
        self,
        chip_root,
        source_name=None,
        require_images=False,
        selected_name=None,
    ):
        if not chip_root.exists():
            self.chip_dropdown["values"] = []
            self.chip_var.set("No chips found")
            return

        chips = [
            path.name
            for path in self._chip_folders(
                chip_root,
                source_name=source_name,
                require_images=require_images,
            )
        ]
        self.chip_dropdown["values"] = chips

        if chips:
            selected = (
                selected_name
                if selected_name in chips
                else chips[min(self.view_chip_index, len(chips) - 1)]
            )
            self.view_chip_index = chips.index(selected)
            self.chip_var.set(selected)
        else:
            self.chip_var.set("No chips found")

    def _on_chip_selected(self, _event=None):
        if self.chip_root is None:
            return
        selected_chip = self.chip_var.get()
        if not selected_chip:
            return
        chip_path = self.chip_root / selected_chip
        self.view_chip_index = list(self.chip_dropdown["values"]).index(selected_chip)
        self.view_image_index = 0
        if self.chip_source_folder is not None:
            self.view_folder = self._named_child(
                chip_path,
                self.chip_source_folder,
            )
        else:
            self.view_folder = chip_path
        self.load_current_folder()

    def _flattened_map_name(self):
        map_index = self.view_chip_index + 1
        wafer_match = re.search(
            r"\bwafer\s+(\d+)\b",
            self.chip_var.get(),
            re.IGNORECASE,
        )
        if wafer_match:
            map_index = int(wafer_match.group(1))
        names = {
            "Raw 2x": "map_2x.png",
            "Filtered 2x": "map_2x_filtered.png",
            "Raw 10x": f"map_10x_wafer_{map_index}.png",
            "Processed 10x": f"map_10x_processed_wafer_{map_index}.png",
            "Raw 20x": f"map_20x_wafer_{map_index}.png",
            "Processed 20x": f"map_20x_processed_wafer_{map_index}.png",
        }
        return names.get(self.selected_view)

    def _load_flattened_map(
        self,
        filename,
        title,
        nearest=False,
        view_state=None,
    ):
        map_path = self.view_scan_path / "Maps" / filename
        self.image_var.set(f"Map: {title}")
        if not map_path.is_file():
            self.tile_viewer.clear(f"Map not found: {filename}")
            messagebox.showwarning("Missing Map", f"Map does not exist:\n{map_path}")
            return False
        return self.tile_viewer.load_image(
            map_path,
            title=title,
            nearest=nearest,
            view_state=view_state,
        )

    def image_sort_key(self, p):
        match = re.search(r"_(\d+)\.", p.name)
        return int(match.group(1)) if match else -1

    def get_subfolder(
        self,
        path,
        index,
        source_name=None,
        require_images=False,
        preferred_name=None,
    ):
        if not path.exists():
            return None

        subfolders = self._chip_folders(
            path,
            source_name=source_name,
            require_images=require_images,
        )
        if preferred_name:
            for subfolder in subfolders:
                if subfolder.name == preferred_name:
                    return subfolder
        return subfolders[index] if 0 <= index < len(subfolders) else None

    def _show_missing_folder(self, message):
        self.image_files = []
        self.image_var.set("Image: None")
        self.tile_viewer.clear(message)
        messagebox.showwarning("Missing Folder", message)

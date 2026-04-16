"""
Cross-platform flake labeling — Sobel on images under a ``Raw/`` folder, four classes:
Good / Bad / Unsure / No Flake. Each crop is saved under ``Labeled Data/<Class>/``.
A bottom panel reviews saved files and can move them between folders.
"""
from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import shutil
import sys
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

home_dir = Path(os.path.dirname(os.path.abspath(__file__)))
flake_finder_dir = home_dir / "Flake Recognition"
sys.path.insert(0, str(flake_finder_dir))

import flake_finder  # noqa: E402

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

CROP_SCALE = 1.2

# (slug, button label, outline color key for canvas filters)
CLASS_DEFS: list[tuple[str, str, str]] = [
    ("good", "Good", "good"),
    ("bad", "Bad", "bad"),
    ("unsure", "Unsure", "unsure"),
    ("no_flake", "No Flake", "no_flake"),
]

# Subfolder names under save root (slug → directory name)
CLASS_SAVE_DIRS: dict[str, str] = {
    "good": "Good",
    "bad": "Bad",
    "unsure": "Unsure",
    "no_flake": "No_Flake",
}

# Display folder name → slug (for review moves)
# Bounding-box colors (class)
OUTLINE_HEX = {
    "good": "#22c55e",
    "bad": "#ef4444",
    "unsure": "#eab308",
    "no_flake": "#94a3b8",
}
SEGMENTATION_LINE_HEX = "#ffffff"
CONTOUR_WIDTH = 1
CONTOUR_WIDTH_HOVER = 2
BBOX_WIDTH = 2
BBOX_WIDTH_HOVER = 3

# Canvas pixels — drag smaller than this is treated as a single click (box-select mode).
BOX_SELECT_MIN_DRAG = 5


def _apply_cross_platform_style(root: tk.Tk) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    bg = "#ececf0"
    fg = "#1a1a1e"
    style.configure("App.TFrame", background=bg)
    style.configure("Sidebar.TLabelframe", background=bg)
    style.configure("Sidebar.TLabelframe.Label", background=bg, foreground=fg, font=("TkDefaultFont", 10, "bold"))
    style.configure("Sidebar.TLabel", background=bg, foreground=fg)
    style.configure("Well.TFrame", background="#121214")
    style.configure("Well.TLabel", background="#121214", foreground="#9ca3af", font=("TkDefaultFont", 12))
    style.configure("Nav.TButton", padding=(8, 6))
    style.configure("Class.TButton", padding=(10, 8), font=("TkDefaultFont", 10))
    style.configure("Accent.TButton", padding=(8, 6))
    return style


class DataLabelingApp2:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Flake labeling (overlay)")
        self.root.minsize(960, 640)
        self.root.geometry("1200x780")

        self._style = _apply_cross_platform_style(root)

        self.save_folder_path = home_dir / "Labeled Data"
        self.save_folder_path.mkdir(parents=True, exist_ok=True)
        self._ensure_class_dirs()

        self.status_var = tk.StringVar(value=f"Save folder: {self.save_folder_path}")

        # Image / contour state
        self.images: list[Path | str] = []
        # When an image came from “Open Folder”, root used for relative titles (nested paths).
        self._folder_for_image: list[Path | None] = []
        self.current_image: Image.Image | None = None
        self.image_index = -1
        self.contours: list = []
        self.contour_labels: list[str | None] = []
        self.hover_idx: int | None = None
        self.active_label: str | None = None
        # Per-contour saved crop path (under current save_folder_path); cleared on new image / new save root.
        self.contour_saved_paths: dict[int, Path] = {}

        # Review strip: browse crops in class subfolders
        self.review_paths: list[Path] = []
        self.review_index = 0
        self._review_tk_img: ImageTk.PhotoImage | None = None

        # Display geometry (image fitted inside canvas)
        self._disp_w = 0
        self._disp_h = 0
        self._off_x = 0
        self._off_y = 0
        self._img_w = 0
        self._img_h = 0
        self._tk_image: ImageTk.PhotoImage | None = None

        # Drag rectangle (multi-label) when box_select_var is on
        self._box_dragging = False
        self._box_drag_start: tuple[float, float] | None = None
        self._box_drag_current: tuple[float, float] | None = None
        # After release: frozen rect + matched indices waiting for Enter to confirm
        self._box_pending_rect: tuple[float, float, float, float] | None = None
        self._box_pending_indices: list[int] = []

        self._build_layout()
        self._bind_keys()
        self.create_menu()

        self.label_area.bind("<Configure>", self.on_resize)
        self._review_refresh_list()

    def _build_layout(self):
        outer = ttk.Frame(self.root, style="App.TFrame", padding=0)
        outer.pack(fill=tk.BOTH, expand=True)

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        self.left_panel = ttk.Frame(body, style="App.TFrame", width=292, padding=(10, 10, 6, 10))
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.left_panel.pack_propagate(False)

        self.right_panel = ttk.Frame(body, style="Well.TFrame", padding=0)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_pane = ttk.PanedWindow(self.right_panel, orient=tk.VERTICAL)
        self.right_pane.pack(fill=tk.BOTH, expand=True)

        self.label_area = ttk.Frame(self.right_pane, style="Well.TFrame", padding=4)
        self.right_pane.add(self.label_area, weight=5)

        self.review_outer = ttk.Frame(self.right_pane, style="App.TFrame", padding=(6, 4))
        self.right_pane.add(self.review_outer, weight=2)

        self.right_placeholder = ttk.Label(
            self.label_area,
            text="Open a raw image or folder\n(File → Open Raw Image / Open Folder)\n\n"
            "Only paths under a Raw/ folder are loaded.\n"
            "Arm a class (1–4), click or drag a box (if enabled) — crops save to that class.",
            style="Well.TLabel",
            justify=tk.CENTER,
        )
        self.right_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        self.canvas = tk.Canvas(self.label_area, bg="#121214", highlightthickness=0, bd=0)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        # Note: <Button-1> is the same event as <ButtonPress-1> in Tk — bind only one.
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_button_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_b1_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_button_release)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        self._build_review_strip()

        self._build_sidebar()

        ttk.Label(outer, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(8, 4)).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    def _build_sidebar(self):
        row = 0

        lf_img = ttk.LabelFrame(self.left_panel, text="Current image", style="Sidebar.TLabelframe", padding=8)
        lf_img.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        self.image_name_var = tk.StringVar(value="No image loaded")
        ttk.Label(lf_img, textvariable=self.image_name_var, style="Sidebar.TLabel", wraplength=248, justify=tk.CENTER).pack(
            fill=tk.X
        )

        lf_counts = ttk.LabelFrame(self.left_panel, text="Contours", style="Sidebar.TLabelframe", padding=8)
        lf_counts.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        self.contour_count_var = tk.StringVar(value="Found: —")
        self.labeled_count_var = tk.StringVar(value="Labeled: —")
        ttk.Label(lf_counts, textvariable=self.contour_count_var, style="Sidebar.TLabel").pack(anchor=tk.W)
        ttk.Label(lf_counts, textvariable=self.labeled_count_var, style="Sidebar.TLabel").pack(anchor=tk.W)
        self.hover_info_var = tk.StringVar(value="Hover: —")
        ttk.Label(lf_counts, textvariable=self.hover_info_var, style="Sidebar.TLabel", font=("TkDefaultFont", 9)).pack(
            anchor=tk.W, pady=(4, 0)
        )

        self.active_label_var = tk.StringVar(value="Active class: (none)")
        ttk.Label(lf_counts, textvariable=self.active_label_var, style="Sidebar.TLabel", font=("TkDefaultFont", 9, "bold")).pack(
            anchor=tk.W, pady=(6, 0)
        )

        nav = ttk.Frame(self.left_panel, style="App.TFrame")
        nav.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1
        nav.columnconfigure(0, weight=1)
        nav.columnconfigure(1, weight=1)

        self.previous_image_button = ttk.Button(nav, text="Previous image", style="Nav.TButton", command=self.previous_image)
        self.next_image_button = ttk.Button(nav, text="Next image", style="Nav.TButton", command=self.next_image)
        self.previous_image_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.next_image_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.previous_image_button.state(["disabled"])
        self.next_image_button.state(["disabled"])

        lf_hover = ttk.LabelFrame(
            self.left_panel,
            text="Sobel (flake_finder) overlay",
            style="Sidebar.TLabelframe",
            padding=8,
        )
        lf_hover.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        hover_wrap = tk.Frame(lf_hover, bg="#ececf0")
        hover_wrap.pack(fill=tk.X)
        hover_cb_kw = dict(
            bd=0,
            highlightthickness=0,
            bg="#ececf0",
            activebackground="#ececf0",
            selectcolor="white",
            anchor="w",
        )
        # Default: one Sobel contour at a time, on top of the raw image (same pixels Sobel uses).
        self.hover_only_var = tk.IntVar(value=1)
        tk.Checkbutton(
            hover_wrap,
            text="Only show Sobel outline under mouse (raw image otherwise)",
            variable=self.hover_only_var,
            onvalue=1,
            offvalue=0,
            command=self.redraw_canvas,
            **hover_cb_kw,
        ).pack(fill=tk.X)

        self.box_select_var = tk.IntVar(value=0)
        tk.Checkbutton(
            hover_wrap,
            text="Drag rectangle to label many (armed class)",
            variable=self.box_select_var,
            onvalue=1,
            offvalue=0,
            command=self.redraw_canvas,
            **hover_cb_kw,
        ).pack(fill=tk.X, pady=(6, 0))

        # Filled flake overlay opacity
        opacity_row = tk.Frame(lf_hover, bg="#ececf0")
        opacity_row.pack(fill=tk.X, pady=(8, 0))
        tk.Label(opacity_row, text="Overlay opacity:", bg="#ececf0", anchor="w", font=("TkDefaultFont", 9)).pack(side=tk.LEFT)
        self.overlay_opacity_var = tk.DoubleVar(value=0.35)
        tk.Scale(
            opacity_row,
            variable=self.overlay_opacity_var,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=120,
            bg="#ececf0",
            troughcolor="#d1d5db",
            highlightthickness=0,
            command=lambda _: self.redraw_canvas(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Separator(lf_hover, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 4))

        self.suppress_yellow_var = tk.IntVar(value=0)
        tk.Checkbutton(
            lf_hover,
            text="Suppress yellow edges",
            variable=self.suppress_yellow_var,
            onvalue=1, offvalue=0,
            command=self._reload_contours,
            **hover_cb_kw,
        ).pack(fill=tk.X)

        def _yscale(parent, label, var, lo, hi, res):
            row = tk.Frame(parent, bg="#ececf0")
            row.pack(fill=tk.X, pady=(2, 0))
            tk.Label(row, text=label, bg="#ececf0", anchor="w",
                     font=("TkDefaultFont", 9), width=15).pack(side=tk.LEFT)
            tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, bg="#ececf0", troughcolor="#d1d5db",
                     highlightthickness=0,
                     command=lambda _: self._reload_contours(),
                     ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.yellow_hue_lo_var   = tk.IntVar(value=15)
        self.yellow_hue_hi_var   = tk.IntVar(value=38)
        self.yellow_sat_min_var  = tk.IntVar(value=60)
        self.yellow_val_min_var  = tk.IntVar(value=80)

        _yscale(lf_hover, "Hue lo (0–90):",   self.yellow_hue_lo_var,  0,  90, 1)
        _yscale(lf_hover, "Hue hi (0–90):",   self.yellow_hue_hi_var,  0,  90, 1)
        _yscale(lf_hover, "Sat min (0–255):", self.yellow_sat_min_var, 0, 255, 1)
        _yscale(lf_hover, "Val min (0–255):", self.yellow_val_min_var, 0, 255, 1)

        ttk.Separator(lf_hover, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 4))

        self.filter_yellow_interior_var = tk.IntVar(value=0)
        tk.Checkbutton(
            lf_hover,
            text="Drop yellow-interior contours",
            variable=self.filter_yellow_interior_var,
            onvalue=1, offvalue=0,
            command=self._reload_contours,
            **hover_cb_kw,
        ).pack(fill=tk.X)

        self.yellow_interior_pct_var = tk.IntVar(value=50)
        _yscale(lf_hover, "Interior % (0–100):", self.yellow_interior_pct_var, 1, 100, 1)


        lf_show = ttk.LabelFrame(
            self.left_panel,
            text="Visibility (mask + colored box)",
            style="Sidebar.TLabelframe",
            padding=8,
        )
        lf_show.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        chk_wrap = tk.Frame(lf_show, bg="#ececf0")
        chk_wrap.pack(fill=tk.X)
        cb_kw = dict(
            bd=0,
            highlightthickness=0,
            bg="#ececf0",
            activebackground="#ececf0",
            selectcolor="white",
            anchor="w",
        )
        # Classic tk.Checkbutton + IntVar: ttk BooleanVar is unreliable on some macOS setups.
        self.show_unlabeled_var = tk.IntVar(value=1)
        tk.Checkbutton(
            chk_wrap,
            text="Unlabeled (white mask only)",
            variable=self.show_unlabeled_var,
            onvalue=1,
            offvalue=0,
            command=self.redraw_canvas,
            **cb_kw,
        ).pack(fill=tk.X)

        self.filter_vars: dict[str, tk.IntVar] = {}
        for _slug, name, outline_key in CLASS_DEFS:
            v = tk.IntVar(value=1)
            self.filter_vars[outline_key] = v
            tk.Checkbutton(
                chk_wrap,
                text=f"{name} (box)",
                variable=v,
                onvalue=1,
                offvalue=0,
                command=self.redraw_canvas,
                **cb_kw,
            ).pack(fill=tk.X)

        ttk.Label(
            lf_show,
            text="Only files under a folder named Raw (any depth) are loaded.\n"
            "The opened file is used as-is (no overlay / twin substitution).",
            style="Sidebar.TLabel",
            wraplength=248,
            font=("TkDefaultFont", 8),
            foreground="#555",
        ).pack(anchor=tk.W, pady=(6, 0))

        ttk.Separator(self.left_panel, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", pady=6)
        row += 1

        lf_class = ttk.LabelFrame(
            self.left_panel,
            text="Classes (keys 1–4) — click to arm",
            style="Sidebar.TLabelframe",
            padding=(8, 8, 8, 4),
        )
        lf_class.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        self.buttons: dict[str, ttk.Button] = {}
        for i, (slug, title, _outline) in enumerate(CLASS_DEFS):
            key = str(i + 1)
            btn = ttk.Button(lf_class, text=f"{key}  {title}", style="Class.TButton", command=lambda s=slug: self.set_active_class(s))
            btn.grid(row=i, column=0, sticky="ew", pady=3)
            self.buttons[slug] = btn
            self.root.bind(f"<KeyPress-{key}>", lambda e, s=slug: self.set_active_class(s))

        lf_class.columnconfigure(0, weight=1)

        hint = ttk.Label(
            self.left_panel,
            text="Sobel runs on the image you opened (must be under a Raw/ folder).\n"
            "Crops save immediately under Good / Bad / Unsure / No_Flake.\n"
            "Enable “Drag rectangle…” then drag on the image to label every flake whose box intersects.\n"
            "Right‑click a flake to clear label and delete its saved crop.\n"
            "Use the review strip below the image to check and reclassify files.\n"
            "Up/Down: image.",
            style="Sidebar.TLabel",
            wraplength=248,
            justify=tk.CENTER,
            font=("TkDefaultFont", 9),
        )
        hint.grid(row=row, column=0, sticky="ew", pady=(6, 0))

        self.left_panel.columnconfigure(0, weight=1)

    def _build_review_strip(self) -> None:
        lf = ttk.LabelFrame(self.review_outer, text="Review saved crops", style="Sidebar.TLabelframe", padding=6)
        lf.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(lf, style="App.TFrame")
        top.pack(fill=tk.X)

        ttk.Label(top, text="Folder:", style="Sidebar.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.review_folder_var = tk.StringVar(value=CLASS_SAVE_DIRS["good"])
        self.review_folder_combo = ttk.Combobox(
            top,
            textvariable=self.review_folder_var,
            values=list(CLASS_SAVE_DIRS.values()),
            state="readonly",
            width=12,
        )
        self.review_folder_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.review_folder_combo.bind("<<ComboboxSelected>>", lambda _e: self._review_on_folder_change())

        ttk.Button(top, text="Prev", style="Nav.TButton", command=self._review_prev).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(top, text="Next", style="Nav.TButton", command=self._review_next).pack(side=tk.LEFT, padx=(0, 8))

        self.review_idx_var = tk.StringVar(value="0 / 0")
        ttk.Label(top, textvariable=self.review_idx_var, style="Sidebar.TLabel").pack(side=tk.LEFT)

        self.review_path_var = tk.StringVar(value="—")
        ttk.Label(lf, textvariable=self.review_path_var, style="Sidebar.TLabel", wraplength=520).pack(
            anchor=tk.W, pady=(4, 4)
        )

        self.review_canvas = tk.Canvas(lf, bg="#1a1a1e", highlightthickness=1, highlightbackground="#ccc", height=200)
        self.review_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.review_canvas.bind("<Configure>", self._on_review_canvas_configure)

        move_fr = ttk.Frame(lf, style="App.TFrame")
        move_fr.pack(fill=tk.X)
        ttk.Label(move_fr, text="Move to:", style="Sidebar.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        for slug, title, _ in CLASS_DEFS:
            ttk.Button(
                move_fr,
                text=title,
                style="Nav.TButton",
                command=lambda s=slug: self._review_move_to_slug(s),
            ).pack(side=tk.LEFT, padx=2)

    def _bind_keys(self):
        self.root.bind("<Up>", self.previous_image)
        self.root.bind("<Down>", self.next_image)
        self.root.bind("<Return>", self._confirm_box_selection)
        self.root.bind("<KP_Enter>", self._confirm_box_selection)
        self.root.bind("<Escape>", self._cancel_box_selection)

    @staticmethod
    def _is_under_raw_folder(path: Path) -> bool:
        """True if ``path`` includes a parent directory named Raw (case-insensitive)."""
        try:
            return any(part.lower() == "raw" for part in Path(path).resolve().parts)
        except OSError:
            return False

    def _pil_for_labeling(self, opened: Path) -> Image.Image:
        return Image.open(Path(opened)).convert("RGB")

    def _outline_for_slug(self, slug: str | None) -> str | None:
        if slug is None:
            return None
        for s, _t, ol in CLASS_DEFS:
            if s == slug:
                return ol
        return None

    def _class_title(self, slug: str) -> str:
        for s, t, _o in CLASS_DEFS:
            if s == slug:
                return t
        return slug

    def _ensure_class_dirs(self) -> None:
        base = Path(self.save_folder_path)
        for dirname in CLASS_SAVE_DIRS.values():
            (base / dirname).mkdir(parents=True, exist_ok=True)

    def _safe_stem(self) -> str:
        stem = Path(str(self.images[self.image_index])).stem
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)

    def _crop_basename(self, idx: int) -> str:
        return f"{self._safe_stem()}_c{idx}.png"

    def _save_crop_for_contour(self, idx: int, slug: str, *, skip_review_refresh: bool = False) -> None:
        if self.current_image is None or not self.contours or idx < 0 or idx >= len(self.contours):
            return
        self._ensure_class_dirs()
        dest_dir = Path(self.save_folder_path) / CLASS_SAVE_DIRS[slug]
        dest = dest_dir / self._crop_basename(idx)

        old = self.contour_saved_paths.get(idx)
        if old is not None and old != dest:
            try:
                if old.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists() and dest.resolve() != old.resolve():
                        dest.unlink()
                    shutil.move(str(old), str(dest))
                    self.contour_saved_paths[idx] = dest
                    try:
                        rel = dest.relative_to(self.save_folder_path)
                    except ValueError:
                        rel = dest
                    if not skip_review_refresh:
                        self.status_var.set(f"Flake #{idx + 1}: moved crop → {rel}")
                        self._review_refresh_list()
                    return
            except OSError as e:
                if not skip_review_refresh:
                    self.status_var.set(f"Flake #{idx + 1}: move failed ({e}); saving new file.")

        img_rgb = np.array(self.current_image)
        cnt = np.asarray(self.contours[idx], dtype=np.int32).reshape(-1, 1, 2)
        x, y, x2, y2 = self.find_contour_bounded_box(img_rgb, cnt)
        crop = img_rgb[y:y2, x:x2]
        if crop.size == 0:
            if not skip_review_refresh:
                self.status_var.set(f"Flake #{idx + 1}: crop empty; not saved.")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        self.contour_saved_paths[idx] = dest
        try:
            rel = dest.relative_to(self.save_folder_path)
        except ValueError:
            rel = dest
        if not skip_review_refresh:
            self.status_var.set(f"Flake #{idx + 1} → {self._class_title(slug)} — saved {rel}")
            self._review_refresh_list()

    def set_active_class(self, slug: str):
        self.active_label = slug
        self.active_label_var.set(f"Active class: {self._class_title(slug)}")
        self.status_var.set(f"Armed: {self._class_title(slug)} — click or drag a box (if multi-select is on)")

    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Raw Image…", command=self.open_image)
        file_menu.add_command(label="Open Folder… (Raw images only, recursive)", command=self.open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Choose save folder…", command=self.choose_save_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ]
        )
        if not file_path:
            return

        p = Path(file_path)
        if not self._is_under_raw_folder(p):
            self.status_var.set("Only files under a folder named Raw/ are loaded. Choose a file inside …/Raw/…")
            return
        self.images.append(p)
        self._folder_for_image.append(None)
        self.image_index = len(self.images) - 1
        self._refresh_current_image_from_index()

        if len(self.images) > 1:
            self.next_image_button.state(["!disabled"])
            self.previous_image_button.state(["!disabled"])

    def _collect_images_recursive(self, folder: Path) -> list[Path]:
        """Supported images under ``folder`` (recursive) that live under a ``Raw/`` directory."""
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        folder = Path(folder).resolve()
        out: list[Path] = []
        try:
            for p in folder.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in valid_ext:
                    continue
                if not self._is_under_raw_folder(p):
                    continue
                try:
                    rel = p.relative_to(folder)
                except ValueError:
                    continue
                if any(part.startswith(".") for part in rel.parts):
                    continue
                if "__pycache__" in rel.parts:
                    continue
                out.append(p)
        except OSError:
            return []
        out.sort(key=lambda x: str(x.relative_to(folder)).lower())
        return out

    def _refresh_current_image_from_index(self) -> None:
        p = Path(self.images[self.image_index])
        self.current_image = self._pil_for_labeling(p)
        root = self._folder_for_image[self.image_index]
        if root is not None:
            try:
                label = str(p.relative_to(root)).replace("\\", "/")
            except ValueError:
                label = p.name
        else:
            label = p.name
        self.image_name_var.set(label)
        self._load_contours_for_current_image()

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        root = Path(folder_path).resolve()
        file_list = self._collect_images_recursive(root)
        if not file_list:
            self.status_var.set(
                f"No raw images under {root}. Need files under a folder named Raw/ (recursive scan)."
            )
            return

        start_index = len(self.images)
        for fp in file_list:
            self.images.append(fp)
            self._folder_for_image.append(root)

        self.image_index = start_index
        self._refresh_current_image_from_index()

        self.status_var.set(
            f"Loaded {len(file_list)} raw image(s) under Raw/ from {root.name}. Save: {self.save_folder_path}"
        )
        self.next_image_button.state(["!disabled"])
        self.previous_image_button.state(["!disabled"])

    def _yellow_kwargs(self) -> dict:
        return dict(
            suppress_yellow=bool(self.suppress_yellow_var.get()),
            yellow_hue_lo=int(self.yellow_hue_lo_var.get()),
            yellow_hue_hi=int(self.yellow_hue_hi_var.get()),
            yellow_sat_min=int(self.yellow_sat_min_var.get()),
            yellow_val_min=int(self.yellow_val_min_var.get()),
            filter_yellow_interior=bool(self.filter_yellow_interior_var.get()),
            yellow_interior_pct=int(self.yellow_interior_pct_var.get()),
        )

    def _load_contours_for_current_image(self):
        if self.current_image is None:
            return
        img_bgr = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
        _bg, self.contours = flake_finder.find_flakes(img_bgr, display=False, **self._yellow_kwargs())
        self.contour_labels = [None] * len(self.contours)
        self.contour_saved_paths.clear()
        self.hover_idx = None
        self._box_pending_rect = None
        self._box_pending_indices = []
        n = len(self.contours)
        self.contour_count_var.set(f"Found: {n}")
        self._update_labeled_count()
        self.redraw_canvas()

    def _reload_contours(self) -> None:
        """Re-run detection with updated yellow settings; preserve existing labels by index."""
        if self.current_image is None:
            return
        img_bgr = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
        _bg, new_contours = flake_finder.find_flakes(img_bgr, display=False, **self._yellow_kwargs())
        old_labels = self.contour_labels[:]
        old_paths = dict(self.contour_saved_paths)
        self.contours = new_contours
        self.contour_labels = [
            old_labels[i] if i < len(old_labels) else None
            for i in range(len(new_contours))
        ]
        self.contour_saved_paths = {
            i: old_paths[i] for i in range(len(new_contours)) if i in old_paths
        }
        self.hover_idx = None
        self._box_pending_rect = None
        self._box_pending_indices = []
        self.contour_count_var.set(f"Found: {len(new_contours)}")
        self._update_labeled_count()
        self.redraw_canvas()

    def _update_labeled_count(self):
        n = len(self.contour_labels)
        k = sum(1 for x in self.contour_labels if x is not None)
        self.labeled_count_var.set(f"Labeled: {k} / {n}")

    def _compute_display_geometry(self, cw: int, ch: int):
        if self.current_image is None or cw < 2 or ch < 2:
            return
        self._img_w, self._img_h = self.current_image.size
        iw, ih = self._img_w, self._img_h
        scale = min(cw / iw, ch / ih)
        self._disp_w = max(1, int(iw * scale))
        self._disp_h = max(1, int(ih * scale))
        self._off_x = (cw - self._disp_w) // 2
        self._off_y = (ch - self._disp_h) // 2

    def _canvas_to_image(self, mx: float, my: float) -> tuple[int, int] | None:
        if self._disp_w <= 0 or self._disp_h <= 0:
            return None
        lx = mx - self._off_x
        ly = my - self._off_y
        if lx < 0 or ly < 0 or lx >= self._disp_w or ly >= self._disp_h:
            return None
        ix = int(lx * self._img_w / self._disp_w)
        iy = int(ly * self._img_h / self._disp_h)
        ix = max(0, min(self._img_w - 1, ix))
        iy = max(0, min(self._img_h - 1, iy))
        return ix, iy

    def _pick_contour_at(self, ix: int, iy: int) -> int | None:
        candidates: list[tuple[float, int]] = []
        for i, c in enumerate(self.contours):
            if cv2.pointPolygonTest(c, (float(ix), float(iy)), False) >= 0:
                candidates.append((cv2.contourArea(c), i))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def _contour_to_canvas_flat(self, contour) -> list[float]:
        pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        out: list[float] = []
        sx = self._disp_w / self._img_w
        sy = self._disp_h / self._img_h
        for x, y in pts:
            cx = float(x) * sx + self._off_x
            cy = float(y) * sy + self._off_y
            out.extend((cx, cy))
        return out

    def _bbox_to_canvas(self, x: int, y: int, x2: int, y2: int) -> tuple[float, float, float, float]:
        sx = self._disp_w / self._img_w
        sy = self._disp_h / self._img_h
        return (
            float(x) * sx + self._off_x,
            float(y) * sy + self._off_y,
            float(x2) * sx + self._off_x,
            float(y2) * sy + self._off_y,
        )

    def _canvas_rect_to_image_rect(
        self, cx1: float, cy1: float, cx2: float, cy2: float
    ) -> tuple[int, int, int, int] | None:
        """Clip canvas drag rect to the displayed image; return inclusive image pixel bounds."""
        if self.current_image is None or self._disp_w <= 0:
            return None
        x1, y1 = min(cx1, cx2), min(cy1, cy2)
        x2, y2 = max(cx1, cx2), max(cy1, cy2)
        x1 = max(x1, float(self._off_x))
        y1 = max(y1, float(self._off_y))
        x2 = min(x2, float(self._off_x + self._disp_w))
        y2 = min(y2, float(self._off_y + self._disp_h))
        if x1 >= x2 or y1 >= y2:
            return None
        ix1 = int((x1 - self._off_x) * self._img_w / self._disp_w)
        iy1 = int((y1 - self._off_y) * self._img_h / self._disp_h)
        ix2 = int((x2 - self._off_x) * self._img_w / self._disp_w)
        iy2 = int((y2 - self._off_y) * self._img_h / self._disp_h)
        ix1 = max(0, min(self._img_w - 1, ix1))
        iy1 = max(0, min(self._img_h - 1, iy1))
        ix2 = max(0, min(self._img_w - 1, ix2))
        iy2 = max(0, min(self._img_h - 1, iy2))
        if ix1 > ix2:
            ix1, ix2 = ix2, ix1
        if iy1 > iy2:
            iy1, iy2 = iy2, iy1
        return (ix1, iy1, ix2, iy2)

    def _contours_overlapping_image_rect(self, rx1: int, ry1: int, rx2: int, ry2: int) -> list[int]:
        """Contours whose axis-aligned bounding box overlaps the inclusive image rect."""
        out: list[int] = []
        for i, cnt in enumerate(self.contours):
            x, y, w, h = cv2.boundingRect(np.asarray(cnt))
            if x + w <= rx1 or x > rx2 or y + h <= ry1 or y > ry2:
                continue
            out.append(i)
        return out

    def _confirm_box_selection(self, _event=None) -> None:
        if not self._box_pending_indices:
            return
        if self.active_label is None:
            self.status_var.set("Arm a class first, then press Enter.")
            return
        slug = self.active_label
        indices = self._box_pending_indices
        n = len(indices)
        for i, idx in enumerate(indices):
            self.contour_labels[idx] = slug
            self._save_crop_for_contour(idx, slug, skip_review_refresh=(i < n - 1))
        self._box_pending_rect = None
        self._box_pending_indices = []
        self._update_labeled_count()
        self.redraw_canvas()
        self._review_refresh_list()
        self.status_var.set(f"Labeled {n} flake(s) as {self._class_title(slug)}")

    def _cancel_box_selection(self, _event=None) -> None:
        if self._box_pending_indices or self._box_pending_rect:
            self._box_pending_rect = None
            self._box_pending_indices = []
            self.redraw_canvas()
            self.status_var.set("Box selection cancelled.")

    def _safe_grab_release(self) -> None:
        try:
            self.canvas.grab_release()
        except tk.TclError:
            pass

    def _grab_for_box_drag(self) -> None:
        """Keep receiving B1-Motion / ButtonRelease after pointer leaves canvas (macOS / Tk)."""
        try:
            self.canvas.grab_set_global()
        except tk.TclError:
            try:
                self.canvas.grab_set()
            except tk.TclError:
                pass

    def _event_to_canvas_xy(self, event) -> tuple[float, float]:
        """Mouse position in canvas coordinates (handles global grab / off-widget release)."""
        w = getattr(event, "widget", None)
        if w is self.canvas:
            return float(event.x), float(event.y)
        try:
            return (
                float(event.x_root - self.canvas.winfo_rootx()),
                float(event.y_root - self.canvas.winfo_rooty()),
            )
        except tk.TclError:
            return float(getattr(event, "x", 0)), float(getattr(event, "y", 0))

    def _cancel_box_drag(self) -> None:
        if not self._box_dragging:
            return
        self._safe_grab_release()
        self._box_dragging = False
        self._box_drag_start = None
        self._box_drag_current = None
        self.redraw_canvas()

    def _on_canvas_configure(self, event):
        self.redraw_canvas()

    def _on_canvas_motion(self, event):
        if self._box_dragging:
            return
        if not self.contours or self.current_image is None:
            return
        pos = self._canvas_to_image(event.x, event.y)
        if pos is None:
            if self.hover_idx is not None:
                self.hover_idx = None
                self.hover_info_var.set("Hover: —")
                self.redraw_canvas()
            return
        idx = self._pick_contour_at(pos[0], pos[1])
        if idx != self.hover_idx:
            self.hover_idx = idx
            if idx is None:
                self.hover_info_var.set("Hover: —")
            else:
                lab = self.contour_labels[idx]
                t = "unlabeled" if lab is None else self._class_title(lab)
                self.hover_info_var.set(f"Hover: #{idx + 1} ({t})")
            self.redraw_canvas()

    def _on_canvas_leave(self, _event):
        # While box-dragging, ignore Leave — redraw / macOS often emit Leave mid-drag and
        # used to cancel the rubber band before we added grab + this guard.
        if self._box_dragging:
            return
        if self.hover_idx is not None:
            self.hover_idx = None
            self.hover_info_var.set("Hover: —")
            self.redraw_canvas()

    def _on_canvas_button_press(self, event):
        if int(self.box_select_var.get()) == 0:
            self._on_canvas_click(event)
            return
        if self.active_label is None:
            self.status_var.set("Arm a class (1–4) first, then drag a box.")
            return
        if not self.contours or self.current_image is None:
            return
        self._grab_for_box_drag()
        try:
            self.canvas.focus_set()
        except tk.TclError:
            pass
        mx, my = self._event_to_canvas_xy(event)
        self._box_dragging = True
        self._box_drag_start = (mx, my)
        self._box_drag_current = (mx, my)
        self.redraw_canvas()

    def _on_canvas_b1_motion(self, event):
        if not self._box_dragging:
            return
        mx, my = self._event_to_canvas_xy(event)
        self._box_drag_current = (mx, my)
        self.redraw_canvas()

    def _on_canvas_button_release(self, event):
        self._safe_grab_release()
        start = self._box_drag_start
        was_dragging = self._box_dragging
        self._box_dragging = False
        self._box_drag_start = None
        self._box_drag_current = None
        self.redraw_canvas()

        if not was_dragging or start is None:
            return
        if int(self.box_select_var.get()) == 0:
            return
        end = self._event_to_canvas_xy(event)

        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        if dx < BOX_SELECT_MIN_DRAG and dy < BOX_SELECT_MIN_DRAG:
            self._on_canvas_click(event)
            return

        ir = self._canvas_rect_to_image_rect(start[0], start[1], end[0], end[1])
        if ir is None:
            self.status_var.set("Drag inside the image.")
            return
        rx1, ry1, rx2, ry2 = ir
        indices = self._contours_overlapping_image_rect(rx1, ry1, rx2, ry2)
        if not indices:
            self.status_var.set("No flakes in that box.")
            return
        # Freeze selection — user reviews highlighted boxes, then presses Enter to confirm.
        self._box_pending_rect = (start[0], start[1], end[0], end[1])
        self._box_pending_indices = indices
        self.redraw_canvas()
        title = self._class_title(self.active_label) if self.active_label else "?"
        self.status_var.set(
            f"{len(indices)} flake(s) selected as {title} — press Enter to confirm, Esc to cancel"
        )

    def _on_canvas_click(self, event):
        if self.active_label is None:
            self.status_var.set("Choose a class (1–4) first, then click a flake.")
            return
        pos = self._canvas_to_image(event.x, event.y)
        if pos is None:
            return
        idx = self._pick_contour_at(pos[0], pos[1])
        if idx is None:
            return
        self.contour_labels[idx] = self.active_label
        self._save_crop_for_contour(idx, self.active_label)
        self._update_labeled_count()
        self.redraw_canvas()

    def _on_canvas_right_click(self, event):
        pos = self._canvas_to_image(event.x, event.y)
        if pos is None:
            return
        idx = self._pick_contour_at(pos[0], pos[1])
        if idx is None:
            return
        self.contour_labels[idx] = None
        old = self.contour_saved_paths.pop(idx, None)
        if old is not None:
            try:
                if old.exists():
                    old.unlink()
            except OSError:
                pass
        self._update_labeled_count()
        self.redraw_canvas()
        self._review_refresh_list()
        self.status_var.set(f"Cleared label on flake #{idx + 1}")

    def redraw_canvas(self):
        self.canvas.delete("all")
        if self.current_image is None:
            return

        self.right_placeholder.place_forget()
        self.canvas.pack(fill=tk.BOTH, expand=True)

        cw = max(2, self.canvas.winfo_width())
        ch = max(2, self.canvas.winfo_height())
        if cw < 10 or ch < 10:
            self.root.after(50, self.redraw_canvas)
            return

        self._compute_display_geometry(cw, ch)
        disp = self.current_image.resize((self._disp_w, self._disp_h), Image.Resampling.LANCZOS)

        # Build filled-contour overlay on the display-sized image.
        # Intentionally ignores hover-only mode so the fill is always visible when opacity > 0.
        opacity = float(self.overlay_opacity_var.get())
        if opacity > 0.0 and self.contours:
            disp_np = np.array(disp)
            overlay = np.zeros_like(disp_np)
            sx = self._disp_w / self._img_w
            sy = self._disp_h / self._img_h
            for i, cnt in enumerate(self.contours):
                slug = self.contour_labels[i]
                outline_key = self._outline_for_slug(slug)
                if slug is None:
                    if int(self.show_unlabeled_var.get()) == 0:
                        continue
                    fill_hex = SEGMENTATION_LINE_HEX
                else:
                    if outline_key is None or int(self.filter_vars[outline_key].get()) == 0:
                        continue
                    fill_hex = OUTLINE_HEX[outline_key]
                r = int(fill_hex[1:3], 16)
                g = int(fill_hex[3:5], 16)
                b = int(fill_hex[5:7], 16)
                pts = (np.asarray(cnt, dtype=np.float64).reshape(-1, 2) * [sx, sy]).astype(np.int32)
                cv2.fillPoly(overlay, [pts], (r, g, b))
            mask = overlay.sum(axis=2) > 0
            disp_np[mask] = (
                disp_np[mask] * (1.0 - opacity) + overlay[mask] * opacity
            ).astype(np.uint8)
            disp = Image.fromarray(disp_np)

        self._tk_image = ImageTk.PhotoImage(disp)
        self.canvas.create_image(
            self._off_x + self._disp_w // 2,
            self._off_y + self._disp_h // 2,
            image=self._tk_image,
        )

        img_rgb = np.array(self.current_image)
        hovered = self.hover_idx
        hover_only = int(self.hover_only_var.get()) != 0

        for i, cnt in enumerate(self.contours):
            slug = self.contour_labels[i]
            outline_key = self._outline_for_slug(slug)

            if slug is None:
                if int(self.show_unlabeled_var.get()) == 0:
                    continue
            else:
                if outline_key is None or int(self.filter_vars[outline_key].get()) == 0:
                    continue

            if hover_only and (hovered is None or i != hovered):
                continue

            is_hov = i == hovered
            lw = CONTOUR_WIDTH_HOVER if is_hov else CONTOUR_WIDTH
            flat = self._contour_to_canvas_flat(cnt)
            if len(flat) >= 6:
                self.canvas.create_polygon(
                    *flat,
                    outline=SEGMENTATION_LINE_HEX,
                    fill="",
                    width=lw,
                    smooth=False,
                )

            if slug is not None and outline_key is not None and int(self.filter_vars[outline_key].get()) != 0:
                cnt_i = np.asarray(cnt, dtype=np.int32).reshape(-1, 1, 2)
                x, y, x2, y2 = self.find_contour_bounded_box(img_rgb, cnt_i)
                bx1, by1, bx2, by2 = self._bbox_to_canvas(x, y, x2, y2)
                bbox_color = OUTLINE_HEX[outline_key]
                bw = BBOX_WIDTH_HOVER if is_hov else BBOX_WIDTH
                self.canvas.create_rectangle(bx1, by1, bx2, by2, outline=bbox_color, width=bw, fill="")

        preview_color = (
            OUTLINE_HEX[self._outline_for_slug(self.active_label)]
            if self.active_label is not None
            else "#38bdf8"
        )

        if self._box_dragging and self._box_drag_start and self._box_drag_current:
            x1, y1 = self._box_drag_start
            x2, y2 = self._box_drag_current

            # Highlight each contour whose bounding box overlaps the live drag rect.
            ir = self._canvas_rect_to_image_rect(x1, y1, x2, y2)
            if ir is not None:
                rx1, ry1, rx2, ry2 = ir
                for i, cnt in enumerate(self.contours):
                    cx, cy, cw, ch = cv2.boundingRect(np.asarray(cnt))
                    if cx + cw <= rx1 or cx > rx2 or cy + ch <= ry1 or cy > ry2:
                        continue
                    cnt_i = np.asarray(cnt, dtype=np.int32).reshape(-1, 1, 2)
                    bx1, by1, bx2, by2 = self._bbox_to_canvas(*self.find_contour_bounded_box(img_rgb, cnt_i))
                    self.canvas.create_rectangle(bx1, by1, bx2, by2, outline=preview_color, width=BBOX_WIDTH_HOVER, fill="")

            # Rubber-band rectangle on top.
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#38bdf8", width=2, dash=(5, 4))

        elif self._box_pending_rect is not None and self._box_pending_indices:
            x1, y1, x2, y2 = self._box_pending_rect

            # Draw highlighted bounding boxes for pending flakes.
            for idx in self._box_pending_indices:
                cnt_i = np.asarray(self.contours[idx], dtype=np.int32).reshape(-1, 1, 2)
                bx1, by1, bx2, by2 = self._bbox_to_canvas(*self.find_contour_bounded_box(img_rgb, cnt_i))
                self.canvas.create_rectangle(bx1, by1, bx2, by2, outline=preview_color, width=BBOX_WIDTH_HOVER, fill="")

            # Frozen selection outline.
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#38bdf8", width=2, dash=(5, 4))

    def find_contour_bounded_box(self, img: np.ndarray, contour):
        x, y, w, h = cv2.boundingRect(contour)
        scale = CROP_SCALE
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

    def next_image(self, event=None):
        if not self.images:
            return
        self.image_index = (self.image_index + 1) % len(self.images)
        self._refresh_current_image_from_index()

    def previous_image(self, event=None):
        if not self.images:
            return
        self.image_index = (self.image_index - 1) % len(self.images)
        self._refresh_current_image_from_index()

    def _review_on_folder_change(self, _event=None) -> None:
        self.review_index = 0
        self._review_refresh_list()

    def _review_refresh_list(self, _event=None) -> None:
        if not hasattr(self, "review_folder_var"):
            return
        dname = self.review_folder_var.get()
        folder = Path(self.save_folder_path) / dname
        if not folder.is_dir():
            self.review_paths = []
            self.review_index = 0
            self.review_idx_var.set("0 / 0")
            self._review_show_current()
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.review_paths = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
            key=lambda p: p.name.lower(),
        )
        if self.review_index >= len(self.review_paths):
            self.review_index = max(0, len(self.review_paths) - 1)
        self.review_idx_var.set(
            f"{self.review_index + 1} / {len(self.review_paths)}" if self.review_paths else "0 / 0"
        )
        self._review_show_current()

    def _on_review_canvas_configure(self, _event=None) -> None:
        self._review_show_current()

    def _review_show_current(self) -> None:
        if not hasattr(self, "review_canvas"):
            return
        self.review_canvas.delete("all")
        if not self.review_paths:
            self.review_path_var.set("—")
            self._review_tk_img = None
            return
        p = self.review_paths[self.review_index]
        self.review_path_var.set(p.name)
        self.review_idx_var.set(f"{self.review_index + 1} / {len(self.review_paths)}")
        try:
            im = Image.open(p).convert("RGB")
        except OSError:
            self.review_path_var.set(f"(unreadable) {p.name}")
            self._review_tk_img = None
            return
        cw = max(60, self.review_canvas.winfo_width())
        ch = max(60, self.review_canvas.winfo_height())
        iw, ih = im.size
        if iw <= 0 or ih <= 0:
            return
        scale = min(cw / iw, ch / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        disp = im.resize((nw, nh), Image.Resampling.LANCZOS)
        self._review_tk_img = ImageTk.PhotoImage(disp)
        self.review_canvas.create_image(cw // 2, ch // 2, image=self._review_tk_img)

    def _review_prev(self) -> None:
        if not self.review_paths:
            return
        self.review_index = (self.review_index - 1) % len(self.review_paths)
        self._review_show_current()

    def _review_next(self) -> None:
        if not self.review_paths:
            return
        self.review_index = (self.review_index + 1) % len(self.review_paths)
        self._review_show_current()

    def _review_move_to_slug(self, slug: str) -> None:
        if not self.review_paths:
            return
        p = self.review_paths[self.review_index]
        dest_dir = Path(self.save_folder_path) / CLASS_SAVE_DIRS[slug]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name
        if p.resolve() == dest.resolve():
            self.status_var.set("Already in that folder.")
            return
        try:
            if dest.exists():
                dest = dest_dir / f"{p.stem}_r{datetime.now().strftime('%H%M%S')}{p.suffix}"
            shutil.move(str(p), str(dest))
        except OSError as e:
            self.status_var.set(f"Move failed: {e}")
            return
        self.status_var.set(f"Moved to {CLASS_SAVE_DIRS[slug]}/{dest.name}")
        self._review_refresh_list()

    def choose_save_folder(self):
        path = filedialog.askdirectory()
        if not path:
            return
        self.save_folder_path = Path(path)
        self.save_folder_path.mkdir(parents=True, exist_ok=True)
        self._ensure_class_dirs()
        self.contour_saved_paths.clear()
        self.status_var.set(f"Save folder: {self.save_folder_path}")
        self._review_refresh_list()

    def on_resize(self, event):
        if self.current_image is not None:
            self.redraw_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    DataLabelingApp2(root)
    root.mainloop()

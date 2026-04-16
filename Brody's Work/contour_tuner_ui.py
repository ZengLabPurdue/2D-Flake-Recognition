#!/usr/bin/env python3
"""
Interactive UI to tune contour/edge detection parameters.
Run: python contour_tuner_ui.py
"""
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

# Tkinter
if sys.platform == "darwin":
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
else:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canny_filter_tuner import compute_filtered_canny
from edge_to_contour_methods import (
    _ensure_binary,
    force_close_open_chains,
    smart_connect_endpoints_bridged,
    _find_endpoints,
)
from preprocess import clahe, clahe_then_nlmeans
from batch_robust_contours_and_masks import (
    bin_image_by_2,
    compute_h_channel_edges,
    morph_extend_to_border,
    filter_nested_contours,
    load_contour_params,
    save_contour_params,
)
from edge_methods import (
    grayscale_combined,
    clahe_then_edges,
    unsharp_then_edges,
    background_subtract_then_edges,
    percentile_rescale_edges,
    gamma_rescale_edges,
    canny_edges,
    canny_low_threshold,
    canny_blur_then,
    morph_gradient_edges,
    laplacian_edges,
    log_edges,
    lab_edges,
    bilateral_then_edges,
    blur_then_edges,
)

DEFAULT_IMAGE_DIR = Path("images/x20_images")
MAX_DISPLAY_SIZE = 800
UPDATE_DELAY_MS = 400  # Debounce sliders

# Edge method registry: (key, display_name, requires_canny_params)
EDGE_METHODS = [
    ("canny_h", "Canny + H-channel (default)", True),
    ("canny_only", "Canny only", True),
    ("sobel", "Sobel", False),
    ("scharr", "Scharr", False),
    ("clahe_sobel", "CLAHE + Sobel", False),
    ("unsharp_sobel", "Unsharp + Sobel", False),
    ("laplacian", "Laplacian", False),
    ("log", "Laplacian of Gaussian", False),
    ("morph_gradient", "Morphological gradient", False),
    ("lab_sobel", "LAB L* + Sobel", False),
    ("bilateral_sobel", "Bilateral + Sobel", False),
    ("blur_sobel", "Blur + Sobel", False),
    ("gamma_rescale", "Gamma rescale", False),
    ("percentile_rescale", "Percentile rescale", False),
    ("background_subtract", "Background subtract", False),
]

# Method-specific options: method_key -> [(param_key, label, low, high, default, fmt), ...]
METHOD_OPTIONS = {
    "unsharp_sobel": [
        ("unsharp_sigma", "Unsharp sigma", 0.2, 3.0, 1.0, "%.2f"),
        ("unsharp_strength", "Unsharp strength", 0.5, 3.0, 1.5, "%.2f"),
    ],
    "background_subtract": [
        ("bg_subtract_sigma", "Blur sigma", 5, 71, 31, "%d"),
    ],
    "percentile_rescale": [
        ("percentile", "Percentile", 70.0, 99.0, 95.0, "%.1f"),
    ],
    "gamma_rescale": [
        ("gamma", "Gamma", 0.2, 1.5, 0.5, "%.2f"),
    ],
    "morph_gradient": [
        ("morph_ksize", "Kernel size", 3, 9, 3, "%d"),
    ],
    "laplacian": [
        ("laplacian_ksize", "Kernel size", 3, 7, 3, "%d"),
    ],
    "log": [
        ("log_sigma", "LoG sigma", 0.3, 3.0, 1.0, "%.2f"),
        ("log_ksize", "LoG ksize", 3, 11, 5, "%d"),
    ],
    "bilateral_sobel": [
        ("bilateral_d", "Diameter", 3, 15, 5, "%d"),
        ("bilateral_sigma_color", "Sigma color", 10, 100, 50, "%d"),
        ("bilateral_sigma_space", "Sigma space", 10, 100, 50, "%d"),
    ],
    "blur_sobel": [
        ("blur_sobel_sigma", "Blur sigma", 0.2, 2.0, 0.8, "%.2f"),
    ],
}


def _gray_from_edge_result(results: list, take_max: bool = True) -> np.ndarray:
    """Extract single grayscale edge map from edge_methods result (list of (name, img))."""
    out = None
    for _name, img in results:
        if img.ndim == 3:
            ch = img[:, :, 0]
        else:
            ch = img
        if out is None:
            out = ch.astype(np.float32)
        else:
            out = np.maximum(out, ch.astype(np.float32))
    if out is None:
        return np.zeros((1, 1), dtype=np.uint8)
    return np.clip(out, 0, 255).astype(np.uint8)


def _compute_edges_combined(work: np.ndarray, preprocessed: np.ndarray, params: dict) -> np.ndarray:
    """Compute edge map, combining multiple methods if edge_methods list is set.
    If edge_methods has 2+ methods, computes each and takes pixel-wise maximum (OR).
    Otherwise uses single edge_method. Returns uint8 0-255."""
    methods = params.get("edge_methods")
    if methods and isinstance(methods, (list, tuple)) and len(methods) > 0:
        combined = None
        for m in methods:
            p = dict(params)
            p["edge_method"] = m
            p.pop("edge_methods", None)
            edges = _compute_edges_by_method(work, preprocessed, p)
            e32 = edges.astype(np.float32)
            combined = np.maximum(combined, e32) if combined is not None else e32
        return np.clip(combined, 0, 255).astype(np.uint8)
    return _compute_edges_by_method(work, preprocessed, params)


def _compute_edges_by_method(work: np.ndarray, preprocessed: np.ndarray, params: dict) -> np.ndarray:
    """Compute edge map using selected method. Returns uint8 0-255."""
    method = params.get("edge_method", "canny_h")
    blur = params.get("blur_sigma", 0.6)
    cl, ch = params.get("canny_low", 10), params.get("canny_high", 50)
    edge_threshold = params.get("edge_threshold", 50)  # For non-Canny methods

    min_area = params.get("min_area", 0)

    # Combine two Canny runs: one with canny_low low (sensitive), one with canny_low high (conservative)
    combine_canny_low = params.get("combine_canny_low_range", False)
    cl_low = int(params.get("canny_low_low", 0))
    cl_high = int(params.get("canny_low_high", 100))

    def _canny_h_edges(canny_low_val: int):
        ec, _, _ = compute_filtered_canny(preprocessed, blur, canny_low_val, ch, min_area=min_area)
        eh = compute_h_channel_edges(work, blur_sigma=blur, canny_low=canny_low_val, canny_high=ch)
        return np.maximum(ec.astype(np.float32), eh.astype(np.float32))

    def _canny_only_edges(canny_low_val: int):
        e, _, _ = compute_filtered_canny(preprocessed, blur, canny_low_val, ch, min_area=min_area)
        return e.astype(np.float32)

    if method == "canny_h":
        if combine_canny_low:
            edges1 = _canny_h_edges(cl_low)
            edges2 = _canny_h_edges(cl_high)
            return np.clip(np.maximum(edges1, edges2), 0, 255).astype(np.uint8)
        edges_canny, _, _ = compute_filtered_canny(preprocessed, blur, cl, ch, min_area=min_area)
        edges_h = compute_h_channel_edges(work, blur_sigma=blur, canny_low=cl, canny_high=ch)
        return np.maximum(edges_canny.astype(np.float32), edges_h.astype(np.float32)).astype(np.uint8)
    elif method == "canny_only":
        if combine_canny_low:
            edges1 = _canny_only_edges(cl_low)
            edges2 = _canny_only_edges(cl_high)
            return np.clip(np.maximum(edges1, edges2), 0, 255).astype(np.uint8)
        edges, _, _ = compute_filtered_canny(preprocessed, blur, cl, ch, min_area=min_area)
        return edges

    # Use edge_methods on work (binned). They have their own preprocessing.
    u8 = work.astype(np.uint8) if work.dtype != np.uint8 else work
    if work.ndim == 2:
        u8 = np.stack([u8, u8, u8], axis=-1)

    if method == "sobel":
        results = grayscale_combined(u8)
        out = _gray_from_edge_result([r for r in results if r[0] == "sobel"])
    elif method == "scharr":
        results = grayscale_combined(u8)
        out = _gray_from_edge_result([r for r in results if r[0] == "scharr"])
    elif method == "clahe_sobel":
        results = clahe_then_edges(u8)
        out = _gray_from_edge_result([r for r in results if r[0] == "sobel"])
    elif method == "unsharp_sobel":
        sigma = params.get("unsharp_sigma", 1.0)
        strength = params.get("unsharp_strength", 1.5)
        results = unsharp_then_edges(u8, sigma=sigma, strength=strength)
        out = _gray_from_edge_result([r for r in results if r[0] == "sobel"])
    elif method == "laplacian":
        ksize = int(params.get("laplacian_ksize", 3)) | 1
        results = laplacian_edges(u8, ksize=ksize)
        out = _gray_from_edge_result(results)
    elif method == "log":
        sigma = params.get("log_sigma", 1.0)
        ksize = int(params.get("log_ksize", 5)) | 1
        results = log_edges(u8, sigma=sigma, ksize=ksize)
        out = _gray_from_edge_result(results)
    elif method == "morph_gradient":
        ksize = int(params.get("morph_ksize", 3)) | 1
        results = morph_gradient_edges(u8, ksize=ksize)
        out = _gray_from_edge_result(results)
    elif method == "lab_sobel":
        results = lab_edges(u8)
        out = _gray_from_edge_result([r for r in results if r[0] == "sobel"])
    elif method == "bilateral_sobel":
        d = int(params.get("bilateral_d", 5)) | 1
        sc = int(params.get("bilateral_sigma_color", 50))
        ss = int(params.get("bilateral_sigma_space", 50))
        results = bilateral_then_edges(u8, d=d, sigma_color=sc, sigma_space=ss)
        out = _gray_from_edge_result([r for r in results if r[0] == "sobel"])
    elif method == "blur_sobel":
        sigma = params.get("blur_sobel_sigma", 0.8)
        results = blur_then_edges(u8, sigma=sigma)
        out = _gray_from_edge_result([r for r in results if r[0] == "sobel"])
    elif method == "gamma_rescale":
        gamma = params.get("gamma", 0.5)
        results = gamma_rescale_edges(u8, gamma=gamma)
        out = _gray_from_edge_result(results)
    elif method == "percentile_rescale":
        pct = params.get("percentile", 95.0)
        results = percentile_rescale_edges(u8, pct=pct)
        out = _gray_from_edge_result(results)
    elif method == "background_subtract":
        sigma = int(params.get("bg_subtract_sigma", 31)) | 1
        results = background_subtract_then_edges(u8, sigma=sigma)
        out = _gray_from_edge_result(results)
    else:
        edges, _, _ = compute_filtered_canny(preprocessed, blur, cl, ch, min_area=min_area)
        return edges

    # Non-Canny methods produce 0-255; threshold for stronger edges if desired
    if edge_threshold > 0:
        _, out = cv2.threshold(out, edge_threshold, 255, cv2.THRESH_BINARY)
    return out


def _compute_edge_comparison(work: np.ndarray, preprocessed: np.ndarray, params: dict) -> np.ndarray:
    """Compute multiple edge methods and layout in 2x3 grid for comparison."""
    u8 = work.astype(np.uint8) if work.dtype != np.uint8 else work
    if u8.ndim == 2:
        u8 = np.stack([u8, u8, u8], axis=-1)
    blur = params.get("blur_sigma", 0.6)
    cl, ch = params.get("canny_low", 10), params.get("canny_high", 50)

    def to_rgb(x):
        if x.ndim == 2:
            return np.stack([x, x, x], axis=-1)
        return x

    def canny_h():
        ec, _, _ = compute_filtered_canny(preprocessed, blur, cl, ch, min_area=0)
        eh = compute_h_channel_edges(work, blur_sigma=blur, canny_low=cl, canny_high=ch)
        return np.maximum(ec, eh)

    def sobel():
        r = grayscale_combined(u8)
        return _gray_from_edge_result([x for x in r if x[0] == "sobel"])

    def laplacian():
        return _gray_from_edge_result(laplacian_edges(u8))

    def log():
        return _gray_from_edge_result(log_edges(u8))

    def morph():
        return _gray_from_edge_result(morph_gradient_edges(u8))

    def clahe_sob():
        return _gray_from_edge_result([x for x in clahe_then_edges(u8) if x[0] == "sobel"])

    tiles = [
        ("Canny+H", to_rgb(canny_h())),
        ("Sobel", to_rgb(sobel())),
        ("Laplacian", to_rgb(laplacian())),
        ("LoG", to_rgb(log())),
        ("Morph grad", to_rgb(morph())),
        ("CLAHE+Sobel", to_rgb(clahe_sob())),
    ]
    h, w = tiles[0][1].shape[:2]
    pad = 6
    label_h = 22
    cell_h, cell_w = h // 2, w // 3
    grid_h = 2 * cell_h + label_h * 2 + pad * 3
    grid_w = 3 * cell_w + pad * 4
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid[:] = 50
    for i, (label, img) in enumerate(tiles):
        row, col = i // 3, i % 3
        y_label = row * (cell_h + label_h + pad) + label_h - 4
        y_img = row * (cell_h + label_h + pad) + label_h + pad
        x0 = col * (cell_w + pad) + pad
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        grid[y_img : y_img + cell_h, x0 : x0 + cell_w] = resized
        cv2.putText(grid, label, (x0, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return grid


def run_pipeline(img: np.ndarray, params: dict) -> dict:
    """Run contour pipeline with given params. Returns dict with overlay, edges, masked_bg, etc.
    Uses ContourPipeline for the main flow; adds edge_comparison for UI."""
    from pipeline import ContourPipeline

    pipeline = ContourPipeline()
    result = pipeline.run(img, params, return_edges=True, return_binary=False)
    work = result["work"]
    preprocessed = result["preprocessed"]

    edge_comparison = _compute_edge_comparison(work, preprocessed, params)
    mask = result["mask_shapes"]
    mask_3ch = np.stack([mask, mask, mask], axis=-1) if mask.ndim == 2 else mask

    return {
        "original": result["original"],
        "edges": result["edges"],
        "overlay": result["overlay"],
        "mask_shapes": mask_3ch,
        "masked_background": result["masked_background"],
        "edge_comparison": edge_comparison,
        "n_contours": result["n_contours"],
        "n_edges_px": result["n_edges_px"],
    }


def resize_for_display(img: np.ndarray, max_size: int = MAX_DISPLAY_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


class ContourTunerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Contour Edge Detection Tuner")
        self.root.geometry("1200x850")
        self.root.minsize(900, 600)

        self.img = None
        self.img_path = None
        self.result = None
        self.params = {}
        self.update_timer = None
        self.thread = None

        self._build_ui()
        self._load_defaults()
        self._try_load_first_image()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(top, text="Open Image...", command=self._open_image).pack(side=tk.LEFT, padx=(0, 8))
        self.path_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=280)
        paned.add(left, weight=0)

        scroll = ttk.Scrollbar(left)
        canvas = tk.Canvas(left, yscrollcommand=scroll.set, highlightthickness=0)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=canvas.yview)

        params_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=params_frame, anchor=tk.NW)
        self.params_frame = params_frame
        self._scroll_canvas = canvas

        def make_slider(parent, label, key, low, high, default, fmt="%.2f"):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            self.params[key] = var
            scaletype = float if isinstance(default, float) else int
            s = ttk.Scale(row, from_=low, to=high, variable=var, orient=tk.HORIZONTAL, length=120, command=lambda _: self._schedule_update())
            s.pack(side=tk.LEFT, padx=4)
            val_label = ttk.Label(row, text="", width=8)
            val_label.pack(side=tk.LEFT)

            def upd():
                v = var.get()
                if scaletype == int:
                    val_label.config(text=str(int(v)))
                else:
                    val_label.config(text=fmt % v)
            var.trace_add("write", lambda *a: upd())
            upd()
            return var

        def make_check(parent, label, key, default=True):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=4)
            var = tk.BooleanVar(value=default)
            self.params[key] = var
            ttk.Checkbutton(row, text=label, variable=var, command=self._schedule_update).pack(side=tk.LEFT)
            return var

        ttk.Label(params_frame, text="Edge detection", font=("", 11, "bold")).pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(params_frame, text="Method").pack(anchor=tk.W)
        self._edge_display_to_key = {m[1]: m[0] for m in EDGE_METHODS}
        self.edge_method_var = tk.StringVar(value=EDGE_METHODS[0][1])
        edge_combo = ttk.Combobox(
            params_frame, textvariable=self.edge_method_var, state="readonly",
            values=[m[1] for m in EDGE_METHODS], width=22
        )
        edge_combo.pack(fill=tk.X, pady=2)
        self.params["edge_method_var"] = self.edge_method_var
        make_slider(params_frame, "Blur sigma", "blur_sigma", 0.1, 2.0, 0.6)
        make_slider(params_frame, "Canny low", "canny_low", 0, 100, 10, "%d")
        make_slider(params_frame, "Canny high", "canny_high", 50, 200, 50, "%d")
        make_slider(params_frame, "Min area (Canny)", "min_area", 0, 100, 0, "%d")
        make_slider(params_frame, "Edge threshold", "edge_threshold", 0, 150, 50, "%d")
        ttk.Label(params_frame, text="(0=no threshold for non-Canny)", font=("", 8)).pack(anchor=tk.W)
        ttk.Label(params_frame, text="Preprocessing").pack(anchor=tk.W)
        self.preprocessing_var = tk.StringVar(value="full")
        preproc_combo = ttk.Combobox(
            params_frame, textvariable=self.preprocessing_var, state="readonly",
            values=["Full (CLAHE + denoise)", "CLAHE only", "None"], width=20
        )
        preproc_combo.pack(fill=tk.X, pady=2)
        preproc_combo.bind("<<ComboboxSelected>>", lambda e: self._schedule_update())
        self._preproc_display_to_key = {
            "Full (CLAHE + denoise)": "full",
            "CLAHE only": "clahe",
            "None": "none",
        }

        ttk.Label(params_frame, text="Method-specific", font=("", 11, "bold")).pack(anchor=tk.W, pady=(12, 4))
        self.method_options_frame = ttk.Frame(params_frame)
        self.method_options_frame.pack(fill=tk.X, pady=(0, 8))
        self._method_option_vars = {}  # param_key -> var
        edge_combo.bind("<<ComboboxSelected>>", lambda e: (self._rebuild_method_options(), self._schedule_update()))
        self._rebuild_method_options()

        ttk.Label(params_frame, text="Gap closing", font=("", 11, "bold")).pack(anchor=tk.W, pady=(12, 4))
        make_slider(params_frame, "Bridge gap factor", "bridge_gap_factor", 0.05, 0.35, 0.15)
        make_slider(params_frame, "Force-close factor", "force_close_factor", 0.1, 0.5, 0.3)
        make_slider(params_frame, "Line thickness", "line_thickness", 2, 8, 4, "%d")
        make_slider(params_frame, "Close kernel div", "close_kernel_divisor", 50, 200, 100, "%d")

        ttk.Label(params_frame, text="Border / morph", font=("", 11, "bold")).pack(anchor=tk.W, pady=(12, 4))
        make_slider(params_frame, "Morph margin", "morph_margin", 0, 30, 15, "%d")
        make_slider(params_frame, "Corner radius", "corner_radius", 0, 8, 2, "%d")

        ttk.Label(params_frame, text="Other", font=("", 11, "bold")).pack(anchor=tk.W, pady=(12, 4))
        make_check(params_frame, "2×2 binning", "binning", True)

        ttk.Button(params_frame, text="Update Now", command=self._update_now_and_save).pack(pady=12)

        params_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        view_frame = ttk.Frame(right)
        view_frame.pack(fill=tk.X, pady=(0, 8))
        self.view_var = tk.StringVar(value="overlay")
        for v, lbl in [
            ("overlay", "Overlay"), ("masked_background", "Masked BG"),
            ("edges", "Edges"), ("mask_shapes", "Mask"), ("original", "Original"),
            ("edge_comparison", "Edge Compare"),
        ]:
            ttk.Radiobutton(view_frame, text=lbl, variable=self.view_var, value=v, command=self._refresh_display).pack(side=tk.LEFT, padx=(0, 12))

        self.canvas_img = tk.Canvas(right, bg="#333", highlightthickness=0)
        self.canvas_img.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Load an image to start")
        ttk.Label(main, textvariable=self.status_var).pack(anchor=tk.W, pady=4)

    def _rebuild_method_options(self):
        """Rebuild method-specific options based on selected edge method."""
        for w in self.method_options_frame.winfo_children():
            w.destroy()
        self._method_option_vars.clear()
        display = self.edge_method_var.get()
        method = self._edge_display_to_key.get(display, "canny_h")
        opts = METHOD_OPTIONS.get(method, [])
        if not opts:
            ttk.Label(self.method_options_frame, text="(no options)", font=("", 9), foreground="gray").pack(anchor=tk.W)
            return
        for param_key, label, low, high, default, fmt in opts:
            row = ttk.Frame(self.method_options_frame)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            self.params[param_key] = var
            self._method_option_vars[param_key] = var
            scaletype = float if "f" in fmt or "." in fmt else int
            s = ttk.Scale(row, from_=low, to=high, variable=var, orient=tk.HORIZONTAL, length=100, command=lambda _: self._schedule_update())
            s.pack(side=tk.LEFT, padx=4)
            val_label = ttk.Label(row, text="", width=6)
            val_label.pack(side=tk.LEFT)

            def on_change(v=var, vl=val_label, st=scaletype, f=fmt):
                val = v.get()
                if st == int:
                    vl.config(text=str(int(val)))
                else:
                    vl.config(text=f % val)
                self._schedule_update()
            var.trace_add("write", lambda *a: on_change())
            on_change()
        self.params_frame.update_idletasks()
        if hasattr(self, "_scroll_canvas") and self._scroll_canvas:
            self._scroll_canvas.config(scrollregion=self._scroll_canvas.bbox("all"))

    def _load_defaults(self):
        """Load from robust_contours_settings.json if exists, else use built-in defaults."""
        loaded = load_contour_params()
        p = self.params
        em = loaded.get("edge_method", "canny_h")
        display = next((m[1] for m in EDGE_METHODS if m[0] == em), EDGE_METHODS[0][1])
        self.edge_method_var.set(display)
        self._rebuild_method_options()
        opts = METHOD_OPTIONS.get(em, [])
        for param_key, _lbl, _lo, _hi, default, _fmt in opts:
            if param_key in p:
                p[param_key].set(loaded.get(param_key, default))
        preproc = loaded.get("preprocessing", "full")
        preproc_display = next((k for k, v in self._preproc_display_to_key.items() if v == preproc), "Full (CLAHE + denoise)")
        self.preprocessing_var.set(preproc_display)
        p["blur_sigma"].set(loaded.get("blur_sigma", 0.6))
        p["canny_low"].set(loaded.get("canny_low", 10))
        p["canny_high"].set(loaded.get("canny_high", 50))
        p["min_area"].set(loaded.get("min_area", 0))
        p["edge_threshold"].set(loaded.get("edge_threshold", 50))
        p["bridge_gap_factor"].set(loaded.get("bridge_gap_factor", 0.15))
        p["force_close_factor"].set(loaded.get("force_close_factor", 0.3))
        p["line_thickness"].set(loaded.get("line_thickness", 4))
        p["close_kernel_divisor"].set(loaded.get("close_kernel_divisor", 100))
        p["morph_margin"].set(loaded.get("morph_margin", 15))
        p["corner_radius"].set(loaded.get("corner_radius", 2))
        p["binning"].set(loaded.get("binning", True))

    def _try_load_first_image(self):
        if DEFAULT_IMAGE_DIR.exists():
            files = sorted(DEFAULT_IMAGE_DIR.glob("*.png")) + sorted(DEFAULT_IMAGE_DIR.glob("*.jpg"))
            if files:
                self._load_image(str(files[0]))

    def _open_image(self):
        path = filedialog.askopenfilename(
            initialdir=str(DEFAULT_IMAGE_DIR) if DEFAULT_IMAGE_DIR.exists() else ".",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff"), ("All", "*.*")]
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        try:
            self.img = np.array(Image.open(path))
            self.img_path = path
            self.path_var.set(Path(path).name)
            self._update_now(save_to_json=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _schedule_update(self):
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(UPDATE_DELAY_MS, lambda: self._update_now(save_to_json=False))

    def _update_now_and_save(self):
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        self._update_now(save_to_json=True)

    def _get_params_dict(self) -> dict:
        p = self.params
        display = p.get("edge_method_var", self.edge_method_var).get()
        edge_method = self._edge_display_to_key.get(display, "canny_h")
        preproc_display = self.preprocessing_var.get()
        preproc = self._preproc_display_to_key.get(preproc_display, "full")
        out = {
            "edge_method": edge_method,
            "preprocessing": preproc,
            "blur_sigma": p["blur_sigma"].get(),
            "canny_low": int(p["canny_low"].get()),
            "canny_high": int(p["canny_high"].get()),
            "min_area": int(p["min_area"].get()),
            "edge_threshold": int(p["edge_threshold"].get()),
            "bridge_gap_factor": p["bridge_gap_factor"].get(),
            "force_close_factor": p["force_close_factor"].get(),
            "line_thickness": int(p["line_thickness"].get()),
            "close_kernel_divisor": int(p["close_kernel_divisor"].get()),
            "morph_margin": int(p["morph_margin"].get()),
            "corner_radius": int(p["corner_radius"].get()),
            "binning": p["binning"].get(),
        }
        for param_key in self._method_option_vars:
            if param_key in p:
                v = p[param_key].get()
                out[param_key] = int(v) if abs(v - round(v)) < 1e-6 else float(v)
        return out

    def _update_now(self, save_to_json: bool = False):
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        if self.img is None:
            return
        self.status_var.set("Processing...")
        self.root.update_idletasks()

        def run():
            try:
                res = run_pipeline(self.img.copy(), self._get_params_dict())
                self.root.after(0, lambda: self._on_result(res, save_to_json))
            except Exception as e:
                self.root.after(0, lambda: self._on_error(str(e)))

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def _on_result(self, res: dict, save_to_json: bool = False):
        self.result = res
        if save_to_json:
            try:
                save_contour_params(self._get_params_dict())
                self.status_var.set(
                    f"Contours: {res['n_contours']}  |  Edge pixels: {res['n_edges_px']:,}  |  Settings saved"
                )
            except Exception:
                self.status_var.set(
                    f"Contours: {res['n_contours']}  |  Edge pixels: {res['n_edges_px']:,}"
                )
        else:
            self.status_var.set(
                f"Contours: {res['n_contours']}  |  Edge pixels: {res['n_edges_px']:,}"
            )
        self._refresh_display()

    def _on_error(self, err: str):
        self.status_var.set(f"Error: {err}")
        messagebox.showerror("Error", err)

    def _refresh_display(self):
        if not self.result:
            return
        view = self.view_var.get()
        img = self.result.get(view)
        if img is None:
            return
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        disp = resize_for_display(img)
        ph, pw = disp.shape[:2]
        pil = Image.fromarray(disp)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas_img.delete("all")
        self.canvas_img.create_image(pw // 2, ph // 2, image=self.photo, tags="img")
        self.canvas_img.config(width=pw, height=ph)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ContourTunerApp()
    app.run()

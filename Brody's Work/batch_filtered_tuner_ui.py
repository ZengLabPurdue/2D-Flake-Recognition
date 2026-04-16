#!/usr/bin/env python3
"""
Interactive UI to tune Batch Filtered 2x2 Overlay parameters.
Same layout as contour tuner, but saves to batch_filtered_settings.json
and is used by batch_filtered_sensitive_overlays_2x2.py.

Run: python batch_filtered_tuner_ui.py
"""
import sys
from pathlib import Path

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import from contour tuner - we reuse the same pipeline and UI structure
from contour_tuner_ui import (
    EDGE_METHODS,
    METHOD_OPTIONS,
    DEFAULT_IMAGE_DIR,
    MAX_DISPLAY_SIZE,
    UPDATE_DELAY_MS,
    run_pipeline,
    resize_for_display,
)
from batch_filtered_sensitive_overlays_2x2 import (
    load_filtered_overlay_params,
    save_filtered_overlay_params,
)

# Tkinter
if sys.platform == "darwin":
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
else:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

import threading
import numpy as np
from PIL import Image, ImageTk


class BatchFilteredTunerApp:
    """Tuner for batch_filtered_sensitive_overlays_2x2 - same UI as contour tuner."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Batch Filtered 2x2 Overlay Tuner")
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

        ttk.Label(params_frame, text="Edge detection", font=("", 11, "bold")).pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(params_frame, text="Methods (combine with OR):", font=("", 9)).pack(anchor=tk.W)
        self._edge_check_vars = {}  # method_key -> BooleanVar
        edge_frame = ttk.Frame(params_frame)
        edge_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        edge_inner = ttk.Frame(edge_frame)
        edge_inner.pack(fill=tk.X)
        for method_key, display_name, _ in EDGE_METHODS:
            var = tk.BooleanVar(value=(method_key == "canny_h"))
            self._edge_check_vars[method_key] = var
            def on_check(*a):
                self._schedule_update()
                self._rebuild_method_options()
            cb = ttk.Checkbutton(edge_inner, text=display_name, variable=var, command=on_check)
            cb.pack(anchor=tk.W, padx=(0, 4))
            var.trace_add("write", on_check)
        self.params["_edge_checks"] = self._edge_check_vars
        make_slider(params_frame, "Blur sigma", "blur_sigma", 0, 2.0, 0.6)
        self.combine_canny_var = tk.BooleanVar(value=False)
        cb_combine = ttk.Checkbutton(
            params_frame, text="Combine canny low (low + high)", variable=self.combine_canny_var,
            command=lambda: (self._schedule_update(), self._toggle_canny_low_sliders())
        )
        cb_combine.pack(anchor=tk.W, pady=2)
        self.combine_canny_var.trace_add("write", lambda *a: self._toggle_canny_low_sliders())
        self._canny_low_frame = ttk.Frame(params_frame)
        make_slider(self._canny_low_frame, "Canny low (down)", "canny_low_low", 0, 50, 0, "%d")
        make_slider(self._canny_low_frame, "Canny low (up)", "canny_low_high", 0, 100, 100, "%d")
        self._canny_low_frame.pack(fill=tk.X, pady=2)
        make_slider(params_frame, "Canny low", "canny_low", 0, 100, 10, "%d")
        ttk.Label(params_frame, text="(used when combine off)", font=("", 8)).pack(anchor=tk.W)
        make_slider(params_frame, "Canny high", "canny_high", 0, 200, 50, "%d")
        make_slider(params_frame, "Min area (Canny)", "min_area", 0, 100, 0, "%d")
        make_slider(params_frame, "Edge threshold", "edge_threshold", 0, 150, 50, "%d")
        ttk.Label(params_frame, text="(0=no threshold for non-Canny)", font=("", 8)).pack(anchor=tk.W)
        make_slider(params_frame, "Min area (contour)", "min_area_contour", 0, 1000, 200, "%d")
        self.filter_nested_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            params_frame, text="Filter nested contours", variable=self.filter_nested_var,
            command=self._schedule_update
        ).pack(anchor=tk.W, pady=2)
        ttk.Label(params_frame, text="(remove contours inside larger ones)", font=("", 8)).pack(anchor=tk.W)
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

        ttk.Label(params_frame, text="Method-specific (first checked)", font=("", 11, "bold")).pack(anchor=tk.W, pady=(12, 4))
        self.method_options_frame = ttk.Frame(params_frame)
        self.method_options_frame.pack(fill=tk.X, pady=(0, 8))
        self._method_option_vars = {}
        self._rebuild_method_options()

        ttk.Label(params_frame, text="Gap closing", font=("", 11, "bold")).pack(anchor=tk.W, pady=(12, 4))
        make_slider(params_frame, "Bridge gap factor", "bridge_gap_factor", 0, 0.35, 0.15)
        make_slider(params_frame, "Force-close factor", "force_close_factor", 0, 0.5, 0.3)
        make_slider(params_frame, "Line thickness", "line_thickness", 0, 8, 4, "%d")
        make_slider(params_frame, "Close kernel div", "close_kernel_divisor", 0, 200, 100, "%d")

        ttk.Label(params_frame, text="Border / morph", font=("", 11, "bold")).pack(anchor=tk.W, pady=(12, 4))
        make_slider(params_frame, "Morph margin", "morph_margin", 0, 30, 15, "%d")
        make_slider(params_frame, "Corner radius", "corner_radius", 0, 8, 2, "%d")

        ttk.Label(params_frame, text="2x2 binning (always on)", font=("", 9), foreground="gray").pack(anchor=tk.W, pady=4)
        self.params["binning"] = tk.BooleanVar(value=True)

        ttk.Label(params_frame, text="Color filters (contour selection)", font=("", 11, "bold")).pack(anchor=tk.W, pady=(12, 4))
        ttk.Label(params_frame, text="Keep contours passing ANY checked filter", font=("", 9)).pack(anchor=tk.W)
        self.filter_yellow_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Yellow (HSV V > 70)", variable=self.filter_yellow_var, command=self._schedule_update).pack(anchor=tk.W, padx=(0, 4))
        self.filter_blue_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Blue (HSL H > 100°)", variable=self.filter_blue_var, command=self._schedule_update).pack(anchor=tk.W, padx=(0, 4))
        self.filter_good_flake_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Good flake (RGB G < 130)", variable=self.filter_good_flake_var, command=self._schedule_update).pack(anchor=tk.W, padx=(0, 4))

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

        self.status_var = tk.StringVar(value="Load an image to start (settings for batch_filtered_sensitive_overlays_2x2)")
        ttk.Label(main, textvariable=self.status_var).pack(anchor=tk.W, pady=4)

    def _toggle_canny_low_sliders(self):
        if getattr(self, "combine_canny_var", None) and getattr(self, "_canny_low_frame", None):
            if self.combine_canny_var.get():
                self._canny_low_frame.pack(fill=tk.X, pady=2)
            else:
                self._canny_low_frame.pack_forget()

    def _get_first_checked_method(self) -> str:
        """Return the first checked edge method key, or canny_h if none."""
        for method_key, _display, _ in EDGE_METHODS:
            if self._edge_check_vars.get(method_key, tk.BooleanVar(value=False)).get():
                return method_key
        return "canny_h"

    def _rebuild_method_options(self):
        for w in self.method_options_frame.winfo_children():
            w.destroy()
        self._method_option_vars.clear()
        method = self._get_first_checked_method()
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
        loaded = load_filtered_overlay_params()
        p = self.params
        edge_methods = loaded.get("edge_methods")
        if edge_methods and isinstance(edge_methods, (list, tuple)):
            for method_key in self._edge_check_vars:
                self._edge_check_vars[method_key].set(method_key in edge_methods)
        else:
            em = loaded.get("edge_method", "canny_h")
            for method_key in self._edge_check_vars:
                self._edge_check_vars[method_key].set(method_key == em)
        self._rebuild_method_options()
        opts = METHOD_OPTIONS.get(self._get_first_checked_method(), [])
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
        p["min_area_contour"].set(loaded.get("min_area_contour", 200))
        self.filter_nested_var.set(loaded.get("filter_nested_contours", False))
        self.filter_yellow_var.set(loaded.get("filter_yellow", False))
        self.filter_blue_var.set(loaded.get("filter_blue", False))
        self.filter_good_flake_var.set(loaded.get("filter_good_flake", False))
        p["bridge_gap_factor"].set(loaded.get("bridge_gap_factor", 0.15))
        p["force_close_factor"].set(loaded.get("force_close_factor", 0.3))
        p["line_thickness"].set(loaded.get("line_thickness", 4))
        p["close_kernel_divisor"].set(loaded.get("close_kernel_divisor", 100))
        p["morph_margin"].set(loaded.get("morph_margin", 15))
        p["corner_radius"].set(loaded.get("corner_radius", 2))
        p["binning"].set(True)
        self.combine_canny_var.set(loaded.get("combine_canny_low_range", False))
        if "canny_low_low" in p:
            p["canny_low_low"].set(loaded.get("canny_low_low", 0))
        if "canny_low_high" in p:
            p["canny_low_high"].set(loaded.get("canny_low_high", 100))
        self._toggle_canny_low_sliders()

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
        checked = [k for k in self._edge_check_vars if self._edge_check_vars[k].get()]
        edge_methods = checked if checked else ["canny_h"]
        preproc_display = self.preprocessing_var.get()
        preproc = self._preproc_display_to_key.get(preproc_display, "full")
        out = {
            "edge_method": edge_methods[0],
            "edge_methods": edge_methods,
            "preprocessing": preproc,
            "blur_sigma": p["blur_sigma"].get(),
            "combine_canny_low_range": self.combine_canny_var.get(),
            "canny_low_low": int(p["canny_low_low"].get()) if "canny_low_low" in p else 0,
            "canny_low_high": int(p["canny_low_high"].get()) if "canny_low_high" in p else 100,
            "canny_low": int(p["canny_low"].get()),
            "canny_high": int(p["canny_high"].get()),
            "min_area": int(p["min_area"].get()),
            "edge_threshold": int(p["edge_threshold"].get()),
            "min_area_contour": int(p["min_area_contour"].get()),
            "filter_nested_contours": self.filter_nested_var.get(),
            "filter_yellow": self.filter_yellow_var.get(),
            "filter_blue": self.filter_blue_var.get(),
            "filter_good_flake": self.filter_good_flake_var.get(),
            "yellow_v_min": 70,
            "blue_h_min": 100,
            "good_flake_g_max": 130,
            "bridge_gap_factor": p["bridge_gap_factor"].get(),
            "force_close_factor": p["force_close_factor"].get(),
            "line_thickness": int(p["line_thickness"].get()),
            "close_kernel_divisor": int(p["close_kernel_divisor"].get()),
            "morph_margin": int(p["morph_margin"].get()),
            "corner_radius": int(p["corner_radius"].get()),
            "binning": True,
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
                params = self._get_params_dict()
                params["binning"] = True  # Always 2x2 for this tuner
                res = run_pipeline(self.img.copy(), params)
                self.root.after(0, lambda: self._on_result(res, save_to_json))
            except Exception as e:
                self.root.after(0, lambda: self._on_error(str(e)))

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def _on_result(self, res: dict, save_to_json: bool = False):
        self.result = res
        if save_to_json:
            try:
                save_filtered_overlay_params(self._get_params_dict())
                self.status_var.set(
                    f"Contours: {res['n_contours']}  |  Edge pixels: {res['n_edges_px']:,}  |  Saved to batch_filtered_settings.json"
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
    app = BatchFilteredTunerApp()
    app.run()

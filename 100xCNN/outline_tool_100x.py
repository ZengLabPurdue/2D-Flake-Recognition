#!/usr/bin/env python3
"""Tkinter UI for reviewing 100x flake masks and exporting YOLO labels."""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog, ttk

from export_100x_yolo_seg_labels import DEFAULT_OUT_DIR, IMAGE_EXTS, polygon_to_yolo_line


BASE = Path(__file__).resolve().parent
IMG_DIR = BASE / "images"
ANN_DIR = BASE / "contour_annotations"
MIN_POLY_POINTS = 3
LABEL_GOOD = "good"
LABEL_BAD = "bad"
CLASS_IDS = {LABEL_GOOD: 0, LABEL_BAD: 1}
SAM_CKPT_CANDIDATES = [
    BASE / "sam_vit_b_01ec64.pth",
    BASE.parent / "Brody's Work" / "sam_vit_b_01ec64.pth",
]


class LabeledContour(TypedDict):
    points: list[tuple[int, int]]
    label: str


def image_paths() -> list[Path]:
    if not IMG_DIR.exists():
        return []
    return sorted(p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def ann_path(img_path: Path) -> Path:
    return ANN_DIR / f"{img_path.stem}.json"


def normalize_label(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in CLASS_IDS:
            return lowered
    return LABEL_GOOD


def contour_points(contour: LabeledContour | Any) -> list[tuple[int, int]]:
    if isinstance(contour, dict) and "points" in contour:
        pts = contour["points"]
        return [(int(x), int(y)) for x, y in pts]
    return [(int(x), int(y)) for x, y in contour]


def contour_label(contour: LabeledContour | Any) -> str:
    if isinstance(contour, dict) and "label" in contour:
        return normalize_label(contour["label"])
    return LABEL_GOOD


def load_annotations(img_path: Path) -> list[LabeledContour]:
    path = ann_path(img_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    contours_out: list[LabeledContour] = []
    for entry in data.get("contours", []):
        if isinstance(entry, dict) and "points" in entry:
            pts = [(int(x), int(y)) for x, y in entry["points"]]
            contours_out.append({"points": pts, "label": normalize_label(entry.get("label"))})
            continue
        pts = [(int(x), int(y)) for x, y in entry]
        contours_out.append({"points": pts, "label": LABEL_GOOD})
    return contours_out


def save_annotations(img_path: Path, polygons: list[LabeledContour]) -> Path:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    path = ann_path(img_path)
    serialized = []
    for contour in polygons:
        serialized.append(
            {
                "label": contour_label(contour),
                "points": [[int(x), int(y)] for x, y in contour_points(contour)],
            }
        )
    path.write_text(
        json.dumps({"image": img_path.name, "contours": serialized}, indent=2),
        encoding="utf-8",
    )
    return path


def export_yolo(img_path: Path, polygons: list[LabeledContour], out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    if not polygons:
        raise ValueError("No polygons to export")
    with Image.open(img_path) as img:
        width, height = img.size
    lines = []
    for contour in polygons:
        pts = contour_points(contour)
        label = contour_label(contour)
        cls_id = CLASS_IDS[label]
        line = polygon_to_yolo_line(pts, width=width, height=height, class_id=cls_id, decimals=6)
        if line:
            lines.append(line)
    if not lines:
        raise ValueError("No valid polygons to export")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, out_dir / img_path.name)
    label_path = out_dir / f"{img_path.stem}.txt"
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return label_path


def pick_sam_checkpoint() -> Path | None:
    for path in SAM_CKPT_CANDIDATES:
        if path.is_file():
            return path
    return None


def pick_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def mask_to_polygons(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[tuple[int, int]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 1:
            continue
        epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        poly = [(int(x), int(y)) for x, y in approx]
        if len(poly) >= MIN_POLY_POINTS:
            polygons.append(poly)
    return polygons


def color_for_label(label: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if label == LABEL_BAD:
        return (255, 140, 60, 55), (255, 140, 60, 230)
    return (60, 210, 120, 55), (60, 210, 120, 230)


def point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    qx = ax + t * dx
    qy = ay + t * dy
    return float(np.hypot(px - qx, py - qy))


class OutlineTool(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("100x CNN Mask Labeler")
        self.geometry("1300x850")
        self.paths = image_paths()
        self.idx = 0
        self.img_path: Path | None = None
        self.img_bgr: np.ndarray | None = None
        self.polygons: list[LabeledContour] = []
        self.active: list[tuple[int, int]] = []
        self.selected_idx: int | None = None
        self.scale = 1.0
        self.offset = (0, 0)
        self._photo = None
        self.sam_mode = tk.BooleanVar(value=False)
        self.sam_predictor = None
        self.sam_lock = threading.Lock()
        self.sam_ready = False
        self.sam_busy = False
        self.sam_encoded_path: Path | None = None
        self.sam_pos_points: list[tuple[int, int]] = []
        self.sam_neg_points: list[tuple[int, int]] = []
        self.label_choice = tk.StringVar(value=LABEL_GOOD)
        self._build_ui()
        if self.paths:
            self.load_image(0)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self, padding=8)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(controls, text="Open Folder", command=self.open_folder).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Prev", command=lambda: self.load_image(self.idx - 1)).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Next", command=lambda: self.load_image(self.idx + 1)).pack(fill=tk.X, pady=2)
        ttk.Label(controls, text="Contour label:", font=("", 10, "bold")).pack(anchor=tk.W, pady=(12, 2))
        label_row = ttk.Frame(controls)
        label_row.pack(fill=tk.X)
        ttk.Radiobutton(label_row, text="Good", value=LABEL_GOOD, variable=self.label_choice).pack(side=tk.LEFT)
        ttk.Radiobutton(label_row, text="Bad", value=LABEL_BAD, variable=self.label_choice).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(controls, text="Apply Label To Selected", command=self.apply_label_to_selected).pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(
            controls,
            text="SAM click mode",
            variable=self.sam_mode,
            command=self.on_sam_toggle,
        ).pack(fill=tk.X, pady=(16, 2))
        ttk.Label(
            controls,
            text="With SAM checked: left-click adds a SAM mask, right-click adds an exclude point.",
            wraplength=220,
        ).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(controls, text="Save JSON", command=self.save_json).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Export Training Label", command=self.export_label).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Undo Point", command=self.undo_point).pack(fill=tk.X, pady=(16, 2))
        ttk.Button(controls, text="Finish Polygon", command=self.finish_polygon).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Delete Last Polygon", command=self.delete_last).pack(fill=tk.X, pady=2)
        self.status = tk.StringVar(value="Put 100x images in 100xCNN/images")
        ttk.Label(controls, textvariable=self.status, wraplength=220).pack(fill=tk.X, pady=(16, 0))
        ttk.Label(
            controls,
            text="Manual mode:\nLeft-click: add point\nDouble-click: finish\nRight-click: undo point",
            wraplength=220,
        ).pack(fill=tk.X, pady=(16, 0))

        self.canvas = tk.Canvas(self, bg="#111", cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Double-Button-1>", lambda _e: self.finish_polygon())
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())
        self.bind("g", lambda _e: self.label_choice.set(LABEL_GOOD))
        self.bind("b", lambda _e: self.label_choice.set(LABEL_BAD))

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=str(IMG_DIR))
        if not folder:
            return
        folder_path = Path(folder)
        self.paths = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if self.paths:
            self.load_image(0)

    def load_image(self, idx: int) -> None:
        if not self.paths:
            self.status.set("No images found")
            return
        if self.img_path is not None:
            self.save_json(silent=True)
        self.idx = max(0, min(idx, len(self.paths) - 1))
        self.img_path = self.paths[self.idx]
        self.img_bgr = cv2.imread(str(self.img_path), cv2.IMREAD_COLOR)
        if self.img_bgr is None:
            self.status.set(f"Could not load {self.img_path.name}")
            return
        self.polygons = load_annotations(self.img_path)
        self.active = []
        self.selected_idx = None
        self.sam_pos_points = []
        self.sam_neg_points = []
        self.sam_encoded_path = None
        self.status.set(f"[{self.idx + 1}/{len(self.paths)}] {self.img_path.name} - {len(self.polygons)} contour(s)")
        self.redraw()
        if self.sam_mode.get() and self.sam_ready:
            threading.Thread(target=self.encode_sam_image, daemon=True).start()

    def image_point(self, event) -> tuple[int, int]:
        ox, oy = self.offset
        x = int(round((event.x - ox) / max(self.scale, 1e-6)))
        y = int(round((event.y - oy) / max(self.scale, 1e-6)))
        if self.img_bgr is None:
            return x, y
        h, w = self.img_bgr.shape[:2]
        return max(0, min(x, w - 1)), max(0, min(y, h - 1))

    def canvas_point(self, x: int, y: int) -> tuple[float, float]:
        ox, oy = self.offset
        return x * self.scale + ox, y * self.scale + oy

    def hit_test_polygon(self, cx: float, cy: float) -> int | None:
        best_idx = None
        best_dist = 12.0
        for idx, contour in enumerate(self.polygons):
            pts_img = contour_points(contour)
            if len(pts_img) < 2:
                continue
            pts_canvas = [self.canvas_point(x, y) for x, y in pts_img]
            for j in range(len(pts_canvas)):
                ax, ay = pts_canvas[j]
                bx, by = pts_canvas[(j + 1) % len(pts_canvas)]
                dist = point_to_segment_dist(cx, cy, ax, ay, bx, by)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        return best_idx

    def apply_label_to_selected(self) -> None:
        if self.selected_idx is None:
            self.status.set("Select a contour first (click near its edge in manual mode).")
            return
        label = normalize_label(self.label_choice.get())
        contour = self.polygons[self.selected_idx]
        contour["label"] = label
        self.status.set(f"Contour #{self.selected_idx + 1} labeled as {label}")
        self.redraw()

    def add_point(self, event) -> None:
        if self.img_bgr is None:
            return
        if self.sam_mode.get():
            self.add_sam_point(event, positive=True)
            return
        if not self.active:
            hit = self.hit_test_polygon(float(event.x), float(event.y))
            if hit is not None:
                self.selected_idx = hit
                lbl = contour_label(self.polygons[hit])
                self.status.set(f"Selected contour #{hit + 1} ({lbl}). Press Apply Label To Selected to change.")
                self.redraw()
                return
        self.active.append(self.image_point(event))
        self.status.set(f"Drawing polygon: {len(self.active)} point(s)")
        self.redraw()

    def on_right_click(self, event) -> None:
        if self.sam_mode.get():
            self.add_sam_point(event, positive=False)
            return
        self.undo_point()

    def finish_polygon(self) -> None:
        if len(self.active) < MIN_POLY_POINTS:
            self.status.set(f"Need at least {MIN_POLY_POINTS} points")
            return
        label = normalize_label(self.label_choice.get())
        self.polygons.append({"points": list(self.active), "label": label})
        self.active = []
        self.selected_idx = len(self.polygons) - 1
        self.status.set(f"Added {label} polygon. Total: {len(self.polygons)}")
        self.redraw()

    def undo_point(self) -> None:
        if self.active:
            self.active.pop()
            self.redraw()

    def delete_last(self) -> None:
        if self.polygons:
            self.polygons.pop()
            self.selected_idx = None
            self.status.set(f"Deleted last polygon. Total: {len(self.polygons)}")
            self.redraw()

    def on_sam_toggle(self) -> None:
        if not self.sam_mode.get():
            self.status.set("SAM mode off. Manual polygon drawing enabled.")
            return
        self.active = []
        self.redraw()
        if self.sam_ready:
            if self.img_path != self.sam_encoded_path:
                threading.Thread(target=self.encode_sam_image, daemon=True).start()
            else:
                self.status.set("SAM mode on. Left-click a flake to add its mask.")
            return
        threading.Thread(target=self.load_sam, daemon=True).start()

    def load_sam(self) -> None:
        if self.sam_ready or self.sam_busy:
            return
        self.sam_busy = True
        self.after(0, lambda: self.status.set("Loading SAM..."))
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            checkpoint = pick_sam_checkpoint()
            if checkpoint is None:
                raise FileNotFoundError(
                    "Missing sam_vit_b_01ec64.pth. Put it in 100xCNN/ or Brody's Work/."
                )

            device = pick_device(torch)
            sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
            sam.to(device)
            sam.eval()
            predictor = SamPredictor(sam)
            with self.sam_lock:
                self.sam_predictor = predictor
                self.sam_ready = True
            self.after(0, lambda: self.status.set(f"SAM ready on {device}. Encoding image..."))
            self.encode_sam_image(torch_module=torch)
        except Exception as exc:
            self.after(0, lambda exc=exc: self.status.set(f"SAM load error: {exc}"))
        finally:
            self.sam_busy = False

    def encode_sam_image(self, torch_module=None) -> None:
        if self.sam_predictor is None or self.img_bgr is None or self.img_path is None:
            return
        self.sam_busy = True
        path = self.img_path
        self.after(0, lambda: self.status.set(f"Encoding {path.name} for SAM..."))
        try:
            if torch_module is None:
                import torch as torch_module
            rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            with self.sam_lock, torch_module.inference_mode():
                self.sam_predictor.set_image(rgb)
            self.sam_encoded_path = path
            self.after(0, lambda: self.status.set("SAM ready. Left-click a flake to add its mask."))
        except Exception as exc:
            self.after(0, lambda exc=exc: self.status.set(f"SAM encode error: {exc}"))
        finally:
            self.sam_busy = False

    def add_sam_point(self, event, *, positive: bool) -> None:
        if self.img_bgr is None:
            return
        point = self.image_point(event)
        if positive:
            self.sam_pos_points.append(point)
        else:
            self.sam_neg_points.append(point)

        if not self.sam_ready or self.sam_predictor is None:
            self.status.set("SAM is still loading. Try again in a moment.")
            return
        if self.sam_busy:
            self.status.set("SAM is busy. Try again in a moment.")
            return
        if self.sam_encoded_path != self.img_path:
            threading.Thread(target=self.encode_sam_image, daemon=True).start()
            return
        if not self.sam_pos_points:
            self.status.set("Add a positive left-click before exclude points.")
            return
        threading.Thread(target=self.run_sam_click, daemon=True).start()

    def run_sam_click(self) -> None:
        if self.sam_predictor is None or self.img_bgr is None:
            return
        self.sam_busy = True
        pos_points = list(self.sam_pos_points)
        neg_points = list(self.sam_neg_points)
        all_points = pos_points + neg_points
        all_labels = [1] * len(pos_points) + [0] * len(neg_points)
        self.after(
            0,
            lambda: self.status.set(
                f"Running SAM with {len(pos_points)} include / {len(neg_points)} exclude point(s)..."
            ),
        )
        try:
            import torch

            coords = np.array(all_points, dtype=np.float32)
            labels = np.array(all_labels, dtype=np.int32)
            with self.sam_lock, torch.inference_mode():
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=True,
                )
            best_idx = int(scores.argmax())
            polygons = mask_to_polygons(masks[best_idx])
            if polygons:
                label = normalize_label(self.label_choice.get())
                self.polygons.extend([{"points": poly, "label": label} for poly in polygons])
                self.sam_pos_points = []
                self.sam_neg_points = []
                self.after(0, self.redraw)
                self.after(
                    0,
                    lambda: self.status.set(
                        f"SAM added {len(polygons)} mask contour(s), score={float(scores[best_idx]):.3f}"
                    ),
                )
            else:
                self.after(0, lambda: self.status.set("SAM did not return a valid contour"))
        except Exception as exc:
            self.after(0, lambda exc=exc: self.status.set(f"SAM click error: {exc}"))
        finally:
            self.sam_busy = False

    def save_json(self, silent: bool = False) -> None:
        if self.img_path is None:
            return
        path = save_annotations(self.img_path, self.polygons)
        if not silent:
            self.status.set(f"Saved {len(self.polygons)} contour(s) -> {path.name}")

    def export_label(self) -> None:
        if self.img_path is None:
            return
        try:
            save_annotations(self.img_path, self.polygons)
            label = export_yolo(self.img_path, self.polygons)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.status.set(f"Exported -> {label.relative_to(BASE)}")

    def redraw(self) -> None:
        if self.img_bgr is None:
            return
        cw = max(self.canvas.winfo_width(), 900)
        ch = max(self.canvas.winfo_height(), 700)
        h, w = self.img_bgr.shape[:2]
        self.scale = min(cw / w, ch / h)
        dw, dh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.offset = (ox, oy)

        rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        canvas_img = Image.new("RGB", (cw, ch), (17, 17, 17))
        canvas_img.paste(pil, (ox, oy))
        draw = ImageDraw.Draw(canvas_img, "RGBA")

        for idx, contour in enumerate(self.polygons):
            pts = [self.canvas_point(x, y) for x, y in contour_points(contour)]
            label = contour_label(contour)
            fill_rgba, stroke_rgba = color_for_label(label)
            if len(pts) >= 3:
                flat = [v for pt in pts for v in pt]
                draw.polygon(flat, fill=fill_rgba)
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    draw.line([a, b], fill=stroke_rgba, width=3 if idx == self.selected_idx else 2)
                prefix = "G" if label == LABEL_GOOD else "B"
                draw.text(pts[0], f"{prefix}{idx + 1}", fill=(255, 255, 255, 240))

        if self.active:
            pts = [self.canvas_point(x, y) for x, y in self.active]
            for a, b in zip(pts, pts[1:]):
                draw.line([a, b], fill=(255, 80, 80, 255), width=2)
            for x, y in pts:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 80, 80, 255))

        self._photo = ImageTk.PhotoImage(canvas_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)


def main() -> int:
    OutlineTool().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

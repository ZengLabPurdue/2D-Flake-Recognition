"""
Core pipeline interfaces and ContourPipeline class.
Components are interchangeable via registry lookup by name.
"""
from abc import ABC, abstractmethod
from typing import Protocol, List, Optional, Callable, Any
import numpy as np


# --- Protocol definitions (interchangeable interfaces) ---

class Preprocessor(Protocol):
    """Preprocess image before edge detection. Callable: (img, params) -> img."""

    def __call__(self, img: np.ndarray, params: dict) -> np.ndarray:
        ...


class EdgeDetector(Protocol):
    """Compute edge map from work + preprocessed images. Callable: (work, preprocessed, params) -> edges uint8."""

    def __call__(self, work: np.ndarray, preprocessed: np.ndarray, params: dict) -> np.ndarray:
        ...


class GapCloser(Protocol):
    """Close gaps in binary edge map. Callable: (binary, params) -> binary."""

    def __call__(self, binary: np.ndarray, bin_h: int, bin_w: int, params: dict) -> np.ndarray:
        ...


class Binner(Protocol):
    """Downsample image. Callable: (img, factor) -> img."""

    def __call__(self, img: np.ndarray, factor: int) -> np.ndarray:
        ...


# --- Registry types ---
PREPROCESSOR_REGISTRY: dict[str, Preprocessor] = {}
EDGE_DETECTOR_REGISTRY: dict[str, EdgeDetector] = {}
GAP_CLOSER_REGISTRY: dict[str, GapCloser] = {}
BINNER_REGISTRY: dict[str, Binner] = {}


def register_preprocessor(name: str):
    """Decorator to register a preprocessor."""
    def decorator(fn: Callable):
        PREPROCESSOR_REGISTRY[name] = fn
        return fn
    return decorator


def register_edge_detector(name: str):
    """Decorator to register an edge detector."""
    def decorator(fn: Callable):
        EDGE_DETECTOR_REGISTRY[name] = fn
        return fn
    return decorator


def register_gap_closer(name: str):
    """Decorator to register a gap closer."""
    def decorator(fn: Callable):
        GAP_CLOSER_REGISTRY[name] = fn
        return fn
    return decorator


def register_binner(name: str):
    """Decorator to register a binner."""

    def decorator(fn: Callable):
        BINNER_REGISTRY[name] = fn
        return fn

    return decorator


# --- ContourPipeline: composes components ---

class ContourPipeline:
    """
    Runs the full contour detection pipeline with interchangeable components.
    Components are resolved from params by name (preprocessing, edge_method, edge_methods).
    """

    def __init__(
        self,
        preprocessors: Optional[dict] = None,
        edge_detectors: Optional[dict] = None,
        gap_closers: Optional[dict] = None,
        binners: Optional[dict] = None,
    ):
        self.preprocessors = preprocessors or PREPROCESSOR_REGISTRY
        self.edge_detectors = edge_detectors or EDGE_DETECTOR_REGISTRY
        self.gap_closers = gap_closers or GAP_CLOSER_REGISTRY
        self.binners = binners or BINNER_REGISTRY

    def _get_preprocessor(self, params: dict) -> Preprocessor:
        name = params.get("preprocessing", "full")
        if name not in self.preprocessors:
            name = "full"
        return self.preprocessors[name]

    def _get_edge_detector(self, params: dict) -> EdgeDetector:
        # Combined detector handles edge_methods list; single method uses "default" detector
        return self.edge_detectors.get("combined", self.edge_detectors.get("canny_h"))

    def _get_binner(self, params: dict) -> Binner:
        name = "2x2" if params.get("binning", True) else "none"
        return self.binners.get(name) or self.binners.get("2x2")

    def run(
        self,
        img: np.ndarray,
        params: dict,
        *,
        return_edges: bool = True,
        return_binary: bool = False,
        scale_contours_to_original: bool = True,
    ) -> dict:
        """
        Run pipeline on image. Returns dict with:
        - contours: list of contours (at original scale if scale_contours_to_original)
        - overlay: original with contours drawn
        - edges: edge map (if return_edges)
        - binary: gap-closed binary (if return_binary)
        - work, preprocessed: intermediate (for debugging)
        - n_contours, n_edges_px: stats
        """
        orig = _ensure_rgb_uint8(img)
        orig_h, orig_w = orig.shape[:2]

        # Binning
        use_binning = params.get("binning", True)
        binner = self._get_binner(params)
        if use_binning:
            work = binner(orig, 2)
            if work.dtype != np.uint8:
                work = np.clip(work, 0, 255).astype(np.uint8)
            bin_h, bin_w = work.shape[:2]
            scale_x = orig_w / bin_w
            scale_y = orig_h / bin_h
        else:
            work = np.clip(orig, 0, 255).astype(np.uint8)
            bin_h, bin_w = orig_h, orig_w
            scale_x = scale_y = 1.0

        # Preprocess
        preprocessor = self._get_preprocessor(params)
        preprocessed = preprocessor(work, params)

        # Edge detection
        edge_detector = self._get_edge_detector(params)
        edges = edge_detector(work, preprocessed, params)

        # Gap closing (uses built-in logic for now; could be pluggable)
        from edge_to_contour_methods import _ensure_binary
        binary = _ensure_binary(edges).copy()
        binary = self._gap_close(binary, bin_h, bin_w, params)

        # Find contours
        import cv2
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_c = []
        for c in contours:
            if len(c) > 0:
                first, last = c[0][0], c[-1][0]
                if not np.array_equal(first, last):
                    c = np.vstack([c, c[0:1]])
                all_c.append(c)

        from batch_robust_contours_and_masks import filter_nested_contours
        if params.get("filter_nested_contours", False):
            all_c = filter_nested_contours(all_c)

        # Scale to original and filter by area
        if scale_contours_to_original and use_binning:
            scaled = []
            for c in all_c:
                sc = c.copy().astype(np.float32)
                sc[:, 0, 0] *= scale_y
                sc[:, 0, 1] *= scale_x
                scaled.append(sc.astype(np.int32))
        else:
            scaled = all_c

        min_area_contour = params.get("min_area_contour", 0)
        if min_area_contour > 0:
            scaled = [c for c in scaled if cv2.contourArea(c) >= min_area_contour]

        # Color filter (yellow/blue/good flake)
        from .color_filters import filter_contours_by_color
        scaled = filter_contours_by_color(
            orig,
            scaled,
            filter_yellow=params.get("filter_yellow", False),
            filter_blue=params.get("filter_blue", False),
            filter_good_flake=params.get("filter_good_flake", False),
            yellow_v_min=params.get("yellow_v_min", 70),
            blue_h_min=params.get("blue_h_min", 100),
            good_flake_g_max=params.get("good_flake_g_max", 130),
        )

        # Build outputs
        overlay = orig.copy()
        lt = max(2, int(min(orig_w, orig_h) / 500))
        if scaled:
            cv2.drawContours(overlay, scaled, -1, (0, 255, 0), lt)

        shape_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if scaled:
            cv2.fillPoly(shape_mask, scaled, 255)

        masked_bg = orig.copy()
        for ch in range(3):
            masked_bg[:, :, ch] = np.where(shape_mask > 127, 0, orig[:, :, ch])

        out = {
            "original": orig,
            "overlay": overlay,
            "contours": scaled,
            "mask_shapes": shape_mask,
            "masked_background": masked_bg,
            "n_contours": len(scaled),
            "n_edges_px": int(np.sum(edges > 127)),
            "work": work,
            "preprocessed": preprocessed,
        }
        if return_edges:
            out["edges"] = np.stack([edges, edges, edges], axis=-1) if edges.ndim == 2 else edges
        if return_binary:
            out["binary"] = binary
        return out

    def _gap_close(self, binary: np.ndarray, bin_h: int, bin_w: int, params: dict) -> np.ndarray:
        """Apply gap closing (morph, bridge, force-close, extend to border)."""
        import cv2
        import numpy as np
        from edge_to_contour_methods import smart_connect_endpoints_bridged, force_close_open_chains, _find_endpoints
        from batch_robust_contours_and_masks import morph_extend_to_border

        diag = int(np.hypot(bin_h, bin_w))
        bridge_factor = params.get("bridge_gap_factor", 0.15)
        force_factor = params.get("force_close_factor", 0.3)
        line_thick = max(1, int(params.get("line_thickness", 4)))
        close_div = max(1, int(params.get("close_kernel_divisor", 100)))
        morph_margin = params.get("morph_margin", 15)
        corner_rad = params.get("corner_radius", 2)

        adaptive_bridge = max(50, int(diag * bridge_factor))
        adaptive_force = max(100, int(diag * force_factor))
        ksize = max(3, int(min(bin_h, bin_w) / close_div))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = smart_connect_endpoints_bridged(
            binary, max_gap=adaptive_bridge, line_thickness=line_thick,
            direction_weight=2.0, extend_pixels=2
        )
        binary = force_close_open_chains(
            binary, line_thickness=line_thick, max_iterations=200, max_gap=adaptive_force
        )
        remaining = _find_endpoints(binary)
        if remaining:
            h, w = binary.shape
            rad = min(adaptive_force * 2, max(h, w))
            for y1, x1 in remaining:
                best_d, best_p = float("inf"), None
                for py in range(max(0, y1 - rad), min(h, y1 + int(rad) + 1)):
                    for px in range(max(0, x1 - rad), min(w, x1 + int(rad) + 1)):
                        if binary[py, px] < 127 or (py == y1 and px == x1):
                            continue
                        d = np.hypot(py - y1, px - x1)
                        if 0.5 < d < best_d:
                            best_d, best_p = d, (py, px)
                if best_p:
                    cv2.line(binary, (x1, y1), (best_p[1], best_p[0]), 255, line_thick)
        binary = morph_extend_to_border(binary, bin_h, bin_w, margin=morph_margin, corner_radius=corner_rad)
        return binary


def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure RGB uint8 for pipeline."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

"""Small, failure-safe acceleration helpers for Tk image presentation."""

from __future__ import annotations

import os
from time import perf_counter

import cv2


CPU_BACKEND = "cpu"
OPENCL_BACKEND = "opencl"
AUTO_BACKEND = "auto"
VALID_BACKENDS = {AUTO_BACKEND, CPU_BACKEND, OPENCL_BACKEND}


def opencl_available():
    try:
        return bool(cv2.ocl.haveOpenCL())
    except (AttributeError, cv2.error):
        return False


def opencl_device_name():
    if not opencl_available():
        return None
    try:
        return cv2.ocl.Device_getDefault().name() or "OpenCL device"
    except (AttributeError, cv2.error):
        return "OpenCL device"


def requested_backend(environment_name, default=AUTO_BACKEND):
    value = os.environ.get(environment_name, default).strip().casefold()
    return value if value in VALID_BACKENDS else default


class DisplayResizer:
    """Resize with CPU or OpenCL while always returning a NumPy image.

    Tk ultimately needs host memory, so camera ``auto`` intentionally chooses
    the CPU. Set ``FLAKE_SEARCH_CAMERA_RENDERER=opencl`` to benchmark a specific
    system without changing code.
    """

    def __init__(self, environment_name="FLAKE_SEARCH_CAMERA_RENDERER"):
        self.requested = requested_backend(environment_name)
        self.backend = (
            OPENCL_BACKEND
            if self.requested == OPENCL_BACKEND and opencl_available()
            else CPU_BACKEND
        )
        self.device_name = (
            opencl_device_name()
            if self.backend == OPENCL_BACKEND
            else None
        )
        self.fallback_reason = (
            "OpenCL is unavailable"
            if self.requested == OPENCL_BACKEND and self.backend != OPENCL_BACKEND
            else None
        )

    @property
    def label(self):
        if self.backend == OPENCL_BACKEND:
            return f"OpenCL ({self.device_name})"
        return "CPU"

    def resize(self, image, size, interpolation):
        started_at = perf_counter()
        if self.backend == OPENCL_BACKEND:
            try:
                cv2.ocl.setUseOpenCL(True)
                resized = cv2.resize(
                    cv2.UMat(image),
                    size,
                    interpolation=interpolation,
                ).get()
                return resized, perf_counter() - started_at
            except (AttributeError, cv2.error) as exc:
                self.backend = CPU_BACKEND
                self.device_name = None
                self.fallback_reason = str(exc)

        resized = cv2.resize(image, size, interpolation=interpolation)
        return resized, perf_counter() - started_at

"""Standalone access to the region classifier used by the application.

This module intentionally contains no classifier fork. Importing it exposes the
objects from ``App/Scanning/region_classifier.py`` so benchmarks and experiments
in this folder always exercise the production implementation.
"""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_DIRECTORY = REPOSITORY_ROOT / "App"
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))

from Scanning import region_classifier as _app_region_classifier  # noqa: E402


APP_IMPLEMENTATION_PATH = Path(_app_region_classifier.__file__).resolve()

# Re-export the complete implementation, including diagnostic helpers. This is
# deliberately broader than ``from ... import *`` so private timing/inspection
# helpers remain available to standalone experiments when needed.
__all__ = [
    name
    for name in dir(_app_region_classifier)
    if not name.startswith("__")
]
globals().update({
    name: getattr(_app_region_classifier, name)
    for name in __all__
})

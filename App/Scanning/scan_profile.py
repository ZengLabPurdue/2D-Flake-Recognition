from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

import cv2
import numpy as np

from config import HOME_DIR
from Imaging import image_metadata
from .contour_finder import region_contrast_rgb
from .contour_extractor import get_region_from_point


PROFILE_SCHEMA = "flake-search.scan-search-profile"
PROFILE_VERSION = 4
SUPPORTED_PROFILE_VERSIONS = {1, 2, 3, PROFILE_VERSION}
PROFILE_FILENAME = "profile.json"
DEFAULT_PROFILES_DIR = HOME_DIR / "Profiles"
ScanProfileError = ValueError

FILTER_BAD_COLOR = "bad_color"
FILTER_INTENSITY_RANGE = "intensity_range"
FILTER_COLOR_DISTANCE = "color_distance"
FILTER_TYPES = {
    FILTER_BAD_COLOR,
    FILTER_INTENSITY_RANGE,
    FILTER_COLOR_DISTANCE,
}
FILTER_TYPE_LABELS = {
    FILTER_BAD_COLOR: "Matches a bad color",
    FILTER_INTENSITY_RANGE: "Bad intensity range",
    FILTER_COLOR_DISTANCE: "Too far from a color",
}


def _profile_slug(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "profile"
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    return f"profile_{slug}" if slug.upper() in reserved else slug


def _read_image(path):
    try:
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error):
        return None


def _write_image(path, image):
    image_metadata.save_png(
        path,
        image,
        metadata={
            "vignette_applied": True,
            "source": "scan_profile",
        },
    )


def build_region_overlay(image_bgr, region_mask, seed_point):
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ScanProfileError("The source image must be a three-channel BGR image.")
    if region_mask is None or region_mask.shape != image_bgr.shape[:2]:
        raise ScanProfileError("The region mask does not match the source image.")

    selected = region_mask > 0
    if not np.any(selected):
        raise ScanProfileError("The selected region is empty.")

    preview = image_bgr.copy()
    green = np.full_like(preview, (0, 255, 0))
    blended = cv2.addWeighted(preview, 0.68, green, 0.32, 0)
    preview[selected] = blended[selected]
    contours, _ = cv2.findContours(
        selected.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(preview, contours, -1, (255, 255, 255), 2)
    cv2.circle(preview, seed_point, 5, (0, 0, 255), -1)
    return preview


class ScanProfile:
    def __init__(self, profiles_dir=DEFAULT_PROFILES_DIR):
        self.profiles_dir = Path(profiles_dir)
        self.clear()

    def clear(self):
        self.name = ""
        self.path = None
        self.created_at = ""
        self.updated_at = ""
        self.minimum_size_um = None
        self.maximum_size_um = None
        self.classes = []
        self.filters = []

    def profile_directory(self, name):
        return self.profiles_dir / _profile_slug(name)

    def find_class(self, name):
        name = name.strip().casefold()
        for index, profile_class in enumerate(self.classes):
            if profile_class["name"].casefold() == name:
                return index
        return None

    def get_class(self, index):
        return self.classes[index]

    def set_class(
        self,
        name,
        source_path,
        image_bgr,
        region_mask,
        seed_point,
        threshold,
        minimum_size_um=None,
        maximum_size_um=None,
        group="",
        identify=True,
        connectivity=8,
        index=None,
        derived_from=None,
    ):
        class_id = None
        if index is not None:
            existing_class = self.classes[index]
            class_id = existing_class["id"]
            if derived_from is None:
                derived_from = existing_class.get("derived_from")
        profile_class = self._make_class(
            name,
            source_path,
            image_bgr,
            region_mask,
            seed_point,
            threshold,
            connectivity,
            minimum_size_um,
            maximum_size_um,
            class_id=class_id,
            group=group,
            identify=identify,
            derived_from=derived_from,
        )
        duplicate = self.find_class(profile_class["name"])
        if duplicate is not None and duplicate != index:
            raise ScanProfileError(f"A class named '{profile_class['name']}' already exists.")
        if index is None:
            self.classes.append(profile_class)
        else:
            self.classes[index] = profile_class
        return profile_class

    def set_size_requirement(self, minimum=None, maximum=None):
        minimum, maximum = self.validate_size_requirement(minimum, maximum)
        self.minimum_size_um = minimum
        self.maximum_size_um = maximum
        return minimum, maximum

    def copy_class(self, index, name=None, *, source_profile_name=None):
        """Copy a class into this profile with a new identity and unique label."""
        source = self.get_class(index)
        copied_name = self._unique_class_name(name or f"{source['name']} copy")
        origin = {
            "profile": source_profile_name or self.name or None,
            "class_id": source["id"],
        }
        copied = self._make_class(
            copied_name,
            source["source_path"],
            source["image_bgr"].copy(),
            source["region_mask"].copy(),
            source["seed_point"],
            source["threshold"],
            source["connectivity"],
            source["minimum_size_um"],
            source["maximum_size_um"],
            contrast=source["contrast_rgb"],
            group=source["group"],
            identify=source["identify"],
            derived_from=origin,
        )
        self.classes.append(copied)
        return copied

    def extend_from_profile(self, source_profile):
        """Append independent copies of every class and filter in another profile."""
        added = []
        for source in source_profile.classes:
            copied_name = self._unique_class_name(source["name"])
            origin = {
                "profile": source_profile.name or None,
                "class_id": source["id"],
            }
            copied = self._make_class(
                copied_name,
                source["source_path"],
                source["image_bgr"].copy(),
                source["region_mask"].copy(),
                source["seed_point"],
                source["threshold"],
                source["connectivity"],
                source["minimum_size_um"],
                source["maximum_size_um"],
                contrast=source["contrast_rgb"],
                group=source["group"],
                identify=source["identify"],
                derived_from=origin,
            )
            self.classes.append(copied)
            added.append(copied)
        added_filters = []
        for source_filter in source_profile.filters:
            copied = self._copy_filter_data(
                source_filter,
                self._next_filter_name(),
                source_profile.name,
            )
            self.filters.append(copied)
            added_filters.append(copied)
        return added, added_filters

    def find_filter(self, name):
        name = name.strip().casefold()
        for index, profile_filter in enumerate(self.filters):
            if profile_filter["name"].casefold() == name:
                return index
        return None

    def get_filter(self, index):
        return self.filters[index]

    def set_filter(
        self,
        filter_type,
        *,
        source_path=None,
        image_bgr=None,
        region_mask=None,
        seed_point=None,
        threshold=15,
        connectivity=8,
        minimum_intensity=None,
        maximum_intensity=None,
        distance_threshold=None,
        index=None,
        derived_from=None,
    ):
        name = self._next_filter_name() if index is None else self.filters[index]["name"]
        filter_id = None if index is None else self.filters[index]["id"]
        if index is not None and derived_from is None:
            derived_from = self.filters[index].get("derived_from")
        profile_filter = self._make_filter(
            name,
            filter_type,
            source_path=source_path,
            image=image_bgr,
            region_mask=region_mask,
            seed_point=seed_point,
            threshold=threshold,
            connectivity=connectivity,
            minimum_intensity=minimum_intensity,
            maximum_intensity=maximum_intensity,
            distance_threshold=distance_threshold,
            filter_id=filter_id,
            derived_from=derived_from,
        )
        if index is None:
            self.filters.append(profile_filter)
        else:
            self.filters[index] = profile_filter
        return profile_filter

    def copy_filter(self, index):
        source = self.get_filter(index)
        copied = self._copy_filter_data(source, self._next_filter_name(), self.name)
        self.filters.append(copied)
        return copied

    def _copy_filter_data(self, source, name, source_profile_name=None):
        origin = {
            "profile": source_profile_name or None,
            "filter_id": source["id"],
        }
        return self._make_filter(
            name,
            source["type"],
            source_path=source.get("source_path"),
            image=(
                source["image_bgr"].copy()
                if source.get("image_bgr") is not None
                else None
            ),
            region_mask=(
                source["region_mask"].copy()
                if source.get("region_mask") is not None
                else None
            ),
            seed_point=source.get("seed_point"),
            threshold=source.get("threshold", 15),
            connectivity=source.get("connectivity", 8),
            minimum_intensity=source.get("minimum_intensity"),
            maximum_intensity=source.get("maximum_intensity"),
            distance_threshold=source.get("distance_threshold"),
            derived_from=origin,
        )

    def remove_filter(self, index):
        return self.filters.pop(index)

    def matching_configuration(self, contrast_threshold=None):
        """Build classifier input directly from the editable in-memory profile."""
        classes = []
        for class_index, profile_class in enumerate(self.classes):
            tolerance = (
                profile_class["threshold"]
                if contrast_threshold is None
                else contrast_threshold
            )
            classes.append({
                "name": profile_class["name"],
                "contrast_rgb": np.asarray(profile_class["contrast_rgb"], dtype=np.float64),
                "tolerance": float(tolerance),
                "class_index": class_index,
                "group": profile_class["group"],
                "identify": profile_class["identify"],
                "minimum_size_um": profile_class["minimum_size_um"],
                "maximum_size_um": profile_class["maximum_size_um"],
            })
        filters = []
        for profile_filter in self.filters:
            item = {
                "name": profile_filter["name"],
                "type": profile_filter["type"],
            }
            if profile_filter["contrast_rgb"] is not None:
                item["contrast_rgb"] = np.asarray(
                    profile_filter["contrast_rgb"],
                    dtype=np.float64,
                )
            if profile_filter["type"] == FILTER_BAD_COLOR:
                item["tolerance"] = float(profile_filter["threshold"])
            elif profile_filter["type"] == FILTER_INTENSITY_RANGE:
                item["minimum_intensity"] = profile_filter["minimum_intensity"]
                item["maximum_intensity"] = profile_filter["maximum_intensity"]
            else:
                item["distance_threshold"] = profile_filter["distance_threshold"]
            filters.append(item)
        return {
            "classes": classes,
            "filters": filters,
            "minimum_size_um": self.minimum_size_um,
            "maximum_size_um": self.maximum_size_um,
        }

    @property
    def groups(self):
        return sorted(
            {item["group"] for item in self.classes if item["group"]},
            key=str.casefold,
        )

    def remove_class(self, index):
        return self.classes.pop(index)

    def move_class(self, old_index, new_index):
        profile_class = self.classes.pop(old_index)
        self.classes.insert(new_index, profile_class)
        return profile_class

    def save_profile(self, name=None, overwrite=False):
        name = (self.name if name is None else name).strip()
        if not name:
            raise ScanProfileError("Enter a profile name before saving.")
        if not self.classes and not self.filters:
            raise ScanProfileError("Add at least one confirmed class or filter before saving.")

        names = [profile_class["name"].casefold() for profile_class in self.classes]
        if len(names) != len(set(names)):
            raise ScanProfileError("Class names must be unique.")
        for profile_class in self.classes:
            self._validate_class(profile_class)
        for profile_filter in self.filters:
            self._validate_filter(profile_filter)

        profile_minimum, profile_maximum = self.validate_size_requirement(
            self.minimum_size_um,
            self.maximum_size_um,
        )

        destination = self.profile_directory(name)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)

        images_dir = destination / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        saved_classes = []

        for profile_class in self.classes:
            class_id = profile_class["id"]
            filename = f"{_profile_slug(profile_class['name'])}_{class_id[:8]}_source.png"
            source_path = images_dir / filename
            _write_image(source_path, profile_class["image_bgr"])
            minimum, maximum = self.validate_size_requirement(
                profile_class["minimum_size_um"],
                profile_class["maximum_size_um"],
            )
            red, green, blue = profile_class["contrast_rgb"]
            saved_classes.append({
                "id": class_id,
                "name": profile_class["name"],
                "source_image": f"images/{filename}",
                "seed_point": {
                    "x": profile_class["seed_point"][0],
                    "y": profile_class["seed_point"][1],
                },
                "flood_fill": {
                    "threshold": profile_class["threshold"],
                    "connectivity": profile_class["connectivity"],
                },
                "size_requirement": (
                    {"minimum_size_um": minimum, "maximum_size_um": maximum}
                    if minimum is not None or maximum is not None
                    else None
                ),
                "contrast_rgb": {"red": red, "green": green, "blue": blue},
                "group": profile_class["group"] or None,
                "identify": profile_class["identify"],
            })
            if profile_class["derived_from"] is not None:
                saved_classes[-1]["derived_from"] = profile_class["derived_from"]

        saved_filters = []
        for profile_filter in self.filters:
            saved_filter = {
                "id": profile_filter["id"],
                "name": profile_filter["name"],
                "type": profile_filter["type"],
            }
            if profile_filter["type"] in (FILTER_BAD_COLOR, FILTER_COLOR_DISTANCE):
                filter_id = profile_filter["id"]
                filename = f"{_profile_slug(profile_filter['name'])}_{filter_id[:8]}_source.png"
                source_path = images_dir / filename
                _write_image(source_path, profile_filter["image_bgr"])
                red, green, blue = profile_filter["contrast_rgb"]
                saved_filter.update({
                    "source_image": f"images/{filename}",
                    "seed_point": {
                        "x": profile_filter["seed_point"][0],
                        "y": profile_filter["seed_point"][1],
                    },
                    "flood_fill": {
                        "threshold": profile_filter["threshold"],
                        "connectivity": profile_filter["connectivity"],
                    },
                    "contrast_rgb": {"red": red, "green": green, "blue": blue},
                })
                if profile_filter["type"] == FILTER_COLOR_DISTANCE:
                    saved_filter["distance_threshold"] = profile_filter[
                        "distance_threshold"
                    ]
            else:
                saved_filter["intensity_range"] = {
                    "minimum": profile_filter["minimum_intensity"],
                    "maximum": profile_filter["maximum_intensity"],
                }
            if profile_filter["derived_from"] is not None:
                saved_filter["derived_from"] = profile_filter["derived_from"]
            saved_filters.append(saved_filter)

        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema": PROFILE_SCHEMA,
            "version": PROFILE_VERSION,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "size_requirement": (
                {
                    "minimum_size_um": profile_minimum,
                    "maximum_size_um": profile_maximum,
                }
                if profile_minimum is not None or profile_maximum is not None
                else None
            ),
            "classes": saved_classes,
            "filters": saved_filters,
        }
        (destination / PROFILE_FILENAME).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.load_profile(destination)

    def load_profile(self, path):
        path = Path(path)
        profile_path = path / PROFILE_FILENAME if path.is_dir() else path
        if not profile_path.is_file():
            raise ScanProfileError(f"Profile JSON was not found: {profile_path}")

        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScanProfileError(f"Could not read profile JSON: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("schema") != PROFILE_SCHEMA:
            raise ScanProfileError("This file is not a scan search profile.")
        profile_version = payload.get("version")
        if profile_version not in SUPPORTED_PROFILE_VERSIONS:
            raise ScanProfileError(f"Unsupported profile version: {payload.get('version')!r}.")

        name = payload.get("name")
        saved_classes = payload.get("classes")
        saved_filters = payload.get("filters", [])
        if not isinstance(name, str) or not name.strip():
            raise ScanProfileError("The profile name is missing.")
        if not isinstance(saved_classes, list) or not isinstance(saved_filters, list):
            raise ScanProfileError("The profile class and filter lists must be arrays.")
        if not saved_classes and not saved_filters:
            raise ScanProfileError("The profile must contain at least one class or filter.")

        profile_dir = profile_path.parent.resolve()
        profile_size = self._read_size(payload.get("size_requirement"))
        loaded_classes = []
        loaded_filters = []

        for index, saved_class in enumerate(saved_classes, start=1):
            if not isinstance(saved_class, dict):
                raise ScanProfileError(f"Class {index} must be a JSON object.")

            class_name = saved_class.get("name")
            seed = saved_class.get("seed_point")
            flood_fill = saved_class.get("flood_fill")
            if not isinstance(class_name, str) or not class_name.strip():
                raise ScanProfileError(f"Class {index} has no name.")
            if not isinstance(seed, dict) or not isinstance(flood_fill, dict):
                raise ScanProfileError(f"Class {index} is missing region metadata.")

            try:
                seed_point = (int(seed["x"]), int(seed["y"]))
                threshold = int(flood_fill["threshold"])
                connectivity = int(flood_fill["connectivity"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ScanProfileError(f"Class {index} has invalid region metadata.") from exc

            source_path = self._asset_path(
                profile_dir,
                saved_class.get("source_image"),
                index,
            )
            if not image_metadata.is_vignette_corrected(source_path):
                raise ScanProfileError(
                    f"Class {class_name} does not use a vignette-corrected PNG image."
                )
            image = _read_image(source_path)
            if image is None:
                raise ScanProfileError(f"Could not read source image for class {class_name}.")
            region_mask = self._region_mask(
                image,
                seed_point,
                threshold,
                connectivity,
                class_name,
            )

            size = (
                self._read_size(saved_class.get("size_requirement"))
                if "size_requirement" in saved_class
                else (profile_size if profile_version < 3 else (None, None))
            )
            contrast = saved_class.get("contrast_rgb")
            if isinstance(contrast, dict):
                contrast = (
                    contrast.get("red"),
                    contrast.get("green"),
                    contrast.get("blue"),
                )
            elif profile_version >= 2:
                raise ScanProfileError(f"Class {class_name} has invalid RGB contrast.")
            else:
                # Version 1 stored the absolute region color. Recalculate its
                # contrast from the source image and the current flake mask.
                contrast = None

            identify = saved_class.get("identify", True)
            legacy_reject = profile_version == 3 and saved_class.get("reject", False)
            if not legacy_reject or identify:
                loaded_classes.append(self._make_class(
                    class_name,
                    source_path,
                    image,
                    region_mask,
                    seed_point,
                    threshold,
                    connectivity,
                    size[0],
                    size[1],
                    saved_class.get("id"),
                    contrast,
                    group=saved_class.get("group", ""),
                    identify=identify,
                    derived_from=saved_class.get("derived_from"),
                ))
            if legacy_reject:
                loaded_filters.append(self._make_filter(
                    f"Filter {len(loaded_filters) + 1}",
                    FILTER_BAD_COLOR,
                    source_path=source_path,
                    image=image,
                    region_mask=region_mask,
                    seed_point=seed_point,
                    threshold=threshold,
                    connectivity=connectivity,
                    contrast=contrast,
                    derived_from={
                        "profile": name.strip(),
                        "class_id": saved_class.get("id"),
                    },
                ))

        for index, saved_filter in enumerate(saved_filters, start=1):
            if not isinstance(saved_filter, dict):
                raise ScanProfileError(f"Filter {index} must be a JSON object.")
            filter_name = saved_filter.get("name")
            filter_type = saved_filter.get("type")
            if not isinstance(filter_name, str) or not filter_name.strip():
                raise ScanProfileError(f"Filter {index} has no name.")

            common = {
                "filter_id": saved_filter.get("id"),
                "derived_from": saved_filter.get("derived_from"),
            }
            if filter_type == FILTER_INTENSITY_RANGE:
                intensity_range = saved_filter.get("intensity_range")
                if not isinstance(intensity_range, dict):
                    raise ScanProfileError(
                        f"Filter {filter_name} has an invalid intensity range."
                    )
                loaded_filters.append(self._make_filter(
                    filter_name,
                    filter_type,
                    minimum_intensity=intensity_range.get("minimum"),
                    maximum_intensity=intensity_range.get("maximum"),
                    **common,
                ))
                continue

            seed = saved_filter.get("seed_point")
            flood_fill = saved_filter.get("flood_fill")
            contrast = saved_filter.get("contrast_rgb")
            if not isinstance(seed, dict) or not isinstance(flood_fill, dict):
                raise ScanProfileError(
                    f"Filter {filter_name} is missing region metadata."
                )
            if isinstance(contrast, dict):
                contrast = (
                    contrast.get("red"),
                    contrast.get("green"),
                    contrast.get("blue"),
                )
            try:
                seed_point = (int(seed["x"]), int(seed["y"]))
                threshold = int(flood_fill["threshold"])
                connectivity = int(flood_fill["connectivity"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ScanProfileError(
                    f"Filter {filter_name} has invalid region metadata."
                ) from exc
            source_path = self._asset_path(
                profile_dir,
                saved_filter.get("source_image"),
                f"filter {index}",
            )
            if not image_metadata.is_vignette_corrected(source_path):
                raise ScanProfileError(
                    f"Filter {filter_name} does not use a vignette-corrected PNG image."
                )
            image = _read_image(source_path)
            if image is None:
                raise ScanProfileError(
                    f"Could not read source image for filter {filter_name}."
                )
            region_mask = self._region_mask(
                image,
                seed_point,
                threshold,
                connectivity,
                filter_name,
            )
            loaded_filters.append(self._make_filter(
                filter_name,
                filter_type,
                source_path=source_path,
                image=image,
                region_mask=region_mask,
                seed_point=seed_point,
                threshold=threshold,
                connectivity=connectivity,
                distance_threshold=saved_filter.get("distance_threshold"),
                contrast=contrast,
                **common,
            ))

        class_names = [profile_class["name"].casefold() for profile_class in loaded_classes]
        if len(class_names) != len(set(class_names)):
            raise ScanProfileError("Class names must be unique.")

        self.name = name.strip()
        self.path = profile_dir
        self.created_at = str(payload.get("created_at", ""))
        self.updated_at = str(payload.get("updated_at", ""))
        self.minimum_size_um, self.maximum_size_um = profile_size
        self.classes = loaded_classes
        self.filters = loaded_filters
        return self

    def validate_size_requirement(self, minimum, maximum):
        for label, value in (("Minimum size", minimum), ("Maximum size", maximum)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ScanProfileError(f"{label} must be a number of micrometers.")
            if value is not None and value <= 0:
                raise ScanProfileError(f"{label} must be greater than zero.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ScanProfileError("Minimum size cannot be greater than maximum size.")
        return (
            None if minimum is None else float(minimum),
            None if maximum is None else float(maximum),
        )

    def _make_class(
        self,
        name,
        source_path,
        image,
        region_mask,
        seed_point,
        threshold,
        connectivity,
        minimum,
        maximum,
        class_id=None,
        contrast=None,
        *,
        group="",
        identify=True,
        derived_from=None,
    ):
        minimum, maximum = self.validate_size_requirement(minimum, maximum)
        if class_id is not None and (
            not isinstance(class_id, str) or not class_id.strip()
        ):
            raise ScanProfileError("Class ID must be non-empty text.")
        group = self._validate_group(group)
        identify = self._validate_flag(identify, "Identify")
        derived_from = self._validate_derived_from(derived_from)
        profile_class = {
            "id": class_id or uuid4().hex,
            "name": name.strip(),
            "source_path": Path(source_path),
            "image_bgr": np.ascontiguousarray(image),
            "region_mask": np.ascontiguousarray(region_mask),
            "seed_point": (int(seed_point[0]), int(seed_point[1])),
            "threshold": int(threshold),
            "connectivity": int(connectivity),
            "minimum_size_um": minimum,
            "maximum_size_um": maximum,
            "contrast_rgb": contrast,
            "group": group,
            "identify": identify,
            "derived_from": derived_from,
        }
        self._validate_class(profile_class)
        if contrast is None:
            try:
                profile_class["contrast_rgb"] = region_contrast_rgb(image, region_mask)
            except ValueError as exc:
                raise ScanProfileError(
                    f"Could not determine the background for class {name}."
                ) from exc
        elif (
            not isinstance(contrast, (list, tuple))
            or len(contrast) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not -255 <= value <= 255
                for value in contrast
            )
        ):
            raise ScanProfileError(f"Class {name} has invalid RGB contrast.")
        else:
            profile_class["contrast_rgb"] = tuple(
                int(round(value)) for value in contrast
            )
        return profile_class

    def _make_filter(
        self,
        name,
        filter_type,
        *,
        source_path=None,
        image=None,
        region_mask=None,
        seed_point=None,
        threshold=15,
        connectivity=8,
        minimum_intensity=None,
        maximum_intensity=None,
        distance_threshold=None,
        filter_id=None,
        contrast=None,
        derived_from=None,
    ):
        if filter_type not in FILTER_TYPES:
            raise ScanProfileError(f"Unknown filter type: {filter_type!r}.")
        if filter_id is not None and (
            not isinstance(filter_id, str) or not filter_id.strip()
        ):
            raise ScanProfileError("Filter ID must be non-empty text.")
        profile_filter = {
            "id": filter_id or uuid4().hex,
            "name": str(name).strip(),
            "type": filter_type,
            "source_path": Path(source_path) if source_path is not None else None,
            "image_bgr": (
                np.ascontiguousarray(image) if image is not None else None
            ),
            "region_mask": (
                np.ascontiguousarray(region_mask)
                if region_mask is not None
                else None
            ),
            "seed_point": (
                (int(seed_point[0]), int(seed_point[1]))
                if seed_point is not None
                else None
            ),
            "threshold": int(threshold),
            "connectivity": int(connectivity),
            "contrast_rgb": contrast,
            "minimum_intensity": minimum_intensity,
            "maximum_intensity": maximum_intensity,
            "distance_threshold": distance_threshold,
            "derived_from": self._validate_derived_from(derived_from),
        }
        if filter_type in (FILTER_BAD_COLOR, FILTER_COLOR_DISTANCE):
            if contrast is None:
                try:
                    contrast = region_contrast_rgb(image, region_mask)
                except (TypeError, ValueError) as exc:
                    raise ScanProfileError(
                        f"Could not determine the reference color for {name}."
                    ) from exc
            profile_filter["contrast_rgb"] = self._validate_contrast(
                contrast,
                f"Filter {name}",
            )
        if filter_type == FILTER_INTENSITY_RANGE:
            (
                profile_filter["minimum_intensity"],
                profile_filter["maximum_intensity"],
            ) = self.validate_intensity_range(
                minimum_intensity,
                maximum_intensity,
            )
        if filter_type == FILTER_COLOR_DISTANCE:
            profile_filter["distance_threshold"] = self.validate_color_distance(
                distance_threshold
            )
        self._validate_filter(profile_filter)
        return profile_filter

    def _validate_class(self, profile_class):
        name = profile_class["name"]
        image = profile_class["image_bgr"]
        mask = profile_class["region_mask"]
        if not name:
            raise ScanProfileError("Every class must have a name.")
        self._validate_group(profile_class.get("group", ""))
        self._validate_flag(profile_class.get("identify"), "Identify")
        self._validate_derived_from(profile_class.get("derived_from"))
        if image.ndim != 3 or image.shape[2] != 3:
            raise ScanProfileError(f"Class {name!r} has an invalid source image.")
        if mask.shape != image.shape[:2] or not np.any(mask > 0):
            raise ScanProfileError(f"Class {name!r} has an invalid region mask.")
        if profile_class["connectivity"] not in (4, 8):
            raise ScanProfileError("Flood-fill connectivity must be 4 or 8.")
        if not 0 <= profile_class["threshold"] <= 255:
            raise ScanProfileError("Flood-fill threshold must be between 0 and 255.")

        seed_x, seed_y = profile_class["seed_point"]
        height, width = image.shape[:2]
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ScanProfileError(f"Class {name!r} has an invalid seed point.")
        if mask[seed_y, seed_x] == 0:
            raise ScanProfileError(f"Class {name!r} does not contain its seed point.")

    def _validate_filter(self, profile_filter):
        name = profile_filter.get("name")
        filter_type = profile_filter.get("type")
        if not isinstance(name, str) or not name.strip():
            raise ScanProfileError("Every filter must have a name.")
        if filter_type not in FILTER_TYPES:
            raise ScanProfileError(f"Filter {name!r} has an invalid type.")
        self._validate_derived_from(profile_filter.get("derived_from"))
        if filter_type == FILTER_INTENSITY_RANGE:
            self.validate_intensity_range(
                profile_filter.get("minimum_intensity"),
                profile_filter.get("maximum_intensity"),
            )
            return

        image = profile_filter.get("image_bgr")
        mask = profile_filter.get("region_mask")
        seed_point = profile_filter.get("seed_point")
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ScanProfileError(f"Filter {name!r} has an invalid source image.")
        if mask is None or mask.shape != image.shape[:2] or not np.any(mask > 0):
            raise ScanProfileError(f"Filter {name!r} has an invalid region mask.")
        if profile_filter.get("connectivity") not in (4, 8):
            raise ScanProfileError("Flood-fill connectivity must be 4 or 8.")
        threshold = profile_filter.get("threshold")
        if not isinstance(threshold, int) or not 0 <= threshold <= 255:
            raise ScanProfileError("Filter tolerance must be between 0 and 255.")
        if seed_point is None:
            raise ScanProfileError(f"Filter {name!r} has no seed point.")
        seed_x, seed_y = seed_point
        height, width = image.shape[:2]
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ScanProfileError(f"Filter {name!r} has an invalid seed point.")
        if mask[seed_y, seed_x] == 0:
            raise ScanProfileError(f"Filter {name!r} does not contain its seed point.")
        self._validate_contrast(profile_filter.get("contrast_rgb"), f"Filter {name}")
        if filter_type == FILTER_COLOR_DISTANCE:
            self.validate_color_distance(profile_filter.get("distance_threshold"))

    def validate_intensity_range(self, minimum, maximum):
        values = []
        for label, value in (("Minimum intensity", minimum), ("Maximum intensity", maximum)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 <= value <= 255
            ):
                raise ScanProfileError(f"{label} must be between 0 and 255.")
            values.append(float(value))
        minimum, maximum = values
        if minimum > maximum:
            raise ScanProfileError("Minimum intensity cannot exceed maximum intensity.")
        return minimum, maximum

    @staticmethod
    def validate_color_distance(value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ScanProfileError("Color distance must be zero or greater.")
        return float(value)

    @staticmethod
    def _validate_contrast(contrast, label):
        if (
            not isinstance(contrast, (list, tuple, np.ndarray))
            or len(contrast) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or not -255 <= value <= 255
                for value in contrast
            )
        ):
            raise ScanProfileError(f"{label} has invalid RGB contrast.")
        return tuple(int(round(float(value))) for value in contrast)

    def _region_mask(self, image, seed_point, threshold, connectivity, item_name):
        try:
            _, _, region_mask, contour = get_region_from_point(
                image,
                seed_point,
                threshold,
                connectivity,
            )
        except (cv2.error, ValueError) as exc:
            raise ScanProfileError(
                f"Could not reconstruct the reference region for {item_name}."
            ) from exc
        if contour is None or not np.any(region_mask > 0):
            raise ScanProfileError(
                f"Could not reconstruct the reference region for {item_name}."
            )
        return region_mask

    def _read_size(self, size):
        if size is None:
            return None, None
        if not isinstance(size, dict):
            raise ScanProfileError("The profile size requirement must be an object or null.")
        return self.validate_size_requirement(
            size.get("minimum_size_um"),
            size.get("maximum_size_um"),
        )

    def _unique_class_name(self, preferred_name):
        preferred_name = str(preferred_name).strip() or "Class"
        if self.find_class(preferred_name) is None:
            return preferred_name
        suffix = 2
        while self.find_class(f"{preferred_name} {suffix}") is not None:
            suffix += 1
        return f"{preferred_name} {suffix}"

    def _next_filter_name(self):
        number = 1
        while self.find_filter(f"Filter {number}") is not None:
            number += 1
        return f"Filter {number}"

    @staticmethod
    def _validate_group(group):
        if group is None:
            return ""
        if not isinstance(group, str):
            raise ScanProfileError("Class group must be text.")
        return group.strip()

    @staticmethod
    def _validate_flag(value, label):
        if not isinstance(value, bool):
            raise ScanProfileError(f"{label} must be selected or cleared.")
        return value

    @staticmethod
    def _validate_derived_from(value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ScanProfileError("Class copy metadata must be an object or null.")
        profile = value.get("profile")
        class_id = value.get("class_id")
        filter_id = value.get("filter_id")
        if profile is not None and not isinstance(profile, str):
            raise ScanProfileError("Copied class profile must be text or null.")
        if class_id is not None and not isinstance(class_id, str):
            raise ScanProfileError("Copied class ID must be text or null.")
        if filter_id is not None and not isinstance(filter_id, str):
            raise ScanProfileError("Copied filter ID must be text or null.")
        result = {"profile": profile}
        if class_id is not None:
            result["class_id"] = class_id
        if filter_id is not None:
            result["filter_id"] = filter_id
        return result

    def _asset_path(self, profile_dir, relative_path, item_label):
        label = (
            f"Class {item_label}"
            if isinstance(item_label, int)
            else str(item_label).capitalize()
        )
        if not isinstance(relative_path, str) or not relative_path:
            raise ScanProfileError(f"{label} has a missing asset path.")
        relative_path = Path(relative_path)
        if relative_path.is_absolute():
            raise ScanProfileError(f"{label} asset paths must be relative.")
        path = (profile_dir / relative_path).resolve()
        if profile_dir not in path.parents:
            raise ScanProfileError(f"{label} asset path leaves the profile folder.")
        if not path.is_file():
            raise ScanProfileError(f"{label} asset was not found: {relative_path}")
        return path

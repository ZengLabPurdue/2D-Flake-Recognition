"""Persistence and image helpers for user-created scan search profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import unicodedata
from uuid import uuid4

import cv2
import numpy as np

from config import HOME_DIR
from .contour_extractor import get_region_from_point


PROFILE_SCHEMA = "flake-search.scan-search-profile"
PROFILE_VERSION = 1
PROFILE_FILENAME = "profile.json"
DEFAULT_PROFILES_DIR = HOME_DIR / "Profiles"


class ScanProfileError(ValueError):
    """Raised when a scan profile or one of its assets is invalid."""


@dataclass
class ProfileClassDraft:
    """An in-memory class sample that has not been saved yet."""

    name: str
    source_path: Path
    image_bgr: np.ndarray
    region_mask: np.ndarray
    seed_point: tuple[int, int]
    threshold: int
    connectivity: int = 8
    minimum_size_um: float | None = None
    maximum_size_um: float | None = None


@dataclass(frozen=True)
class ScanProfileClass:
    id: str
    name: str
    source_image: Path
    seed_point: tuple[int, int]
    threshold: int
    connectivity: int
    minimum_size_um: float | None
    maximum_size_um: float | None
    average_color_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class ScanSearchProfile:
    name: str
    path: Path
    created_at: str
    updated_at: str
    classes: tuple[ScanProfileClass, ...]
    version: int = PROFILE_VERSION


def _profile_slug(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
    slug = slug or "profile"

    # These names cannot be used as folders on Windows.
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if slug.upper() in reserved:
        slug = f"profile_{slug}"
    return slug


def _read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Read an image through imdecode so non-ASCII Windows paths work reliably."""

    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    try:
        return cv2.imdecode(encoded, flags)
    except cv2.error:
        return None


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ScanProfileError(f"Could not encode profile image: {path.name}")
    path.write_bytes(encoded.tobytes())


def build_region_overlay(
    image_bgr: np.ndarray,
    region_mask: np.ndarray,
    seed_point: tuple[int, int],
) -> np.ndarray:
    """Return a preview with the flood-filled region highlighted in green."""

    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ScanProfileError("The source image must be a three-channel BGR image.")
    if region_mask.shape != image_bgr.shape[:2]:
        raise ScanProfileError("The region mask does not match the source image.")

    selected = region_mask > 0
    if not np.any(selected):
        raise ScanProfileError("The selected region is empty.")

    preview = image_bgr.copy()
    green = np.zeros_like(preview)
    green[:, :] = (0, 255, 0)
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


class ScanProfileStore:
    """Save, validate, load, and preview scan search profiles."""

    def __init__(self, profiles_dir: Path | str = DEFAULT_PROFILES_DIR):
        self.profiles_dir = Path(profiles_dir)
        self.active_profile: ScanSearchProfile | None = None

    def profile_directory(self, name: str) -> Path:
        return self.profiles_dir / _profile_slug(name)

    def save_profile(
        self,
        name: str,
        classes: list[ProfileClassDraft] | tuple[ProfileClassDraft, ...],
        *,
        overwrite: bool = False,
    ) -> ScanSearchProfile:
        name = name.strip()
        if not name:
            raise ScanProfileError("Enter a profile name before saving.")

        class_drafts = list(classes)
        if not class_drafts:
            raise ScanProfileError("Add at least one confirmed class before saving.")

        seen_names: set[str] = set()
        for draft in class_drafts:
            class_name = draft.name.strip()
            if not class_name:
                raise ScanProfileError("Every class must have a name.")
            normalized_name = class_name.casefold()
            if normalized_name in seen_names:
                raise ScanProfileError(f"Class names must be unique: {class_name}")
            seen_names.add(normalized_name)
            self._validate_draft(draft)
            self.validate_size_requirement(
                draft.minimum_size_um,
                draft.maximum_size_um,
            )

        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        destination = self.profile_directory(name)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)

        staging = self.profiles_dir / f".{destination.name}-{uuid4().hex}.tmp"
        backup: Path | None = None

        try:
            (staging / "images").mkdir(parents=True)

            now = datetime.now(timezone.utc).isoformat()
            class_entries = []

            for draft in class_drafts:
                class_id = uuid4().hex
                class_slug = _profile_slug(draft.name)
                filename_base = f"{class_slug}_{class_id[:8]}"
                source_relative = Path("images") / f"{filename_base}_source.png"

                source_image = np.ascontiguousarray(draft.image_bgr)
                _write_png(staging / source_relative, source_image)

                minimum_size_um, maximum_size_um = self.validate_size_requirement(
                    draft.minimum_size_um,
                    draft.maximum_size_um,
                )
                red, green, blue = self._average_region_color(draft)

                class_entries.append({
                    "id": class_id,
                    "name": draft.name.strip(),
                    "source_image": source_relative.as_posix(),
                    "seed_point": {
                        "x": int(draft.seed_point[0]),
                        "y": int(draft.seed_point[1]),
                    },
                    "flood_fill": {
                        "threshold": int(draft.threshold),
                        "connectivity": int(draft.connectivity),
                    },
                    "size_requirement": (
                        {
                            "minimum_size_um": minimum_size_um,
                            "maximum_size_um": maximum_size_um,
                        }
                        if minimum_size_um is not None or maximum_size_um is not None
                        else None
                    ),
                    "average_color_rgb": {
                        "red": red,
                        "green": green,
                        "blue": blue,
                    },
                })

            payload = {
                "schema": PROFILE_SCHEMA,
                "version": PROFILE_VERSION,
                "name": name,
                "created_at": now,
                "updated_at": now,
                "classes": class_entries,
            }
            (staging / PROFILE_FILENAME).write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )

            if destination.exists():
                backup = self.profiles_dir / f".{destination.name}-{uuid4().hex}.bak"
                destination.replace(backup)

            try:
                staging.replace(destination)
            except Exception:
                if backup is not None and backup.exists() and not destination.exists():
                    backup.replace(destination)
                raise

            if backup is not None and backup.exists():
                shutil.rmtree(backup)

        finally:
            if staging.exists():
                shutil.rmtree(staging)

        return self.load_profile(destination)

    def load_profile(self, path: Path | str) -> ScanSearchProfile:
        requested_path = Path(path)
        profile_path = requested_path / PROFILE_FILENAME if requested_path.is_dir() else requested_path

        if not profile_path.is_file():
            raise ScanProfileError(f"Profile JSON was not found: {profile_path}")

        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScanProfileError(f"Could not read profile JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise ScanProfileError("The profile JSON must contain an object.")
        if payload.get("schema") != PROFILE_SCHEMA:
            raise ScanProfileError("This file is not a scan search profile.")
        if payload.get("version") != PROFILE_VERSION:
            raise ScanProfileError(
                f"Unsupported profile version: {payload.get('version')!r}."
            )

        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ScanProfileError("The profile name is missing.")

        class_payloads = payload.get("classes")
        if not isinstance(class_payloads, list) or not class_payloads:
            raise ScanProfileError("The profile must contain at least one class.")

        profile_dir = profile_path.parent.resolve()
        legacy_minimum_size_um, legacy_maximum_size_um = self._load_size_requirement(
            payload.get("size_requirement")
        )
        loaded_classes = []
        seen_names: set[str] = set()
        for index, class_payload in enumerate(class_payloads, start=1):
            loaded = self._load_class(
                profile_dir,
                class_payload,
                index,
                legacy_size_requirement=(
                    legacy_minimum_size_um,
                    legacy_maximum_size_um,
                ),
            )
            normalized_name = loaded.name.casefold()
            if normalized_name in seen_names:
                raise ScanProfileError(f"Duplicate class name: {loaded.name}")
            seen_names.add(normalized_name)
            loaded_classes.append(loaded)

        profile = ScanSearchProfile(
            name=name.strip(),
            path=profile_dir,
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            classes=tuple(loaded_classes),
        )
        self.active_profile = profile
        return profile

    def render_class_overlay(self, profile_class: ScanProfileClass) -> np.ndarray:
        source = _read_image(profile_class.source_image, cv2.IMREAD_COLOR)
        if source is None:
            raise ScanProfileError(f"Could not read assets for class {profile_class.name}.")

        mask = self._region_mask_from_image(
            source,
            profile_class.seed_point,
            profile_class.threshold,
            profile_class.connectivity,
            profile_class.name,
        )
        return build_region_overlay(source, mask, profile_class.seed_point)

    def profile_class_to_draft(
        self,
        profile_class: ScanProfileClass,
    ) -> ProfileClassDraft:
        """Reconstruct an editable draft from a saved profile class."""

        source = _read_image(profile_class.source_image, cv2.IMREAD_COLOR)
        if source is None:
            raise ScanProfileError(f"Could not read assets for class {profile_class.name}.")

        mask = self._region_mask_from_image(
            source,
            profile_class.seed_point,
            profile_class.threshold,
            profile_class.connectivity,
            profile_class.name,
        )
        draft = ProfileClassDraft(
            name=profile_class.name,
            source_path=profile_class.source_image,
            image_bgr=source,
            region_mask=mask,
            seed_point=profile_class.seed_point,
            threshold=profile_class.threshold,
            connectivity=profile_class.connectivity,
            minimum_size_um=profile_class.minimum_size_um,
            maximum_size_um=profile_class.maximum_size_um,
        )
        self._validate_draft(draft)
        return draft

    @staticmethod
    def _validate_draft(draft: ProfileClassDraft) -> None:
        image = draft.image_bgr
        mask = draft.region_mask
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ScanProfileError(f"Class {draft.name!r} has an invalid source image.")
        if mask is None or mask.shape != image.shape[:2] or not np.any(mask > 0):
            raise ScanProfileError(f"Class {draft.name!r} has an invalid region mask.")
        if draft.connectivity not in (4, 8):
            raise ScanProfileError("Flood-fill connectivity must be 4 or 8.")
        if not 0 <= int(draft.threshold) <= 255:
            raise ScanProfileError("Flood-fill threshold must be between 0 and 255.")

        seed_x, seed_y = draft.seed_point
        height, width = image.shape[:2]
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ScanProfileError(f"Class {draft.name!r} has an invalid seed point.")
        if mask[seed_y, seed_x] == 0:
            raise ScanProfileError(f"Class {draft.name!r} does not contain its seed point.")

    @staticmethod
    def _average_region_color(draft: ProfileClassDraft) -> tuple[int, int, int]:
        selected_pixels = draft.image_bgr[draft.region_mask > 0]
        if selected_pixels.size == 0:
            raise ScanProfileError(f"Class {draft.name!r} has an empty region.")

        blue, green, red = np.mean(selected_pixels, axis=0)
        return int(round(red)), int(round(green)), int(round(blue))

    @staticmethod
    def _region_mask_from_image(
        image_bgr: np.ndarray,
        seed_point: tuple[int, int],
        threshold: int,
        connectivity: int,
        class_name: str,
    ) -> np.ndarray:
        try:
            _, _, region_mask, contour = get_region_from_point(
                image_bgr=image_bgr,
                seed_point=seed_point,
                threshold=threshold,
                connectivity=connectivity,
            )
        except (cv2.error, ValueError) as exc:
            raise ScanProfileError(
                f"Could not reconstruct the region for class {class_name}."
            ) from exc

        if contour is None or not np.any(region_mask > 0):
            raise ScanProfileError(f"Could not reconstruct the region for class {class_name}.")
        return region_mask

    @staticmethod
    def validate_size_requirement(
        minimum_size_um: float | None,
        maximum_size_um: float | None,
    ) -> tuple[float | None, float | None]:
        for label, value in (
            ("Minimum size", minimum_size_um),
            ("Maximum size", maximum_size_um),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ScanProfileError(f"{label} must be a number of micrometers.")
            if value is not None and value <= 0:
                raise ScanProfileError(f"{label} must be greater than zero.")

        if (
            minimum_size_um is not None
            and maximum_size_um is not None
            and minimum_size_um > maximum_size_um
        ):
            raise ScanProfileError("Minimum size cannot be greater than maximum size.")
        return (
            None if minimum_size_um is None else float(minimum_size_um),
            None if maximum_size_um is None else float(maximum_size_um),
        )

    def _load_size_requirement(
        self,
        payload: object,
    ) -> tuple[float | None, float | None]:
        if payload is None:
            return None, None
        if not isinstance(payload, dict):
            raise ScanProfileError("The profile size requirement must be an object or null.")

        # The pixel-area keys were used briefly before size was defined in um.
        # They cannot be converted without scan scale metadata, so treat them as unset.
        if (
            "minimum_size_um" not in payload
            and "maximum_size_um" not in payload
            and (
                "minimum_area_pixels" in payload
                or "maximum_area_pixels" in payload
            )
        ):
            return None, None

        minimum = payload.get("minimum_size_um")
        maximum = payload.get("maximum_size_um")
        return self.validate_size_requirement(minimum, maximum)

    def _load_class(
        self,
        profile_dir: Path,
        payload: object,
        index: int,
        legacy_size_requirement: tuple[float | None, float | None] = (None, None),
    ) -> ScanProfileClass:
        if not isinstance(payload, dict):
            raise ScanProfileError(f"Class {index} must be a JSON object.")

        class_id = payload.get("id")
        name = payload.get("name")
        if not isinstance(class_id, str) or not class_id:
            raise ScanProfileError(f"Class {index} has no id.")
        if not isinstance(name, str) or not name.strip():
            raise ScanProfileError(f"Class {index} has no name.")

        source_path = self._resolve_asset(profile_dir, payload.get("source_image"), index)

        seed = payload.get("seed_point")
        flood_fill = payload.get("flood_fill")
        if not isinstance(seed, dict) or not isinstance(flood_fill, dict):
            raise ScanProfileError(f"Class {index} is missing region metadata.")

        try:
            seed_point = (int(seed["x"]), int(seed["y"]))
            threshold = int(flood_fill["threshold"])
            connectivity = int(flood_fill["connectivity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScanProfileError(f"Class {index} has invalid region metadata.") from exc

        if connectivity not in (4, 8) or not 0 <= threshold <= 255:
            raise ScanProfileError(f"Class {index} has invalid flood-fill values.")

        source = _read_image(source_path, cv2.IMREAD_COLOR)
        if source is None:
            raise ScanProfileError(f"Could not read source image for class {name}.")
        source_height, source_width = source.shape[:2]
        seed_x, seed_y = seed_point
        if not (0 <= seed_x < source_width and 0 <= seed_y < source_height):
            raise ScanProfileError(f"Class {name} has an invalid seed point.")

        region_mask = self._region_mask_from_image(
            source,
            seed_point,
            threshold,
            connectivity,
            name.strip(),
        )

        if "size_requirement" in payload:
            minimum_size_um, maximum_size_um = self._load_size_requirement(
                payload.get("size_requirement")
            )
        else:
            minimum_size_um, maximum_size_um = legacy_size_requirement

        average_color = payload.get("average_color_rgb")
        if average_color is None:
            blue, green, red = np.mean(source[region_mask > 0], axis=0)
            average_color_rgb = (
                int(round(red)),
                int(round(green)),
                int(round(blue)),
            )
        elif isinstance(average_color, dict):
            try:
                color_values = (
                    average_color["red"],
                    average_color["green"],
                    average_color["blue"],
                )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= value <= 255
                    for value in color_values
                ):
                    raise ValueError
                average_color_rgb = tuple(int(round(value)) for value in color_values)
            except (KeyError, TypeError, ValueError) as exc:
                raise ScanProfileError(
                    f"Class {name} has an invalid average RGB color."
                ) from exc
        else:
            raise ScanProfileError(f"Class {name} has an invalid average RGB color.")

        return ScanProfileClass(
            id=class_id,
            name=name.strip(),
            source_image=source_path,
            seed_point=seed_point,
            threshold=threshold,
            connectivity=connectivity,
            minimum_size_um=minimum_size_um,
            maximum_size_um=maximum_size_um,
            average_color_rgb=average_color_rgb,
        )

    @staticmethod
    def _resolve_asset(profile_dir: Path, relative_path: object, class_index: int) -> Path:
        if not isinstance(relative_path, str) or not relative_path:
            raise ScanProfileError(f"Class {class_index} has a missing asset path.")

        relative = Path(relative_path)
        if relative.is_absolute():
            raise ScanProfileError(f"Class {class_index} asset paths must be relative.")

        resolved = (profile_dir / relative).resolve()
        if resolved != profile_dir and profile_dir not in resolved.parents:
            raise ScanProfileError(f"Class {class_index} asset path leaves the profile folder.")
        if not resolved.is_file():
            raise ScanProfileError(f"Class {class_index} asset was not found: {relative_path}")
        return resolved

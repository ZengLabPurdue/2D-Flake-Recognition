import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from Scanning.region_classifier import (  # noqa: E402
    LEGEND_BOTTOM_LEFT,
    LEGEND_BOTTOM_RIGHT,
    LEGEND_TOP_LEFT,
    LEGEND_TOP_RIGHT,
    classify_contour_regions,
    draw_class_legend,
    load_profile_configuration,
    match_profile_class,
    match_profile_filter,
)
from Scanning.contour_finder import find_flakes  # noqa: E402
from Scanning.scan_profile import (  # noqa: E402
    FILTER_BAD_COLOR,
    FILTER_COLOR_DISTANCE,
    FILTER_INTENSITY_RANGE,
    PROFILE_VERSION,
    ScanProfile,
)


class ScanProfileOptionsTests(unittest.TestCase):
    def _add_class(
        self,
        profile,
        name,
        contrast,
        *,
        group="",
        identify=True,
        minimum=None,
        maximum=None,
    ):
        image = np.full((8, 8, 3), 180, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        profile_class = profile._make_class(
            name,
            Path(f"{name}.png"),
            image,
            mask,
            (3, 3),
            5,
            8,
            minimum,
            maximum,
            contrast=contrast,
            group=group,
            identify=identify,
        )
        profile.classes.append(profile_class)
        return profile_class

    def _add_color_filter(self, profile, filter_type=FILTER_BAD_COLOR):
        image = np.full((8, 8, 3), 180, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        profile_filter = profile._make_filter(
            profile._next_filter_name(),
            filter_type,
            source_path=Path("filter.png"),
            image=image,
            region_mask=mask,
            seed_point=(3, 3),
            threshold=5,
            contrast=(10, 11, 12),
            distance_threshold=20 if filter_type == FILTER_COLOR_DISTANCE else None,
        )
        profile.filters.append(profile_filter)
        return profile_filter

    def test_version_four_round_trip_persists_classes_filters_and_profile_options(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            profile.set_size_requirement(4, 90)
            self._add_class(
                profile,
                "Thin blue",
                (-20, -10, 4),
                group="hBN",
                identify=False,
                minimum=8,
                maximum=40,
            )
            self._add_color_filter(profile)
            profile.filters.append(profile._make_filter(
                profile._next_filter_name(),
                FILTER_INTENSITY_RANGE,
                minimum_intensity=20,
                maximum_intensity=50,
            ))
            self._add_color_filter(profile, FILTER_COLOR_DISTANCE)

            loaded = profile.save_profile("Options")
            payload = json.loads((loaded.path / "profile.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["version"], PROFILE_VERSION)
            self.assertEqual(
                payload["size_requirement"],
                {"minimum_size_um": 4.0, "maximum_size_um": 90.0},
            )
            self.assertEqual(loaded.groups, ["hBN"])
            self.assertFalse(loaded.classes[0]["identify"])
            self.assertEqual(loaded.classes[0]["minimum_size_um"], 8.0)
            self.assertEqual(
                [item["name"] for item in loaded.filters],
                ["Filter 1", "Filter 2", "Filter 3"],
            )
            self.assertEqual(
                [item["type"] for item in loaded.filters],
                [FILTER_BAD_COLOR, FILTER_INTENSITY_RANGE, FILTER_COLOR_DISTANCE],
            )

    def test_version_two_profiles_default_to_identify(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            self._add_class(profile, "Legacy", (1, 2, 3))
            saved = profile.save_profile("Legacy")
            profile_path = saved.path / "profile.json"
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            payload["version"] = 2
            payload.pop("size_requirement", None)
            for item in payload["classes"]:
                item.pop("group", None)
                item.pop("identify", None)
                item.pop("reject", None)
            profile_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = ScanProfile(directory).load_profile(profile_path)

            self.assertEqual(loaded.classes[0]["group"], "")
            self.assertTrue(loaded.classes[0]["identify"])
            self.assertEqual(loaded.filters, [])

    def test_copy_and_extend_make_independent_classes_with_provenance(self):
        source = ScanProfile()
        source.name = "Source"
        original = self._add_class(source, "Class A", (5, 6, 7), group="Group A")

        copied = source.copy_class(0)
        copied["image_bgr"][0, 0] = 0

        self.assertEqual(copied["name"], "Class A copy")
        self.assertFalse(np.array_equal(copied["image_bgr"], original["image_bgr"]))
        self.assertEqual(copied["derived_from"]["class_id"], original["id"])

        destination = ScanProfile()
        self._add_class(destination, "Class A", (0, 0, 0))
        added, added_filters = destination.extend_from_profile(source)

        self.assertEqual([item["name"] for item in added], ["Class A 2", "Class A copy"])
        self.assertTrue(all(item["derived_from"]["profile"] == "Source" for item in added))
        self.assertEqual(added_filters, [])

    def test_saved_copy_provenance_points_to_the_stable_source_id(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            source = self._add_class(profile, "Source", (1, 2, 3))
            copied = profile.copy_class(0)

            saved = profile.save_profile("Copies")
            payload = json.loads((saved.path / "profile.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["classes"][0]["id"], source["id"])
            self.assertEqual(payload["classes"][1]["id"], copied["id"])
            self.assertEqual(
                payload["classes"][1]["derived_from"]["class_id"],
                payload["classes"][0]["id"],
            )

    def test_filters_precede_identify_and_disabled_classes_are_ignored(self):
        classes = [
            {
                "name": "Nearest identify",
                "contrast_rgb": np.asarray((10, 10, 10), dtype=float),
                "tolerance": 10,
                "identify": True,
                "minimum_size_um": None,
                "maximum_size_um": None,
            },
            {
                "name": "Disabled exact",
                "contrast_rgb": np.asarray((12, 12, 12), dtype=float),
                "tolerance": 10,
                "identify": False,
                "minimum_size_um": None,
                "maximum_size_um": None,
            },
        ]

        filters = [{
            "name": "Filter 1",
            "type": FILTER_BAD_COLOR,
            "contrast_rgb": np.asarray((14, 14, 14), dtype=float),
            "tolerance": 10,
        }]
        filtered = match_profile_filter(np.asarray((12, 12, 12)), 12, filters)
        match = None if filtered else match_profile_class(np.asarray((12, 12, 12)), classes)

        self.assertEqual(filtered[1]["name"], "Filter 1")
        self.assertIsNone(match)

    def test_all_three_filter_types_match_their_documented_condition(self):
        bad_color = [{
            "name": "Filter 1",
            "type": FILTER_BAD_COLOR,
            "contrast_rgb": np.asarray((10, 10, 10), dtype=float),
            "tolerance": 2,
        }]
        intensity = [{
            "name": "Filter 2",
            "type": FILTER_INTENSITY_RANGE,
            "minimum_intensity": 40,
            "maximum_intensity": 60,
        }]
        far_color = [{
            "name": "Filter 3",
            "type": FILTER_COLOR_DISTANCE,
            "contrast_rgb": np.asarray((0, 0, 0), dtype=float),
            "distance_threshold": 10,
        }]

        self.assertIsNotNone(match_profile_filter(np.asarray((11, 9, 10)), 100, bad_color))
        self.assertIsNotNone(match_profile_filter(np.asarray((0, 0, 0)), 50, intensity))
        self.assertIsNotNone(match_profile_filter(np.asarray((20, 0, 0)), 100, far_color))
        self.assertIsNone(match_profile_filter(np.asarray((5, 0, 0)), 100, far_color))

    def test_class_size_limits_are_applied_during_matching(self):
        classes = [
            {
                "name": "Small",
                "contrast_rgb": np.asarray((1, 1, 1), dtype=float),
                "tolerance": 2,
                "identify": True,
                "minimum_size_um": 2,
                "maximum_size_um": 5,
            }
        ]

        self.assertIsNone(match_profile_class(np.asarray((1, 1, 1)), classes, size_um=6))
        self.assertEqual(
            match_profile_class(np.asarray((1, 1, 1)), classes, size_um=4)[1]["name"],
            "Small",
        )

    def test_matching_loader_reads_profile_size_groups_and_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(
                json.dumps({
                    "size_requirement": {
                        "minimum_size_um": 3,
                        "maximum_size_um": 30,
                    },
                    "classes": [{
                        "name": "Class",
                        "contrast_rgb": {"red": 1, "green": 2, "blue": 3},
                        "flood_fill": {"threshold": 4},
                        "group": "Artifacts",
                        "identify": True,
                        "size_requirement": None,
                    }],
                    "filters": [{
                        "name": "Filter 1",
                        "type": FILTER_INTENSITY_RANGE,
                        "intensity_range": {"minimum": 10, "maximum": 30},
                    }],
                }),
                encoding="utf-8",
            )

            configuration = load_profile_configuration(profile_path)

            self.assertEqual(configuration["minimum_size_um"], 3.0)
            self.assertEqual(configuration["maximum_size_um"], 30.0)
            self.assertEqual(configuration["classes"][0]["group"], "Artifacts")
            self.assertEqual(configuration["filters"][0]["name"], "Filter 1")

    def test_version_three_reject_classes_migrate_to_automatic_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ScanProfile(directory)
            self._add_class(profile, "Old reject", (4, 5, 6), identify=False)
            saved = profile.save_profile("Migration")
            profile_path = saved.path / "profile.json"
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            payload["version"] = 3
            payload.pop("filters", None)
            payload["classes"][0]["reject"] = True
            profile_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = ScanProfile(directory).load_profile(profile_path)
            configuration = load_profile_configuration(profile_path)

            self.assertEqual(loaded.classes, [])
            self.assertEqual(loaded.filters[0]["name"], "Filter 1")
            self.assertEqual(loaded.filters[0]["type"], FILTER_BAD_COLOR)
            self.assertEqual(configuration["classes"], [])
            self.assertEqual(configuration["filters"][0]["type"], FILTER_BAD_COLOR)

    def test_region_classification_filters_before_classes_and_applies_profile_size(self):
        image = np.full((12, 12, 3), 100, dtype=np.uint8)
        image[2:10, 2:10] = 120
        contour = np.asarray([[[2, 2]], [[9, 2]], [[9, 9]], [[2, 9]]], dtype=np.int32)
        hierarchy = np.asarray([[-1, -1, -1, -1]], dtype=np.int32)
        identify_class = {
            "name": "Class",
            "contrast_rgb": np.asarray((20, 20, 20), dtype=float),
            "tolerance": 2,
            "class_index": 0,
            "group": "Material",
            "identify": True,
            "minimum_size_um": None,
            "maximum_size_um": None,
            "display_color_rgb": (255, 0, 0),
        }
        filters = [{
            "name": "Filter 1",
            "type": FILTER_BAD_COLOR,
            "contrast_rgb": np.asarray((20, 20, 20), dtype=float),
            "tolerance": 2,
        }]

        filtered = classify_contour_regions(
            image,
            np.zeros((12, 12), dtype=np.uint8),
            [contour],
            hierarchy,
            [contour],
            [0],
            {0: [0]},
            [identify_class],
            2,
            pixel_size_um=2,
            profile_filters=filters,
        )
        too_small = classify_contour_regions(
            image,
            np.zeros((12, 12), dtype=np.uint8),
            [contour],
            hierarchy,
            [contour],
            [0],
            {0: [0]},
            [identify_class],
            2,
            pixel_size_um=2,
            minimum_size_um=20,
            profile_filters=filters,
        )

        self.assertTrue(filtered["region_results"][0]["filtered"])
        self.assertEqual(filtered["region_results"][0]["filtered_by"], "Filter 1")
        self.assertIsNone(filtered["region_results"][0]["matched_class"])
        self.assertFalse(too_small["region_results"][0]["inside_profile_size"])
        self.assertFalse(too_small["region_results"][0]["filtered"])

    def test_in_memory_profile_configuration_runs_the_preview_pipeline(self):
        image = np.full((100, 100, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (79, 79), (70, 70, 70), -1)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[20:80, 20:80] = 255
        profile = ScanProfile()
        profile.classes.append(profile._make_class(
            "Preview class",
            Path("preview.png"),
            image,
            mask,
            (40, 40),
            255,
            8,
            None,
            None,
            contrast=(110, 110, 110),
        ))

        classified, contours, details = find_flakes(
            image,
            area_threshold=10,
            return_details=True,
            profile_configuration=profile.matching_configuration(),
            color_seed=0,
            legend_position=LEGEND_TOP_RIGHT,
        )

        self.assertEqual(classified.shape, image.shape)
        self.assertEqual(len(contours), 1)
        self.assertEqual(details["legend_position"], LEGEND_TOP_RIGHT)
        self.assertEqual(
            details["region_results"][0]["matched_class"],
            "Preview class",
        )

    def test_class_legend_supports_each_image_corner(self):
        image = np.zeros((200, 240, 3), dtype=np.uint8)
        classes = [{
            "name": "A",
            "group": "",
            "identify": True,
            "display_color_rgb": (255, 80, 40),
        }]
        expectations = {
            LEGEND_TOP_LEFT: (False, False),
            LEGEND_TOP_RIGHT: (True, False),
            LEGEND_BOTTOM_LEFT: (False, True),
            LEGEND_BOTTOM_RIGHT: (True, True),
        }

        for position, (expect_right, expect_bottom) in expectations.items():
            with self.subTest(position=position):
                rendered = draw_class_legend(image, classes, position=position)
                ys, xs = np.nonzero(np.any(rendered != 0, axis=2))

                self.assertGreater(len(xs), 0)
                self.assertEqual(float(np.mean(xs)) > image.shape[1] / 2, expect_right)
                self.assertEqual(float(np.mean(ys)) > image.shape[0] / 2, expect_bottom)

        with self.assertRaises(ValueError):
            draw_class_legend(image, classes, position="center")


if __name__ == "__main__":
    unittest.main()

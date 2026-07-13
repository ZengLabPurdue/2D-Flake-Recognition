import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "App"))

from Scanning.contour_extractor import get_region_from_point
from Scanning.scan_profile import (  # noqa: E402
    ProfileClassDraft,
    ScanProfileError,
    ScanProfileStore,
)


class ScanProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ScanProfileStore(Path(self.temp_dir.name) / "Profiles")

        self.image = np.zeros((40, 50, 3), dtype=np.uint8)
        self.image[10:25, 12:30] = (30, 100, 200)
        _, _, self.mask, contour = get_region_from_point(
            self.image,
            (15, 15),
            threshold=5,
        )
        self.assertIsNotNone(contour)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_draft(self, minimum_size_um=None, maximum_size_um=None):
        return ProfileClassDraft(
            name="Thin flake",
            source_path=Path("sample.png"),
            image_bgr=self.image,
            region_mask=self.mask,
            seed_point=(15, 15),
            threshold=5,
            minimum_size_um=minimum_size_um,
            maximum_size_um=maximum_size_um,
        )

    def test_save_and_load_round_trip(self):
        profile = self.store.save_profile(
            "Graphene Search",
            [self.make_draft(minimum_size_um=12.5, maximum_size_um=500)],
        )

        self.assertEqual(profile.path.name, "Graphene_Search")
        self.assertEqual(profile.name, "Graphene Search")
        self.assertEqual(len(profile.classes), 1)
        self.assertEqual(profile.classes[0].minimum_size_um, 12.5)
        self.assertEqual(profile.classes[0].maximum_size_um, 500)
        self.assertEqual(profile.classes[0].average_color_rgb, (200, 100, 30))

        profile_json = json.loads(
            (profile.path / "profile.json").read_text(encoding="utf-8")
        )
        class_json = profile_json["classes"][0]
        self.assertNotIn("size_requirement", profile_json)
        self.assertEqual(
            class_json["size_requirement"],
            {"minimum_size_um": 12.5, "maximum_size_um": 500.0},
        )
        self.assertEqual(
            class_json["average_color_rgb"],
            {"red": 200, "green": 100, "blue": 30},
        )
        self.assertFalse(Path(class_json["source_image"]).is_absolute())
        self.assertNotIn("region_image", class_json)
        self.assertNotIn("pixel_count", class_json)
        self.assertNotIn("bounding_box", class_json)
        self.assertFalse((profile.path / "regions").exists())

        loaded = self.store.load_profile(profile.path)
        self.assertEqual(loaded.classes[0].seed_point, (15, 15))
        self.assertEqual(loaded.classes[0].minimum_size_um, 12.5)
        self.assertEqual(loaded.classes[0].maximum_size_um, 500)
        self.assertEqual(loaded.classes[0].average_color_rgb, (200, 100, 30))
        self.assertEqual(self.store.active_profile, loaded)
        editable_class = self.store.profile_class_to_draft(loaded.classes[0])
        self.assertEqual(editable_class.name, "Thin flake")
        self.assertEqual(np.count_nonzero(editable_class.region_mask), 270)
        self.assertTrue(np.array_equal(editable_class.image_bgr, self.image))
        self.assertEqual(
            self.store.render_class_overlay(loaded.classes[0]).shape,
            self.image.shape,
        )

    def test_overwrite_replaces_existing_profile(self):
        self.store.save_profile("Profile", [self.make_draft()])

        with self.assertRaises(FileExistsError):
            self.store.save_profile("Profile", [self.make_draft()])

        overwritten = self.store.save_profile(
            "Profile",
            [self.make_draft()],
            overwrite=True,
        )
        self.assertEqual(len(overwritten.classes), 1)

    def test_load_rejects_asset_path_outside_profile(self):
        profile = self.store.save_profile("Profile", [self.make_draft()])
        profile_path = profile.path / "profile.json"
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        payload["classes"][0]["source_image"] = "../outside.png"
        profile_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ScanProfileError, "leaves the profile folder"):
            self.store.load_profile(profile_path)

    def test_region_selection_rejects_out_of_bounds_seed(self):
        with self.assertRaisesRegex(ValueError, "outside the image"):
            get_region_from_point(self.image, (100, 100))

    def test_size_requirement_is_optional_and_validated(self):
        profile = self.store.save_profile("No Size", [self.make_draft()])
        self.assertIsNone(profile.classes[0].minimum_size_um)
        self.assertIsNone(profile.classes[0].maximum_size_um)

        # Older class entries are reconstructed from source image + seed point.
        profile_path = profile.path / "profile.json"
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        class_payload = payload["classes"][0]
        class_payload.pop("size_requirement")
        class_payload.pop("average_color_rgb")
        class_payload["region_image"] = "regions/old_region.png"
        class_payload["pixel_count"] = 270
        class_payload["bounding_box"] = {
            "x": 12,
            "y": 10,
            "width": 18,
            "height": 15,
        }
        profile_path.write_text(json.dumps(payload), encoding="utf-8")
        legacy_profile = self.store.load_profile(profile_path)
        self.assertIsNone(legacy_profile.classes[0].minimum_size_um)
        self.assertIsNone(legacy_profile.classes[0].maximum_size_um)
        self.assertEqual(legacy_profile.classes[0].average_color_rgb, (200, 100, 30))

        payload["size_requirement"] = {
            "minimum_size_um": 10,
            "maximum_size_um": 20,
        }
        profile_path.write_text(json.dumps(payload), encoding="utf-8")
        migrated_profile = self.store.load_profile(profile_path)
        self.assertEqual(migrated_profile.classes[0].minimum_size_um, 10)
        self.assertEqual(migrated_profile.classes[0].maximum_size_um, 20)

        with self.assertRaisesRegex(ScanProfileError, "cannot be greater"):
            self.store.save_profile(
                "Bad Size",
                [self.make_draft(minimum_size_um=500, maximum_size_um=100)],
            )


if __name__ == "__main__":
    unittest.main()

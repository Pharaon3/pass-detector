"""Pass-centric crop schedule (native-length windows, no time-warp)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from dataset import (
    build_pass_centric_schedule,
    parse_training_clip_mode,
    pass_centric_crop_indices,
)


class TestPassCentric(unittest.TestCase):
    def test_parse_mode(self) -> None:
        self.assertEqual(parse_training_clip_mode({"training_clip_mode": "full_clip"}), "full_clip")
        self.assertEqual(parse_training_clip_mode({}), "full_clip")
        with self.assertRaises(ValueError):
            parse_training_clip_mode({"training_clip_mode": "nope"})

    def test_crop_indices_pass(self) -> None:
        lo, hi = pass_centric_crop_indices(100, 20, 750, random_negative=False)
        self.assertEqual(lo, 60)
        self.assertEqual(hi, 140)
        self.assertEqual(hi - lo + 1, 81)

    def test_crop_indices_neg_deterministic(self) -> None:
        lo, hi = pass_centric_crop_indices(-1, 20, 750, random_negative=False)
        self.assertEqual(hi - lo + 1, 81)

    def test_build_schedule_one_pass(self) -> None:
        items = [(Path("x.mp4"), Path("y.json"))]
        cfg = {
            "class_names": ["pass"],
            "fps": 25,
            "num_frames": 750,
            "multi_label": True,
            "label_radius_frames": {"default": 10, "pass": 20},
            "strict_labels": False,
        }
        fake_events = [{"time_sec": 4.0, "class": "pass"}]

        with patch("dataset.load_events_json", return_value=fake_events):
            sch = build_pass_centric_schedule(items, cfg)
        self.assertEqual(len(sch), 1)
        clip_idx, center, R = sch[0]
        self.assertEqual(clip_idx, 0)
        self.assertEqual(R, 20)
        self.assertEqual(center, 100)


if __name__ == "__main__":
    unittest.main()

"""Tests for score_events matching."""

from __future__ import annotations

import unittest

from score_events import match_prediction_to_gt


class TestScoreEvents(unittest.TestCase):
    def test_match_same_class_within_tol(self) -> None:
        preds = [{"time": 5.0, "event": "pass", "confidence": 0.9}]
        gts = [{"time_sec": 5.2, "label": "pass"}]
        r = match_prediction_to_gt(preds, gts, tolerance_sec=1.0)
        self.assertEqual(r.n_matched, 1)
        self.assertEqual(r.n_pred_only, 0)
        self.assertEqual(r.n_gt_only, 0)
        self.assertAlmostEqual(r.mean_conf_all or 0.0, 0.9)

    def test_pred_only_wrong_time(self) -> None:
        preds = [{"time": 0.0, "event": "pass", "confidence": 0.8}]
        gts = [{"time_sec": 5.0, "label": "pass"}]
        r = match_prediction_to_gt(preds, gts, tolerance_sec=1.0)
        self.assertEqual(r.n_matched, 0)
        self.assertEqual(r.n_pred_only, 1)
        self.assertEqual(r.n_gt_only, 1)

    def test_greedy_one_to_one(self) -> None:
        preds = [
            {"time": 5.0, "event": "pass", "confidence": 0.7},
            {"time": 5.3, "event": "pass", "confidence": 0.8},
        ]
        gts = [{"time_sec": 5.1, "label": "pass"}]
        r = match_prediction_to_gt(preds, gts, tolerance_sec=1.0)
        self.assertEqual(r.n_matched, 1)
        self.assertEqual(r.n_pred_only, 1)
        self.assertEqual(r.n_gt_only, 0)

    def test_gt_class_filter(self) -> None:
        preds = [{"time": 1.0, "event": "pass", "confidence": 0.5}]
        gts = [
            {"time_sec": 1.05, "label": "pass"},
            {"time_sec": 2.0, "label": "shot"},
        ]
        r = match_prediction_to_gt(preds, gts, tolerance_sec=1.0, gt_class_filter={"pass"})
        self.assertEqual(r.n_matched, 1)
        self.assertEqual(r.n_gt_only, 0)


if __name__ == "__main__":
    unittest.main()

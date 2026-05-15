"""Lightweight sanity tests for label radius parsing and multi-label postprocess."""

from __future__ import annotations

import unittest

import torch

import numpy as np

from postprocess import (
    MultilabelPostprocessParams,
    build_multilabel_postprocess_params,
    frame_confidence_export,
    postprocess_clip,
    postprocess_config_from_cfg,
    postprocess_multilabel_advanced,
)
from utils.early_stopping import EarlyStopping, EarlyStoppingMulti, build_early_stopper
from utils.labels import (
    events_to_frame_labels,
    filter_events_to_config_classes,
    parse_label_radius_frames,
)


class TestLabelRadius(unittest.TestCase):
    def test_int_backward_compat(self) -> None:
        spec = parse_label_radius_frames(10, ["pass", "shot"])
        self.assertEqual(spec, 10)

    def test_dict_per_class(self) -> None:
        spec = parse_label_radius_frames({"default": 10, "pass": 8}, ["pass", "shot"])
        self.assertEqual(spec, {"pass": 8, "shot": 10})


class TestStrictLabels(unittest.TestCase):
    def test_strict_unknown_raises(self) -> None:
        events = [{"time_sec": 1.0, "class": "not_a_real_class"}]
        class_to_idx = {"pass": 0}
        with self.assertRaises(ValueError) as ctx:
            events_to_frame_labels(
                events,
                class_to_idx,
                num_frames=100,
                fps=25,
                multi_label=True,
                radius_frames=5,
                strict_labels=True,
                label_source="/tmp/x.json",
            )
        self.assertIn("not_a_real_class", str(ctx.exception))


class TestFilterEventsToConfig(unittest.TestCase):
    def test_keeps_only_config_classes(self) -> None:
        events = [
            {"time_sec": 1.0, "class": "pass"},
            {"time_sec": 2.0, "label": "shot"},
            {"time_sec": 3.0, "class": "pass_received"},
        ]
        class_to_idx = {"pass": 0}
        out = filter_events_to_config_classes(events, class_to_idx)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["class"], "pass")


class TestEarlyStopping(unittest.TestCase):
    def test_max_mode_improves_and_patience(self) -> None:
        es = EarlyStopping("val_f1", "max", patience=2, min_delta=0.01)
        stop, imp = es.step({"val_f1": 0.4})
        self.assertFalse(stop)
        self.assertTrue(imp)
        stop, imp = es.step({"val_f1": 0.42})
        self.assertFalse(stop)
        self.assertTrue(imp)
        stop, imp = es.step({"val_f1": 0.42})
        self.assertFalse(stop)
        self.assertFalse(imp)
        stop, imp = es.step({"val_f1": 0.42})
        self.assertTrue(stop)
        self.assertFalse(imp)

    def test_min_mode(self) -> None:
        es = EarlyStopping("val_loss", "min", patience=1, min_delta=0.0)
        es.step({"val_loss": 1.0})
        stop, _ = es.step({"val_loss": 1.0})
        self.assertTrue(stop)

    def test_invalid_mode(self) -> None:
        with self.assertRaises(ValueError):
            EarlyStopping("val_f1", "auto", patience=2, min_delta=0.0)

    def test_invalid_patience(self) -> None:
        with self.assertRaises(ValueError):
            EarlyStopping("val_f1", "max", patience=0, min_delta=0.0)

    def test_missing_monitor_key(self) -> None:
        es = EarlyStopping("val_f1", "max", patience=2, min_delta=0.0)
        with self.assertRaises(KeyError):
            es.step({"val_loss": 0.1})


class TestEarlyStoppingMulti(unittest.TestCase):
    def test_loss_improvement_resets_patience_when_f1_flat(self) -> None:
        es = EarlyStoppingMulti(
            [("val_f1", "max"), ("val_loss", "min")],
            patience=2,
            min_delta=0.01,
        )
        stop, imp, names = es.step({"val_f1": 0.2, "val_loss": 1.0})
        self.assertFalse(stop)
        self.assertTrue(imp)
        self.assertEqual(set(names), {"val_f1", "val_loss"})
        stop, imp, names = es.step({"val_f1": 0.2, "val_loss": 0.5})
        self.assertFalse(stop)
        self.assertTrue(imp)
        self.assertEqual(names, ["val_loss"])
        stop, imp, names = es.step({"val_f1": 0.2, "val_loss": 0.5})
        self.assertFalse(stop)
        self.assertFalse(imp)
        self.assertEqual(names, [])
        stop, imp, names = es.step({"val_f1": 0.2, "val_loss": 0.5})
        self.assertTrue(stop)
        self.assertFalse(imp)

    def test_build_early_stopper_monitors(self) -> None:
        es = build_early_stopper(
            {
                "patience": 1,
                "min_delta": 0.0,
                "monitors": [{"metric": "val_loss", "mode": "min"}],
            }
        )
        self.assertIsInstance(es, EarlyStoppingMulti)


class TestPostprocess(unittest.TestCase):
    def test_nested_postprocess_params_build(self) -> None:
        cfg = {
            "fps": 25,
            "num_frames": 750,
            "num_classes": 2,
            "class_names": ["a", "b"],
            "multi_label": True,
            "activation": "sigmoid",
            "threshold": 0.2,
            "min_event_gap_sec": 0.2,
            "postprocess": {
                "smoothing": {"enabled": False, "window_frames": 5},
                "peak_picking": {"enabled": True, "mode": "local_max"},
                "thresholds": {"default": 0.2, "a": 0.1},
                "min_gap_sec": {"default": 0.2},
                "top_k_per_class": None,
                "top_k_total": None,
            },
        }
        ml = build_multilabel_postprocess_params(cfg)
        self.assertEqual(ml.thresholds[0], 0.1)
        self.assertEqual(ml.thresholds[1], 0.2)
        self.assertEqual(ml.peak_picking_mode, "local_max")

    def test_invalid_peak_picking_mode(self) -> None:
        cfg = {
            "fps": 25,
            "num_frames": 750,
            "num_classes": 2,
            "class_names": ["a", "b"],
            "multi_label": True,
            "activation": "sigmoid",
            "threshold": 0.2,
            "min_event_gap_sec": 0.2,
            "postprocess": {
                "peak_picking": {"enabled": True, "mode": "typo"},
                "thresholds": {"default": 0.2},
                "min_gap_sec": {"default": 0.2},
            },
        }
        with self.assertRaises(ValueError) as ctx:
            build_multilabel_postprocess_params(cfg)
        self.assertIn("peak_picking.mode", str(ctx.exception))

    def test_postprocess_clip_end_to_end(self) -> None:
        cfg = {
            "fps": 10,
            "num_frames": 5,
            "num_classes": 2,
            "class_names": ["a", "b"],
            "multi_label": True,
            "activation": "sigmoid",
            "threshold": 0.2,
            "min_event_gap_sec": 0.1,
        }
        pp = postprocess_config_from_cfg(cfg)
        logits = torch.zeros((1, 5, 2))
        logits[0, 2, 0] = 5.0
        ev = postprocess_clip(logits, pp)
        self.assertTrue(any(e["event"] == "a" for e in ev))

    def test_plateau_mid_single_run(self) -> None:
        """Plateau L..R: i_out = (argmax_index + (L+R)//2) // 2; one candidate per run."""
        probs = np.zeros((20, 1), dtype=np.float64)
        probs[10:15, 0] = 0.9
        params = MultilabelPostprocessParams(
            fps=10.0,
            class_names=["pass"],
            thresholds=[0.5],
            min_gap_frames=[1],
            smoothing_enabled=False,
            smoothing_window_frames=1,
            peak_picking_enabled=True,
            peak_picking_mode="plateau_mid",
            top_k_per_class=None,
            top_k_total=None,
        )
        ev = postprocess_multilabel_advanced(probs, params)
        self.assertEqual(len(ev), 1)
        # L=10,R=14 -> i_mid=12, i_top=10 -> i_out=11
        self.assertEqual(ev[0]["frame"], 11)
        self.assertAlmostEqual(ev[0]["confidence"], 0.9)

    def test_frame_confidence_export(self) -> None:
        logits = torch.zeros((1, 5, 1))
        logits[0, 2, 0] = 8.0
        d = frame_confidence_export(logits, "sigmoid", ["pass"], fps=10.0, prob_decimals=4)
        self.assertEqual(d["schema"], "per_frame_class_probs_v1")
        self.assertEqual(d["num_frames"], 5)
        self.assertEqual(d["class_names"], ["pass"])
        self.assertEqual(len(d["probs"]), 5)
        self.assertEqual(len(d["probs"][0]), 1)
        self.assertGreater(d["probs"][2][0], 0.99)


if __name__ == "__main__":
    unittest.main()

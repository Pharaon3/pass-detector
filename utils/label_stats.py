"""
Label JSON statistics without decoding video (same radius / strict rules as training).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dataset import (
    build_pass_centric_schedule,
    discover_clip_items,
    parse_allowed_environments,
    parse_training_clip_mode,
    pass_centric_crop_indices,
)
from utils.dataset_video import load_stem_to_relpath
from utils.labels import (
    build_class_to_idx,
    events_to_frame_labels,
    filter_events_to_config_classes,
    load_events_json,
    parse_label_radius_frames,
)


def aggregate_frame_counts_from_json(
    cfg: Dict[str, Any],
    data_root: str | Path,
    split: str | None,
    video_dir: str = "videos",
    labels_dir: str = "labels",
    stems: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, int, List[Path]]:
    """
    Returns:
        pos_frames: shape [C] — per-class count of frames with positive label
        neg_frames: shape [C] — per-class count of frames with label 0
        num_frames_per_clip: T
        label_paths: list of label JSON paths processed
    """
    root = Path(data_root)
    class_names: List[str] = list(cfg["class_names"])
    num_classes = len(class_names)
    class_to_idx = build_class_to_idx(class_names)
    radius = parse_label_radius_frames(cfg.get("label_radius_frames", 10), class_names)
    strict = bool(cfg.get("strict_labels", False))
    multi_label = bool(cfg.get("multi_label", True))
    fps = int(cfg["fps"])
    T = int(cfg["num_frames"])
    stem_to_rel = load_stem_to_relpath(root)
    vdir = root / video_dir
    ldir = root / labels_dir
    allowed = parse_allowed_environments(cfg)
    items = discover_clip_items(
        root, split, vdir, ldir, stem_to_rel, stems=stems, allowed_environments=allowed
    )

    pos = np.zeros(num_classes, dtype=np.float64)
    clip_mode = parse_training_clip_mode(cfg)

    if not multi_label:
        if clip_mode == "pass_centric":
            samples = build_pass_centric_schedule(items, cfg)
            y_cache: Dict[int, torch.Tensor] = {}
            for ci in {s[0] for s in samples}:
                _, lpath = items[ci]
                events = filter_events_to_config_classes(
                    load_events_json(lpath), class_to_idx
                )
                y_cache[ci] = events_to_frame_labels(
                    events,
                    class_to_idx,
                    T,
                    fps,
                    multi_label,
                    radius,
                    strict_labels=strict,
                    label_source=lpath,
                )
            sum_l = 0
            for clip_idx, center, R in samples:
                y_full = y_cache[clip_idx]
                lo, hi = pass_centric_crop_indices(center, R, T, random_negative=(center < 0))
                y_w = y_full[lo : hi + 1]
                arr = y_w.numpy().astype(np.int64)
                sum_l += int(arr.shape[0])
                for c in range(num_classes):
                    pos[c] += int(np.sum(arr == c))
            neg = float(sum_l) - pos
            return pos, neg, T, [p for _, p in items]

        for _, lpath in items:
            events = filter_events_to_config_classes(
                load_events_json(lpath), class_to_idx
            )
            y = events_to_frame_labels(
                events,
                class_to_idx,
                T,
                fps,
                multi_label,
                radius,
                strict_labels=strict,
                label_source=lpath,
            )
            arr = y.numpy().astype(np.int64)
            for c in range(num_classes):
                pos[c] += int(np.sum(arr == c))
        neg = float(len(items) * T) - pos
        return pos, neg, T, [p for _, p in items]

    if clip_mode == "pass_centric":
        samples = build_pass_centric_schedule(items, cfg)
        y_cache: Dict[int, torch.Tensor] = {}
        for ci in {s[0] for s in samples}:
            _, lpath = items[ci]
            events = filter_events_to_config_classes(
                load_events_json(lpath), class_to_idx
            )
            y_cache[ci] = events_to_frame_labels(
                events,
                class_to_idx,
                T,
                fps,
                True,
                radius,
                strict_labels=strict,
                label_source=lpath,
            )
        sum_l = 0
        for clip_idx, center, R in samples:
            y_full = y_cache[clip_idx]
            lo, hi = pass_centric_crop_indices(center, R, T, random_negative=(center < 0))
            y_w = y_full[lo : hi + 1]
            sum_l += int(y_w.shape[0])
            pos += y_w.numpy().astype(np.float64).sum(axis=0)
        neg = float(sum_l) - pos
        return pos, neg, T, [p for _, p in items]

    for _, lpath in items:
        events = filter_events_to_config_classes(load_events_json(lpath), class_to_idx)
        y = events_to_frame_labels(
            events,
            class_to_idx,
            T,
            fps,
            True,
            radius,
            strict_labels=strict,
            label_source=lpath,
        )
        yf = y.numpy()
        pos += yf.sum(axis=0)
    neg = float(len(items) * T) - pos
    return pos, neg, T, [p for _, p in items]


def compute_auto_pos_weight_numpy(
    cfg: Dict[str, Any],
    data_root: str | Path,
    split: str | None,
    clip_max: float,
    video_dir: str = "videos",
    labels_dir: str = "labels",
    stems: Optional[List[str]] = None,
) -> np.ndarray:
    """pos_weight[c] = neg[c] / max(pos[c], 1), then clipped to clip_max."""
    pos, neg, _, _ = aggregate_frame_counts_from_json(
        cfg,
        data_root,
        split,
        video_dir=video_dir,
        labels_dir=labels_dir,
        stems=stems,
    )
    w = neg / np.maximum(pos, 1.0)
    w = np.minimum(w, float(clip_max))
    return w.astype(np.float32)


def _json_event_class_name(ev: Dict[str, Any]) -> str | None:
    for key in ("class", "event", "label", "name"):
        if key in ev:
            return str(ev[key])
    return None


def count_events_per_class_from_json(
    cfg: Dict[str, Any],
    data_root: str | Path,
    split: str | None,
    video_dir: str = "videos",
    labels_dir: str = "labels",
) -> Tuple[Dict[str, int], List[Tuple[str, str, Path]]]:
    """
    Raw event counts from JSON (class field), plus list of unknown (name, stem, path).

    Does not apply strict_labels — caller checks unknowns against config.
    """
    root = Path(data_root)
    class_names = list(cfg["class_names"])
    known = set(class_names)
    stem_to_rel = load_stem_to_relpath(root)
    allowed = parse_allowed_environments(cfg)
    items = discover_clip_items(
        root,
        split,
        root / video_dir,
        root / labels_dir,
        stem_to_rel,
        allowed_environments=allowed,
    )
    counts = {n: 0 for n in class_names}
    unknowns: List[Tuple[str, str, Path]] = []
    for _, lpath in items:
        events = load_events_json(lpath)
        stem = lpath.stem
        for ev in events:
            name = _json_event_class_name(ev)
            if name is None:
                continue
            if name in known:
                counts[name] += 1
            else:
                unknowns.append((name, stem, lpath))
    return counts, unknowns

"""
Helpers to map JSON event annotations to per-frame targets.

Label JSON is intentionally generic (no SoccerNet). Example:

{
  "events": [
    {"time_sec": 4.8, "class": "pass"},
    {"time": 12.0, "event": "shot"}
  ]
}

Field aliases supported: time_sec / time / t for seconds; class / event / label for name.

Per-class label radius:
  `label_radius_frames` may be a single int (same half-width for every class) or a mapping
  with a required ``default`` key plus optional per-class overrides (see config.yaml).
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# int: same radius for all classes; dict: maps class name -> half-width in frames (must include every class_name)
RadiusSpec = Union[int, Dict[str, int]]


def load_events_json(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "events" in data:
        return list(data["events"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported label format in {path}")


def load_label_environment(path: str | Path) -> Optional[str]:
    """
    Read top-level ``environment`` from a label JSON dict (e.g. ``night``, ``dry``).

    Returns ``None`` if missing or not a string. Used to filter clips for training.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    env = data.get("environment")
    if env is None:
        return None
    s = str(env).strip()
    return s if s else None


def filter_events_to_config_classes(
    events: List[Dict[str, Any]],
    class_to_idx: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Keep only events whose class/event/label/name field is a key of ``class_to_idx``.

    Use this when label JSON still lists many action types but the model is trained
    on a subset (e.g. only ``pass``). Events without a resolvable class name are dropped.
    """
    out: List[Dict[str, Any]] = []
    for ev in events:
        try:
            name = _event_class_name(ev)
        except KeyError:
            continue
        if name in class_to_idx:
            out.append(ev)
    return out


def _event_time_seconds(ev: Dict[str, Any]) -> float:
    for key in ("time_sec", "time", "t", "timestamp_sec"):
        if key in ev:
            return float(ev[key])
    raise KeyError(f"Event has no time field: {ev}")


def _event_class_name(ev: Dict[str, Any]) -> str:
    for key in ("class", "event", "label", "name"):
        if key in ev:
            return str(ev[key])
    raise KeyError(f"Event has no class field: {ev}")


def parse_label_radius_frames(raw: Any, class_names: List[str]) -> RadiusSpec:
    """
    Normalize config ``label_radius_frames`` for training / stats.

    - If ``raw`` is int: return that int (global radius, backward compatible).
    - If ``raw`` is dict: require ``default``; return a dict with an entry for every
      name in ``class_names`` using ``raw[name]`` if present else ``default``.
    """
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, dict):
        if "default" not in raw:
            raise ValueError(
                'label_radius_frames dict must include a "default" key; '
                f"got keys: {sorted(raw.keys())}"
            )
        default = int(raw["default"])
        return {name: int(raw.get(name, default)) for name in class_names}
    raise TypeError(
        f"label_radius_frames must be int or dict with 'default', got {type(raw).__name__}"
    )


def radius_frames_for_class(class_name: str, spec: RadiusSpec) -> int:
    """Half-width in frames for stamping positives around one event of ``class_name``."""
    if isinstance(spec, int):
        return int(spec)
    return int(spec[class_name])


def _format_label_source(label_source: str | Path | None) -> str:
    if label_source is None:
        return "<unknown label path>"
    return str(Path(label_source))


def _collect_unknown_labels(
    events: List[Dict[str, Any]],
    class_to_idx: Dict[str, int],
) -> List[Tuple[str, Dict[str, Any]]]:
    unknown: List[Tuple[str, Dict[str, Any]]] = []
    for ev in events:
        try:
            name = _event_class_name(ev)
        except KeyError:
            continue
        if name not in class_to_idx:
            unknown.append((name, ev))
    return unknown


def events_to_frame_labels(
    events: List[Dict[str, Any]],
    class_to_idx: Dict[str, int],
    num_frames: int,
    fps: int,
    multi_label: bool,
    radius_frames: RadiusSpec,
    strict_labels: bool = False,
    label_source: str | Path | None = None,
) -> torch.Tensor:
    """
    Build dense frame labels of shape [T, C] (multi-label float soft targets) or [T] long (multi-class).

    ``radius_frames`` is either a global int or a per-class dict (see ``parse_label_radius_frames``).

    Unknown event class names (relative to ``class_to_idx``):
    - If ``strict_labels`` is True: raises ``ValueError`` listing the unknown name(s), the
      label file path (if provided), and valid class names.
    - If False: emits a ``UserWarning`` per distinct unknown name (includes path/stem) and
      skips those events (they are not stamped into the tensor).

    Callers that load mixed JSON (many action types) with a subset model should pass only
    events whose class is in ``class_to_idx`` (e.g. via ``filter_events_to_config_classes``)
    so extra types never reach this function.

    multi_label True: returns float tensor [T, num_classes] with triangular soft targets
        around each event (peak 1.0 at center, linear falloff to 0 at ±radius).
    multi_class: returns long tensor [T] with values in [0, num_classes] where index 0
        is reserved for background if "background" is in class_to_idx; otherwise uses 0
        as background only where no event spans a frame (see below).

    For multi-class without explicit background in class_names, unlabeled frames are 0
    and the first class index in class_to_idx may collide — users should add "background"
    as first class when using multi-class per-frame classification.
    """
    valid_names = list(class_to_idx.keys())
    unknown_pairs = _collect_unknown_labels(events, class_to_idx)
    src = _format_label_source(label_source)
    stem = Path(label_source).stem if label_source else None

    if unknown_pairs:
        unique_unknown = sorted({name for name, _ in unknown_pairs})
        if strict_labels:
            raise ValueError(
                f"Unknown label class name(s) {unique_unknown} in {src}. "
                f"Valid class names: {valid_names}. "
                f"Set strict_labels: false to warn and skip, or fix the JSON."
            )
        skipped = len(unknown_pairs)
        msg = (
            f"Skipped {skipped} unknown label event(s); unknown name(s)={unique_unknown!r}; "
            f"source={src}; stem={stem!r}. Valid class names: {valid_names}"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)

    num_classes = len(class_to_idx)
    if multi_label:
        y = torch.zeros((num_frames, num_classes), dtype=torch.float32)
        for ev in events:
            try:
                name = _event_class_name(ev)
            except KeyError:
                continue
            if name not in class_to_idx:
                continue
            c = class_to_idx[name]
            r = radius_frames_for_class(name, radius_frames)
            t_sec = _event_time_seconds(ev)
            center = int(round(t_sec * fps))
            lo = max(0, center - r)
            hi = min(num_frames - 1, center + r)
            denom = max(r, 1)
            idx = torch.arange(lo, hi + 1, dtype=torch.float32)
            tri = torch.clamp(1.0 - (idx - float(center)).abs() / float(denom), min=0.0)
            y[lo : hi + 1, c] = torch.maximum(y[lo : hi + 1, c], tri)
        return y

    if "background" in class_to_idx:
        bg = class_to_idx["background"]
    else:
        bg = 0
    y = torch.full((num_frames,), bg, dtype=torch.long)
    for ev in events:
        try:
            name = _event_class_name(ev)
        except KeyError:
            continue
        if name not in class_to_idx:
            continue
        c = class_to_idx[name]
        r = radius_frames_for_class(name, radius_frames)
        t_sec = _event_time_seconds(ev)
        center = int(round(t_sec * fps))
        lo = max(0, center - r)
        hi = min(num_frames - 1, center + r)
        y[lo : hi + 1] = int(c)
    return y


def build_class_to_idx(class_names: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(class_names)}

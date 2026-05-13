"""
Turn per-frame logits into a list of detected events (JSON-serializable dicts).

- activation from config: sigmoid (multi-label) or softmax (multi-class)
- multi-label: per-class threshold and min-gap (seconds), optional temporal smoothing,
  optional peak picking, optional top-k caps
- time_sec = frame_index / fps

Backward compatible with a single global ``threshold`` and ``min_event_gap_sec`` in YAML,
or with a nested ``postprocess:`` block (see README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MultilabelPostprocessParams:
    """Resolved multi-label detection settings (one entry per class index)."""

    fps: float
    class_names: List[str]
    thresholds: List[float]
    min_gap_frames: List[int]
    smoothing_enabled: bool
    smoothing_window_frames: int
    peak_picking_enabled: bool
    top_k_per_class: Optional[List[Optional[int]]]
    top_k_total: Optional[int]


@dataclass
class PostprocessConfig:
    fps: float
    activation: Literal["sigmoid", "softmax"]
    threshold: float
    min_event_gap_sec: float
    class_names: List[str]
    multi_label: bool
    multilabel: Optional[MultilabelPostprocessParams] = None


def logits_to_probs(logits: torch.Tensor, activation: str) -> torch.Tensor:
    """
    logits: [T, num_classes] or [1, T, C] -> squeeze batch if present
    returns probs same shape [T, C]
    """
    if logits.dim() == 3:
        if logits.size(0) != 1:
            raise ValueError("Expected batch size 1 for inference postprocess")
        logits = logits.squeeze(0)
    if activation == "sigmoid":
        return torch.sigmoid(logits)
    if activation == "softmax":
        return F.softmax(logits, dim=-1)
    raise ValueError(f"Unknown activation: {activation}")


def resolve_per_class_float(
    value: Union[int, float, Dict[str, Any]],
    class_names: Sequence[str],
) -> List[float]:
    """
    If scalar: repeat for all classes.
    If dict: require ``default``; each class uses ``value[name]`` if present else default.
    """
    if isinstance(value, (int, float)):
        f = float(value)
        return [f] * len(class_names)
    if isinstance(value, dict):
        if "default" not in value:
            raise ValueError(f"Per-class mapping must include 'default', got keys {sorted(value.keys())}")
        d0 = float(value["default"])
        return [float(value.get(name, d0)) for name in class_names]
    raise TypeError(f"Expected number or dict, got {type(value).__name__}")


def resolve_optional_topk_per_class(
    value: Any,
    class_names: Sequence[str],
) -> Optional[List[Optional[int]]]:
    """
    None: no per-class top-k.
    int: same K for every class (may be 0 meaning drop all — avoid in config).
    dict: optional per-class with ``default`` (default may be null meaning unlimited).
    """
    if value is None:
        return None
    if isinstance(value, int):
        k = int(value)
        return [k] * len(class_names)
    if isinstance(value, dict):
        if "default" not in value:
            raise ValueError("top_k_per_class dict must include 'default'")
        raw_def = value["default"]
        out: List[Optional[int]] = []
        for name in class_names:
            raw = value.get(name, raw_def)
            if raw is None:
                out.append(None)
            else:
                out.append(int(raw))
        return out
    raise TypeError(f"top_k_per_class: expected null, int, or dict, got {type(value).__name__}")


def build_multilabel_postprocess_params(cfg: Dict[str, Any]) -> MultilabelPostprocessParams:
    """Merge legacy top-level keys with optional ``postprocess`` nested block."""
    class_names = list(cfg["class_names"])
    fps = float(cfg["fps"])
    fallback_th = float(cfg.get("threshold", 0.5))
    fallback_gap_sec = float(cfg.get("min_event_gap_sec", 1.0))

    pp = cfg.get("postprocess")
    use_nested = isinstance(pp, dict) and any(
        k in pp
        for k in (
            "thresholds",
            "min_gap_sec",
            "smoothing",
            "peak_picking",
            "top_k_per_class",
            "top_k_total",
        )
    )

    if use_nested:
        assert isinstance(pp, dict)
        th_src = pp.get("thresholds", fallback_th)
        gap_src = pp.get("min_gap_sec", fallback_gap_sec)
        sm = pp.get("smoothing") or {}
        pk = pp.get("peak_picking") or {}
        smoothing_enabled = bool(sm.get("enabled", False))
        smoothing_window = int(sm.get("window_frames", 5))
        peak_enabled = bool(pk.get("enabled", False))
        top_k_pc = resolve_optional_topk_per_class(pp.get("top_k_per_class"), class_names)
        top_k_tot = pp.get("top_k_total")
        top_k_total = int(top_k_tot) if top_k_tot is not None else None
    else:
        th_src = fallback_th
        gap_src = fallback_gap_sec
        smoothing_enabled = False
        smoothing_window = 5
        peak_enabled = False
        top_k_pc = None
        top_k_total = None

    thresholds = resolve_per_class_float(th_src, class_names)
    min_gap_sec = resolve_per_class_float(gap_src, class_names)
    min_gap_frames = [max(1, int(round(sec * fps))) for sec in min_gap_sec]

    return MultilabelPostprocessParams(
        fps=fps,
        class_names=class_names,
        thresholds=thresholds,
        min_gap_frames=min_gap_frames,
        smoothing_enabled=smoothing_enabled,
        smoothing_window_frames=max(1, smoothing_window),
        peak_picking_enabled=peak_enabled,
        top_k_per_class=top_k_pc,
        top_k_total=top_k_total,
    )


def postprocess_config_from_cfg(cfg: Dict[str, Any]) -> PostprocessConfig:
    ml = build_multilabel_postprocess_params(cfg) if bool(cfg.get("multi_label", True)) else None
    return PostprocessConfig(
        fps=float(cfg["fps"]),
        activation=str(cfg.get("activation", "sigmoid")),  # type: ignore[arg-type]
        threshold=float(cfg.get("threshold", 0.5)),
        min_event_gap_sec=float(cfg.get("min_event_gap_sec", 1.0)),
        class_names=list(cfg["class_names"]),
        multilabel=ml,
        multi_label=bool(cfg.get("multi_label", True)),
    )


def temporal_moving_average_probs(arr: np.ndarray, window_frames: int) -> np.ndarray:
    """Simple moving average along time for each class column. ``arr`` shape [T, C]."""
    t, c = arr.shape
    w = int(window_frames)
    if w <= 1:
        return arr
    pad_left = (w - 1) // 2
    pad_right = w - 1 - pad_left
    kernel = np.ones(w, dtype=np.float64) / w
    out = np.empty((t, c), dtype=np.float64)
    for ci in range(c):
        col = arr[:, ci].astype(np.float64, copy=False)
        padded = np.pad(col, (pad_left, pad_right), mode="edge")
        conv = np.convolve(padded, kernel, mode="valid")
        out[:, ci] = conv
    return out.astype(arr.dtype, copy=False)


def local_peak_mask_1d(col: np.ndarray) -> np.ndarray:
    """Boolean mask where ``col[t]`` is a local maximum (ties allowed at plateaus)."""
    t = col.shape[0]
    m = np.zeros(t, dtype=bool)
    if t == 1:
        m[0] = True
        return m
    m[0] = col[0] >= col[1]
    for i in range(1, t - 1):
        m[i] = col[i] >= col[i - 1] and col[i] >= col[i + 1]
    m[t - 1] = col[t - 1] >= col[t - 2]
    return m


def _nms_per_class_gaps(
    candidates: List[Dict[str, Any]],
    min_gap_frames: List[int],
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """Greedy NMS on pre-sorted (by confidence desc) list; gap depends on event class name."""
    idx = {name: i for i, name in enumerate(class_names)}
    kept: List[Dict[str, Any]] = []
    for ev in candidates:
        f = int(ev["frame"])
        cls = ev["event"]
        gap = int(min_gap_frames[idx[cls]])
        ok = True
        for k in kept:
            if k["event"] != cls:
                continue
            if abs(int(k["frame"]) - f) < gap:
                ok = False
                break
        if ok:
            kept.append(ev)
    return kept


def _apply_top_k_per_class(
    events: List[Dict[str, Any]],
    limits: Optional[List[Optional[int]]],
    class_names: List[str],
) -> List[Dict[str, Any]]:
    if limits is None:
        return events
    from collections import defaultdict

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        groups[str(ev["event"])].append(ev)
    out: List[Dict[str, Any]] = []
    for ci, name in enumerate(class_names):
        lim = limits[ci]
        lst = groups.get(name, [])
        if lim is None:
            out.extend(lst)
            continue
        lst = sorted(lst, key=lambda e: float(e["confidence"]), reverse=True)
        out.extend(lst[: max(0, int(lim))])
    return out


def _apply_top_k_total(events: List[Dict[str, Any]], k: Optional[int]) -> List[Dict[str, Any]]:
    if k is None:
        return events
    ranked = sorted(events, key=lambda e: float(e["confidence"]), reverse=True)
    return ranked[: max(0, int(k))]


def postprocess_multilabel(
    probs: np.ndarray,
    class_names: List[str],
    fps: float,
    threshold: float,
    min_gap_frames: int,
) -> List[Dict[str, Any]]:
    """
    Legacy entry: single threshold and single min gap (frames).
    ``probs`` shape [T, C].
    """
    params = MultilabelPostprocessParams(
        fps=fps,
        class_names=class_names,
        thresholds=[float(threshold)] * len(class_names),
        min_gap_frames=[int(min_gap_frames)] * len(class_names),
        smoothing_enabled=False,
        smoothing_window_frames=1,
        peak_picking_enabled=False,
        top_k_per_class=None,
        top_k_total=None,
    )
    return postprocess_multilabel_advanced(probs, params)


def postprocess_multilabel_advanced(
    probs: np.ndarray,
    params: MultilabelPostprocessParams,
) -> List[Dict[str, Any]]:
    arr = np.asarray(probs, dtype=np.float64, order="C")
    if params.smoothing_enabled and params.smoothing_window_frames > 1:
        arr = temporal_moving_average_probs(arr, params.smoothing_window_frames)
    t, c = arr.shape
    assert c == len(params.class_names)
    candidates: List[Dict[str, Any]] = []
    for ci in range(c):
        col = arr[:, ci]
        th = float(params.thresholds[ci])
        peaks = local_peak_mask_1d(col) if params.peak_picking_enabled else np.ones(t, dtype=bool)
        for fi in range(t):
            if not peaks[fi]:
                continue
            p = float(col[fi])
            if p < th:
                continue
            candidates.append(
                {
                    "frame": fi,
                    "time": fi / params.fps,
                    "event": params.class_names[ci],
                    "confidence": p,
                }
            )
    candidates.sort(key=lambda e: e["confidence"], reverse=True)
    selected = _nms_per_class_gaps(candidates, params.min_gap_frames, params.class_names)
    selected = _apply_top_k_per_class(selected, params.top_k_per_class, params.class_names)
    selected = _apply_top_k_total(selected, params.top_k_total)
    selected.sort(key=lambda e: int(e["frame"]))
    return selected


def postprocess_multiclass(
    probs: np.ndarray,
    class_names: List[str],
    fps: float,
    threshold: float,
    min_gap_frames: int,
) -> List[Dict[str, Any]]:
    """
    probs: [T, C] from softmax — emit onset frames when argmax is class c with prob >= th.
    """
    t, c = probs.shape
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    events: List[Dict[str, Any]] = []
    prev = -1
    for fi in range(t):
        cls = int(pred[fi])
        p = float(conf[fi])
        name = class_names[cls]
        if name == "background":
            prev = cls
            continue
        if p < threshold:
            prev = cls
            continue
        if cls != prev:
            events.append(
                {
                    "frame": fi,
                    "time": fi / fps,
                    "event": name,
                    "confidence": p,
                }
            )
        prev = cls
    events.sort(key=lambda e: e["confidence"], reverse=True)
    events = _nms_per_class_gaps(events, [int(min_gap_frames)] * len(class_names), class_names)
    events.sort(key=lambda e: e["frame"])
    return events


def postprocess_clip(
    logits: torch.Tensor,
    cfg: PostprocessConfig,
) -> List[Dict[str, Any]]:
    """
    logits: [1, T, num_classes] on CPU or CUDA
    """
    probs = logits_to_probs(logits, cfg.activation)
    arr = probs.detach().float().cpu().numpy()
    if cfg.multi_label:
        if cfg.multilabel is not None:
            return postprocess_multilabel_advanced(arr, cfg.multilabel)
        min_gap_frames = max(1, int(round(cfg.min_event_gap_sec * cfg.fps)))
        return postprocess_multilabel(arr, cfg.class_names, cfg.fps, cfg.threshold, min_gap_frames)
    min_gap_frames = max(1, int(round(cfg.min_event_gap_sec * cfg.fps)))
    return postprocess_multiclass(arr, cfg.class_names, cfg.fps, cfg.threshold, min_gap_frames)

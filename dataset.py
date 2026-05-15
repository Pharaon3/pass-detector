"""
PyTorch Dataset: paired video + JSON annotations.

Layouts:
  - Flat: ``{root}/{video_dir}/{stem}.mp4`` and ``{root}/{labels_dir}/{stem}.json``
  - Per-clip subfolders: ``{video_dir}/{stem}/{stem}.mp4`` with label either next to it
    (``{labels_dir}/{stem}/{stem}.json``) or flat (``{labels_dir}/{stem}.json``).

Each clip is resized temporally to `num_frames` at `fps` (see utils.video).
Frame targets are built via utils.labels.events_to_frame_labels on label events whose
class is listed in config ``class_names`` (see utils.labels.filter_events_to_config_classes);
other event types in the JSON files are ignored.

Training scope (``training_clip_mode`` in config):
  - ``full_clip`` (default): one dataset index per video; model sees the whole ``num_frames`` window
    (e.g. 750). Inference uses the same length.
  - ``pass_centric``: one index per annotated pass (plus one random window for clips with no pass).
    Each sample is a **native-length** crop of model-time frames ``[pass - 2R, pass + 2R]``
    (``R`` = ``label_radius_frames`` for that class), length ``4*R+1`` when unobstructed — **no**
    temporal resampling. Batches pad shorter clips to ``max T`` in the batch with normalized black;
    training / val loss uses ``valid_mask`` so padded frames are ignored. Inference still uses full
    ``num_frames`` clips (unchanged in ``infer.py`` / ``batch_infer_and_score.py``).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import AbstractSet, Any, Dict, FrozenSet, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from utils.dataset_video import VIDEO_EXTS, load_stem_to_relpath, resolve_clip_video_path
from utils.labels import (
    _event_class_name,
    _event_time_seconds,
    build_class_to_idx,
    events_to_frame_labels,
    filter_events_to_config_classes,
    load_events_json,
    load_label_environment,
    parse_label_radius_frames,
    radius_frames_for_class,
)
from utils.video import VideoPreprocessConfig, preprocess_clip_to_tensor

_EXT_SET = frozenset(VIDEO_EXTS)


def parse_training_clip_mode(cfg: Dict[str, Any]) -> str:
    mode = str(cfg.get("training_clip_mode", "full_clip")).strip().lower()
    if mode not in ("full_clip", "pass_centric"):
        raise ValueError(
            f'training_clip_mode must be "full_clip" or "pass_centric", got {mode!r}'
        )
    return mode


def pass_centric_crop_indices(center: int, R: int, num_frames: int, *, random_negative: bool) -> Tuple[int, int]:
    """
    Inclusive ``[lo, hi]`` frame indices on the full clip timeline (0 .. num_frames-1).

    For a pass at ``center`` use half-span ``2 * R`` each side (user: ``2 * label_radius_frames``).
    For ``center == -1`` (no-pass clip) use a window of length ``min(num_frames, 4*R+1)``:
    random start if ``random_negative`` else centered for deterministic stats.
    """
    T = int(num_frames)
    if center < 0:
        win = min(T, 4 * int(R) + 1)
        if win >= T:
            return 0, T - 1
        if random_negative:
            lo = random.randint(0, T - win)
        else:
            lo = max(0, (T - win) // 2)
        return lo, lo + win - 1
    half = 2 * int(R)
    lo = max(0, int(center) - half)
    hi = min(T - 1, int(center) + half)
    if lo > hi:
        return 0, T - 1
    return lo, hi


def build_pass_centric_schedule(
    items: List[Tuple[Path, Path]],
    cfg: Dict[str, Any],
) -> List[Tuple[int, int, int]]:
    """
    One entry per training step for ``pass_centric`` mode.

    Returns list of ``(clip_idx, center_frame, R)``:
      - ``center_frame >= 0``: crop around that pass; ``R`` is that event class label radius.
      - ``center_frame == -1``: clip has no in-config events; ``R`` is max class radius;
        random temporal window at load time.
    """
    class_names: List[str] = list(cfg["class_names"])
    class_to_idx = build_class_to_idx(class_names)
    fps = int(cfg["fps"])
    T = int(cfg["num_frames"])
    radius_spec = parse_label_radius_frames(cfg.get("label_radius_frames", 10), class_names)
    R_ref = max(radius_frames_for_class(n, radius_spec) for n in class_names)
    schedule: List[Tuple[int, int, int]] = []
    for i, (_, lpath) in enumerate(items):
        events = filter_events_to_config_classes(load_events_json(lpath), class_to_idx)
        per: List[Tuple[int, int]] = []
        for ev in events:
            try:
                name = _event_class_name(ev)
            except KeyError:
                continue
            if name not in class_to_idx:
                continue
            R = radius_frames_for_class(name, radius_spec)
            t_sec = _event_time_seconds(ev)
            c = int(round(t_sec * float(fps)))
            c = max(0, min(T - 1, c))
            per.append((c, R))
        if not per:
            schedule.append((i, -1, R_ref))
        else:
            for c, R in per:
                schedule.append((i, c, R))
    return schedule


def parse_allowed_environments(config: Dict[str, Any]) -> Optional[FrozenSet[str]]:
    """
    Parse ``training_environments`` from config.

    - ``null`` / missing → ``None`` (no filter; all clips).
    - Non-empty list of strings → frozenset of stripped names (clip must match exactly).
    - Empty YAML list ``[]`` → error (ambiguous).
    """
    raw = config.get("training_environments")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        out = {str(x).strip() for x in raw if str(x).strip()}
        if not out:
            raise ValueError(
                "config training_environments is empty; omit the key or set null to use all "
                "environments, or list one or more names (e.g. night, dry, snow, child)."
            )
        return frozenset(out)
    raise TypeError(
        f"training_environments must be a list of strings or null, got {type(raw).__name__}"
    )


def filter_clip_items_by_environments(
    items: List[Tuple[Path, Path]],
    allowed: Optional[AbstractSet[str]],
) -> List[Tuple[Path, Path]]:
    """Keep only (video, label) pairs whose label JSON ``environment`` is in ``allowed``."""
    if not allowed:
        return items
    out: List[Tuple[Path, Path]] = []
    for vp, lp in items:
        env = load_label_environment(lp)
        if env is None or env not in allowed:
            continue
        out.append((vp, lp))
    return out


def _label_json_for_stem(labels_dir: Path, stem: str) -> Optional[Path]:
    flat = labels_dir / f"{stem}.json"
    if flat.is_file():
        return flat
    nested = labels_dir / stem / f"{stem}.json"
    if nested.is_file():
        return nested
    return None


def discover_clip_items(
    root: Path,
    split: Optional[str],
    video_dir: Path,
    labels_dir: Path,
    stem_to_rel: Dict[str, str],
    stems: Optional[List[str]] = None,
    allowed_environments: Optional[AbstractSet[str]] = None,
) -> List[Tuple[Path, Path]]:
    """
    Return (video_path, label_json_path) pairs the same way SoccerClipDataset does.

    Used by training utilities that need label paths without decoding video.

    Args:
        stems: If set, resolve exactly these stems (in order). Mutually exclusive with ``split``.
        allowed_environments: If set, keep only clips whose label JSON top-level ``environment``
            string is in this set. Clips with missing ``environment`` are dropped.
    """
    if stems is not None and split:
        raise ValueError("Pass only one of split or stems to discover_clip_items")
    if stems is not None:
        if not stems:
            raise RuntimeError(
                "stems list is empty. Need at least one clip stem for train/val or training."
            )
        return [_resolve_item(root, s, video_dir, labels_dir, stem_to_rel) for s in stems]
    if split:
        split_file = root / split
        stems = [s.strip() for s in split_file.read_text(encoding="utf-8").splitlines() if s.strip()]
        if not stems:
            raise RuntimeError(
                f"Split file {split_file} is empty. Run `python dataset/materialize_from_manifest.py` "
                f"to rebuild train.txt from your videos, or list one stem per line."
            )
        items: List[Tuple[Path, Path]] = []
        for stem in stems:
            items.append(_resolve_item(root, stem, video_dir, labels_dir, stem_to_rel))
        return items

    vids = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in _EXT_SET)
    items = []
    seen_video_paths: set[str] = set()
    for vp in vids:
        lp = labels_dir / f"{vp.stem}.json"
        if lp.is_file():
            items.append((vp, lp))
            seen_video_paths.add(str(vp.resolve()))
        else:
            continue
    for stem in stem_to_rel:
        vp = resolve_clip_video_path(root, stem, video_dir, stem_to_rel)
        if vp is None:
            continue
        key = str(vp.resolve())
        if key in seen_video_paths:
            continue
        lp = _label_json_for_stem(labels_dir, stem)
        if lp is not None:
            seen_video_paths.add(key)
            items.append((vp, lp))

    for sub in sorted(p for p in video_dir.iterdir() if p.is_dir()):
        stem = sub.name
        vp: Optional[Path] = None
        for ext in VIDEO_EXTS:
            cand = sub / f"{stem}{ext}"
            if cand.is_file():
                vp = cand
                break
        if vp is None:
            continue
        lp = _label_json_for_stem(labels_dir, stem)
        if lp is None:
            continue
        key = str(vp.resolve())
        if key in seen_video_paths:
            continue
        seen_video_paths.add(key)
        items.append((vp, lp))

    items = filter_clip_items_by_environments(items, allowed_environments)
    if not items:
        hint = ""
        if allowed_environments:
            hint = (
                f" (no clips left after training_environments={sorted(allowed_environments)!r}; "
                "check label JSON or widen the filter)"
            )
        raise RuntimeError(f"No video/label pairs found under {root}{hint}")
    return items


def _resolve_item(
    root: Path,
    stem: str,
    video_dir: Path,
    labels_dir: Path,
    stem_to_rel: Dict[str, str],
) -> Tuple[Path, Path]:
    lp = _label_json_for_stem(labels_dir, stem)
    if lp is None:
        raise FileNotFoundError(
            f"Missing label JSON for stem '{stem}' under {labels_dir} "
            f"(tried {labels_dir / f'{stem}.json'} and {labels_dir / stem / f'{stem}.json'})"
        )
    vp = resolve_clip_video_path(root, stem, video_dir, stem_to_rel)
    if vp is None:
        rel = stem_to_rel.get(stem)
        hints: list[str] = []
        if rel:
            r = str(rel).replace("\\", "/")
            hints.append(str(root / r))
            hints.append(str(video_dir / r))
        hint = f" (tried: {'; '.join(hints)})" if hints else ""
        raise FileNotFoundError(f"No video for stem '{stem}'{hint}")
    return vp, lp


class SoccerClipDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        config: Dict[str, Any],
        split: Optional[str] = None,
        video_dir: str = "videos",
        labels_dir: str = "labels",
        video_backend: str = "opencv",
        stems: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            root: dataset root containing video_dir and labels_dir
            config: loaded YAML dict (must include fps, num_frames, class_names, multi_label, ...)
            split: optional; if provided, expects `root/split.txt` listing basenames (one per line)
            stems: optional explicit list of basenames (same as split file lines). Mutually exclusive with split.

        ``config['training_clip_mode']``: ``full_clip`` (default) or ``pass_centric`` (see module docstring).
        """
        self.root = Path(root)
        self.config = config
        self.video_dir = self.root / video_dir
        self.labels_dir = self.root / labels_dir
        self.multi_label = bool(config.get("multi_label", True))
        self.fps = int(config["fps"])
        self.num_frames = int(config["num_frames"])
        self.class_names: List[str] = list(config["class_names"])
        self.class_to_idx = build_class_to_idx(self.class_names)
        self.label_radius_spec = parse_label_radius_frames(
            config.get("label_radius_frames", 10),
            self.class_names,
        )
        self.strict_labels = bool(config.get("strict_labels", False))
        self.backend = video_backend
        self.vpre = VideoPreprocessConfig(
            fps=self.fps,
            num_frames=self.num_frames,
            backend=video_backend,  # type: ignore[arg-type]
        )

        if stems is not None and split:
            raise ValueError("SoccerClipDataset: pass only one of split or stems")
        self._stem_to_rel = load_stem_to_relpath(self.root)
        allowed_env = parse_allowed_environments(self.config)
        self.items = discover_clip_items(
            self.root,
            split,
            self.video_dir,
            self.labels_dir,
            self._stem_to_rel,
            stems=stems,
            allowed_environments=allowed_env,
        )

        self.training_clip_mode = parse_training_clip_mode(config)
        self._pass_samples: Optional[List[Tuple[int, int, int]]] = None
        if self.training_clip_mode == "pass_centric":
            self._pass_samples = build_pass_centric_schedule(self.items, config)

    def __len__(self) -> int:
        if self.training_clip_mode == "pass_centric":
            return len(self._pass_samples or ())
        return len(self.items)

    def _getitem_pass_centric(self, idx: int) -> Dict[str, Any]:
        assert self._pass_samples is not None
        clip_idx, center, R = self._pass_samples[idx]
        vpath, lpath = self.items[clip_idx]
        events = filter_events_to_config_classes(
            load_events_json(lpath), self.class_to_idx
        )
        video = preprocess_clip_to_tensor(vpath, self.vpre, device=None)  # [T,3,H,W]
        labels = events_to_frame_labels(
            events,
            self.class_to_idx,
            self.num_frames,
            self.fps,
            self.multi_label,
            self.label_radius_spec,
            strict_labels=self.strict_labels,
            label_source=lpath,
        )
        lo, hi = pass_centric_crop_indices(
            center, R, self.num_frames, random_negative=(center < 0)
        )
        vid_w = video[lo : hi + 1]
        lab_w = labels[lo : hi + 1]
        return {
            "video": vid_w,
            "labels": lab_w,
            "video_path": str(vpath),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.training_clip_mode == "pass_centric":
            return self._getitem_pass_centric(idx)
        vpath, lpath = self.items[idx]
        events = filter_events_to_config_classes(
            load_events_json(lpath), self.class_to_idx
        )
        video = preprocess_clip_to_tensor(vpath, self.vpre, device=None)  # [T,3,H,W]
        labels = events_to_frame_labels(
            events,
            self.class_to_idx,
            self.num_frames,
            self.fps,
            self.multi_label,
            self.label_radius_spec,
            strict_labels=self.strict_labels,
            label_source=lpath,
        )
        return {
            "video": video,  # [T,3,H,W]
            "labels": labels,
            "video_path": str(vpath),
        }

"""
Resolve clip video paths for SoccerClipDataset.

Supports:
  - Flat layout: {data_root}/videos/{stem}.mp4 (and other common extensions)
  - Per-clip folder: {videos_dir}/{stem}/{stem}.mp4 (e.g. dataset/private/<id>/<id>.mp4)
  - Nested manifest path at dataset root: {data_root}/{path} e.g. clip_4/224p.mp4
  - Nested manifest path under videos: {data_root}/videos/{path} e.g. videos/clip_4/224p.mp4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def stem_for_manifest_rel(rel: str) -> str:
    """Same stem convention as dataset/materialize_from_manifest.py."""
    p = Path(str(rel).replace("\\", "/"))
    parent = p.parent.as_posix().replace("/", "_") if p.parent.as_posix() not in (".", "") else ""
    base = p.stem
    if parent:
        return f"{parent}_{base}"
    return base


def load_stem_to_relpath(data_root: Path) -> Dict[str, str]:
    """Map label stem -> manifest relative path from train.json + valid.json under data_root."""
    mapping: Dict[str, str] = {}
    for name in ("train.json", "valid.json"):
        manifest = data_root / name
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for item in data.get("videos", []):
            rel = str(item["path"]).replace("\\", "/")
            stem = stem_for_manifest_rel(rel)
            mapping[stem] = rel
    return mapping


def resolve_clip_video_path(
    data_root: Path,
    stem: str,
    videos_dir: Path,
    stem_to_rel: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """
    Return an existing video file path for this label stem, or None.

    Order: flat videos_dir first, then {data_root}/{rel}, then {videos_dir}/{rel}.
    """
    for ext in VIDEO_EXTS:
        flat = videos_dir / f"{stem}{ext}"
        if flat.is_file():
            return flat

    stem_dir = videos_dir / stem
    if stem_dir.is_dir():
        for ext in VIDEO_EXTS:
            per_clip = stem_dir / f"{stem}{ext}"
            if per_clip.is_file():
                return per_clip

    rel = (stem_to_rel or load_stem_to_relpath(data_root)).get(stem)
    if not rel:
        return None
    rel_norm = str(rel).replace("\\", "/")
    nested_root = data_root / rel_norm
    if nested_root.is_file():
        return nested_root
    nested_videos = videos_dir / rel_norm
    if nested_videos.is_file():
        return nested_videos
    return None

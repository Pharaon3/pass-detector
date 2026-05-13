"""
Video loading and tensor preprocessing for fixed-length clips.

Output tensor shape: [T, 3, 224, 224] with T == num_frames (e.g. 750).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import torch
# ImageNet normalization (applied in float01 space after resize)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class VideoPreprocessConfig:
    fps: int = 25
    num_frames: int = 750
    height: int = 224
    width: int = 224
    backend: Literal["opencv", "pyav"] = "opencv"


def _load_indices_opencv(
    path: str | Path,
    target_fps: int,
    num_frames: int,
) -> Tuple[list[np.ndarray], float]:
    """Sample up to num_frames at target_fps using OpenCV; returns (bgr_frames, src_fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1e-3:
        src_fps = float(target_fps)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total / src_fps if src_fps > 0 else num_frames / target_fps
    clip_duration = num_frames / target_fps
    effective_duration = min(duration_sec, clip_duration)
    # Avoid degenerate timelines when metadata reports zero duration
    effective_duration = max(float(effective_duration), 1.0 / float(target_fps))

    # Target timestamps for each output frame (center of each bin)
    out_times = (np.arange(num_frames, dtype=np.float64) + 0.5) / target_fps
    out_times = np.minimum(out_times, effective_duration - 1e-6)

    # Map each output time to nearest source frame index
    src_indices = np.clip(
        np.round(out_times * src_fps).astype(np.int64),
        0,
        max(total - 1, 0),
    )

    frames: list[np.ndarray] = []
    # Sequential forward decode is much faster than CAP_PROP_POS_FRAMES seeks
    # (many codecs re-decode from keyframes on each seek).
    mono = bool(np.all(src_indices[1:] >= src_indices[:-1])) if len(src_indices) > 1 else True
    if mono:
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        current_idx = -1
        buf: Optional[np.ndarray] = None
        for want in src_indices.tolist():
            while current_idx < int(want):
                ok, frame = cap.read()
                if not ok or frame is None:
                    frame = black
                buf = frame
                current_idx += 1
            assert buf is not None
            frames.append(buf)
    else:
        last_idx: int = -1
        last_frame: Optional[np.ndarray] = None
        for idx in src_indices:
            if int(idx) != last_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
                last_idx = int(idx)
                last_frame = frame
            assert last_frame is not None
            frames.append(last_frame)

    cap.release()
    return frames, float(src_fps)


def _load_indices_pyav(
    path: str | Path,
    target_fps: int,
    num_frames: int,
) -> Tuple[list[np.ndarray], float]:
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    src_fps = float(stream.average_rate) if stream.average_rate else float(target_fps)
    time_base = float(stream.time_base) if stream.time_base else 1.0 / src_fps

    # Decode all frames with timestamps (memory-heavy for long files; clips are short)
    decoded: list[Tuple[float, np.ndarray]] = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            t = float(frame.pts * time_base) if frame.pts is not None else len(decoded) / src_fps
            img = frame.to_ndarray(format="rgb24")
            decoded.append((t, img))

    container.close()
    if not decoded:
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        return [black] * num_frames, src_fps

    times = np.array([t for t, _ in decoded], dtype=np.float64)
    duration_sec = float(times[-1]) if len(times) else 0.0
    clip_duration = num_frames / target_fps
    effective_duration = min(duration_sec + 1e-6, clip_duration)
    effective_duration = max(float(effective_duration), 1.0 / float(target_fps))

    out_times = (np.arange(num_frames, dtype=np.float64) + 0.5) / target_fps
    out_times = np.minimum(out_times, effective_duration - 1e-6)

    frames: list[np.ndarray] = []
    for t in out_times:
        j = int(np.searchsorted(times, t, side="right") - 1)
        j = max(0, min(j, len(decoded) - 1))
        rgb = decoded[j][1]
        frames.append(rgb)

    return frames, src_fps


def preprocess_clip_to_tensor(
    path: str | Path,
    cfg: VideoPreprocessConfig,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Load video, enforce num_frames at cfg.fps, resize to 224x224, ImageNet normalize.

    Returns:
        Float tensor of shape [T, 3, H, W] where T == cfg.num_frames.
    """
    path = Path(path)
    if cfg.backend == "opencv":
        raw_frames, _ = _load_indices_opencv(path, cfg.fps, cfg.num_frames)
        # OpenCV uses BGR
        rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in raw_frames]
    else:
        rgb_list, _ = _load_indices_pyav(path, cfg.fps, cfg.num_frames)

    tensors = []
    for rgb in rgb_list:
        rgb = cv2.resize(rgb, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        tensors.append(x)

    # Stack -> [T, 3, H, W]
    batch = torch.stack(tensors, dim=0)

    mean = IMAGENET_MEAN.to(batch.device, batch.dtype)
    std = IMAGENET_STD.to(batch.device, batch.dtype)
    batch = (batch - mean) / std

    if device is not None:
        batch = batch.to(device)
    return batch

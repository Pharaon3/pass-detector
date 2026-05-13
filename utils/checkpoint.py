"""Save / load training checkpoints with embedded config."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_EPOCH_CKPT_RE = re.compile(r"^epoch_(\d+)\.pt$")


def save_checkpoint(
    path: str | Path,
    *,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]],
    epoch: int,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model_state,
        "epoch": epoch,
        "config": config,
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def prune_epoch_checkpoints(ckpt_dir: str | Path, keep: int = 2) -> None:
    """
    Keep only the ``keep`` most recent ``epoch_NNN.pt`` files; delete older ones.

    Does not remove ``last.pt`` or any other filenames. Call after each epoch save
    so disk holds at most ``keep`` epoch snapshots plus ``last.pt`` (written at end).
    """
    if keep < 1:
        return
    d = Path(ckpt_dir)
    if not d.is_dir():
        return
    numbered: list[tuple[int, Path]] = []
    for p in d.iterdir():
        if not p.is_file():
            continue
        m = _EPOCH_CKPT_RE.match(p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    numbered.sort(key=lambda t: t[0], reverse=True)
    for _, path in numbered[keep:]:
        try:
            path.unlink()
        except OSError:
            pass

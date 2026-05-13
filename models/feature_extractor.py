"""
ResNet-152 backbone: ImageNet-pretrained, global average pooling, 2048-D per frame.

Forward:
    frames: [B, T, 3, H, W]  (typically H=W=224)
    -> reshape to [B*T, 3, H, W]
    -> ResNet (fc removed / Identity)
    -> feats: [B*T, 2048]
    -> reshape to [B, T, 2048]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet152_Weights

# soccer_event_model/ (parent of models/)
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def resolve_resnet_weights_path(weights_path: Optional[str | Path]) -> Optional[Path]:
    """Resolve optional weight file: absolute path, CWD, or relative to soccer_event_model/."""
    if weights_path is None:
        return None
    s = str(weights_path).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    p = Path(weights_path)
    if p.is_file():
        return p.resolve()
    cand = _PACKAGE_ROOT / p
    if cand.is_file():
        return cand.resolve()
    return None


def _strip_prefix(state: dict, prefix: str) -> dict:
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def _load_resnet152_state(model: nn.Module, weights_path: str | Path) -> None:
    """Load weights from a local .pth (full checkpoint or state_dict)."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"resnet152_weights_path not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}")

    # Common training wrappers
    if any(k.startswith("module.") for k in state):
        state = _strip_prefix(state, "module.")
    if any(k.startswith("model.") for k in state):
        state = _strip_prefix(state, "model.")
    if any(k.startswith("backbone.") for k in state):
        state = _strip_prefix(state, "backbone.")

    missing, _unexpected = model.load_state_dict(state, strict=False)
    if missing:
        # Second pass: keep only keys that exist on ResNet-152
        ref = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in ref and v.shape == ref[k].shape}
        if filtered:
            model.load_state_dict(filtered, strict=False)


class ResNet152FeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        freeze: bool = True,
        weights_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        resolved = resolve_resnet_weights_path(weights_path)
        if weights_path and resolved is None:
            if str(weights_path).strip().lower() not in ("null", "none", ""):
                raise FileNotFoundError(
                    f"resnet152_weights_path not found: {weights_path!r} "
                    f"(resolved from package root {_PACKAGE_ROOT})"
                )

        weights_enum = getattr(ResNet152_Weights, "IMAGENET1K_V2", ResNet152_Weights.IMAGENET1K_V1)

        if resolved is not None and resolved.suffix.lower() == ".h5":
            warnings.warn(
                f"resnet152_weights_path is Keras HDF5 ({resolved.name}); PyTorch ResNet-152 "
                f"expects a .pth/.pt state_dict. Using torchvision ImageNet weights ({weights_enum}) "
                f"for the backbone. Keep the .h5 in-repo for tooling that reads Keras weights.",
                UserWarning,
                stacklevel=2,
            )
            backbone = models.resnet152(weights=weights_enum)
        elif resolved is not None:
            backbone = models.resnet152(weights=None)
            _load_resnet152_state(backbone, resolved)
        else:
            backbone = models.resnet152(weights=weights_enum)
        # Remove final classifier: keep 2048-D GAP features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.set_frozen(freeze)

    def set_frozen(self, freeze: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        self._frozen = freeze

    @property
    def frozen(self) -> bool:
        return getattr(self, "_frozen", True)

    def forward(self, frames_btchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames_btchw: [B, T, 3, H, W]
        Returns:
            [B, T, 2048]
        """
        if frames_btchw.dim() != 5:
            raise ValueError(f"Expected 5D input [B,T,3,H,W], got {tuple(frames_btchw.shape)}")
        b, t, c, h, w = frames_btchw.shape
        x = frames_btchw.reshape(b * t, c, h, w)
        feats = self.backbone(x)  # [B*T, 2048]
        return feats.view(b, t, -1)

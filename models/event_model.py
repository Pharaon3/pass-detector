"""
End-to-end clip model: ResNet frame features -> temporal encoder -> logits.

Forward:
    video: [B, T, 3, H, W]
    -> logits: [B, T, num_classes]
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .feature_extractor import ResNet152FeatureExtractor
from .prediction_head import PredictionHead
from .temporal_model import TemporalModel


class EventDetectionModel(nn.Module):
    def __init__(
        self,
        feature_extractor: ResNet152FeatureExtractor,
        temporal: TemporalModel,
        head: PredictionHead,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.temporal = temporal
        self.head = head

    def forward(self, video_btchw):
        """
        Args:
            video_btchw: [B, T, 3, H, W] normalized RGB crops
        Returns:
            logits: [B, T, num_classes]
        """
        feats = self.feature_extractor(video_btchw)  # [B, T, 2048]
        z = self.temporal(feats)  # [B, T, hidden_dim]
        logits = self.head(z)  # [B, T, num_classes]
        return logits


def build_event_model(config: Dict[str, Any]) -> EventDetectionModel:
    """Construct model graph from a loaded YAML config dict."""
    num_classes = int(config["num_classes"])
    if len(config["class_names"]) != num_classes:
        raise ValueError("num_classes must match len(class_names) in config")

    freeze_fe = bool(config.get("freeze_feature_extractor", True))
    weights_path = config.get("resnet152_weights_path")

    feature_extractor = ResNet152FeatureExtractor(
        freeze=freeze_fe,
        weights_path=weights_path,
    )

    tcfg = config.get("temporal", {})
    temporal = TemporalModel(
        in_dim=2048,
        hidden_dim=int(tcfg.get("hidden_dim", 512)),
        model_type=str(tcfg.get("model_type", "transformer")),  # type: ignore[arg-type]
        n_heads=int(tcfg.get("n_heads", 8)),
        n_layers=int(tcfg.get("n_layers", 2)),
        dim_feedforward=int(tcfg.get("dim_feedforward", 2048)),
        dropout=float(tcfg.get("dropout", 0.1)),
    )

    hdim = int(tcfg.get("hidden_dim", 512))
    pcfg = config.get("prediction_head", {})
    head = PredictionHead(
        in_dim=hdim,
        num_classes=num_classes,
        mlp_hidden=int(pcfg.get("hidden_dim", 256)),
        dropout=float(pcfg.get("dropout", 0.1)),
    )

    return EventDetectionModel(feature_extractor, temporal, head)

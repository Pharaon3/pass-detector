"""
Run event detection on a single video clip and write JSON predictions.

Config: weights and architecture come from the checkpoint. If --config is
omitted, threshold / activation / multi_label / min_event_gap_sec / postprocess
are merged from ./config.yaml when present so tuning edits apply without retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from models.event_model import build_event_model
from postprocess import logits_to_probs, postprocess_clip, postprocess_config_from_cfg
from utils.checkpoint import load_checkpoint
from utils.video import VideoPreprocessConfig, preprocess_clip_to_tensor


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML to override checkpoint config (defaults to embedded checkpoint config)",
    )
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg: Dict[str, Any] = dict(ckpt.get("config") or {})
    if args.config:
        override = load_yaml(Path(args.config))
        cfg.update(override)
    else:
        # Postprocess tuning (threshold, etc.) lives at the top level of config.yaml.
        # Checkpoints embed a snapshot from training; merge repo config.yaml so edits
        # there affect infer without having to pass --config every time.
        disk_cfg = _PKG / "config.yaml"
        if disk_cfg.is_file():
            disk = load_yaml(disk_cfg)
            for key in ("threshold", "min_event_gap_sec", "activation", "multi_label", "postprocess"):
                if key in disk:
                    cfg[key] = disk[key]

    if not cfg:
        raise RuntimeError("No config in checkpoint; pass --config path/to/config.yaml")

    device_str = str(cfg.get("inference", {}).get("device", cfg.get("training", {}).get("device", "cuda")))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model = build_event_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    vpre = VideoPreprocessConfig(
        fps=int(cfg["fps"]),
        num_frames=int(cfg["num_frames"]),
        backend=str(cfg.get("video_backend", "opencv")),  # type: ignore[arg-type]
    )
    clip = preprocess_clip_to_tensor(args.video, vpre, device=device)  # [T,3,H,W]
    clip = clip.unsqueeze(0)  # [1,T,3,H,W]

    with torch.inference_mode():
        logits = model(clip)  # [1,T,C]

    activation = str(cfg.get("activation", "sigmoid"))
    pp_cfg = postprocess_config_from_cfg(cfg)
    ml = pp_cfg.multilabel
    if ml is not None:
        th_min = min(ml.thresholds)
        th_max = max(ml.thresholds)
        gaps_sec = [mf / ml.fps for mf in ml.min_gap_frames]
        print(
            f"Inference postprocess: activation={activation} multi_label=True "
            f"smoothing={ml.smoothing_enabled} window={ml.smoothing_window_frames} "
            f"peak_picking={ml.peak_picking_enabled} "
            f"thresholds_per_class min={th_min:.3f} max={th_max:.3f} ({len(ml.class_names)} classes) "
            f"min_gap_sec min={min(gaps_sec):.2f} max={max(gaps_sec):.2f} "
            f"top_k_per_class={ml.top_k_per_class} top_k_total={ml.top_k_total}",
            file=sys.stderr,
        )
    else:
        print(
            f"Inference postprocess: activation={activation} multi_label=False "
            f"threshold={pp_cfg.threshold} min_event_gap_sec={pp_cfg.min_event_gap_sec}",
            file=sys.stderr,
        )
    events = postprocess_clip(logits, pp_cfg)
    if not events:
        probs = logits_to_probs(logits, activation)
        arr = probs.detach().float().cpu().numpy()
        finite = np.isfinite(arr)
        n_nan = int(np.size(arr) - np.sum(finite))
        if np.any(finite):
            arr_f = arr[finite]
            th_ref = pp_cfg.threshold if ml is None else float(min(ml.thresholds))
            print(
                f"No events above min per-class threshold (min={th_ref:.4f}). "
                f"sigmoid(prob) min={float(arr_f.min()):.6f} max={float(arr_f.max()):.6f} "
                f"(finite values); NaN count={n_nan}",
                file=sys.stderr,
            )
        else:
            print(
                f"No events: all probabilities are non-finite (NaN/Inf). NaN count={n_nan}. "
                f"Check checkpoint / training stability.",
                file=sys.stderr,
            )
    # Stable JSON: round floats lightly
    for e in events:
        e["time"] = round(float(e["time"]), 4)
        e["confidence"] = round(float(e["confidence"]), 4)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    print(f"Wrote {len(events)} events to {out_path.resolve()}")


if __name__ == "__main__":
    main()

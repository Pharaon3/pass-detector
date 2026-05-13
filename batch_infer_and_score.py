#!/usr/bin/env python3
"""
Batch inference on the same clips used for training / validation, then score vs label JSON.

Uses the same discovery rules as ``train.py`` (``discover_clip_items``, ``training_environments``)
and the same train/val split as training when ``validation.enabled`` is true
(``validation.split_ratio``, ``validation.seed``).

Outputs:
  - ``<output_dir>/predictions/<stem>.json`` — one prediction list per clip (if --save-predictions)
  - ``<output_dir>/per_clip.jsonl`` — one JSON object per line with scores per stem
  - ``<output_dir>/summary.json`` — rolled-up totals and optional by-split breakdown

Shows a tqdm clip progress bar by default (``pip install tqdm``). Use ``--no-progress-bar`` to disable.

Example:
  python batch_infer_and_score.py \\
    --data_root dataset/private --video_dir . --labels_dir . \\
    --checkpoint checkpoints/best_model.pt \\
    --output-dir ./analyze_run \\
    --which all \\
    --gt-classes pass
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

try:
    from tqdm.auto import tqdm as tqdm_auto
except ImportError:  # pragma: no cover
    tqdm_auto = None  # type: ignore[misc, assignment]

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from dataset import discover_clip_items, parse_allowed_environments
from models.event_model import build_event_model
from postprocess import postprocess_clip, postprocess_config_from_cfg
from score_events import (
    _normalize_gt_event,
    load_ground_truth_events,
    match_prediction_to_gt,
)
from utils.checkpoint import load_checkpoint
from utils.dataset_video import load_stem_to_relpath
from utils.video import VideoPreprocessConfig, preprocess_clip_to_tensor

_DEFAULT_VALIDATION: Dict[str, Any] = {
    "enabled": True,
    "split_ratio": 0.2,
    "seed": 42,
    "threshold": 0.5,
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deterministic_train_val_stems(
    all_stems: List[str], split_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    n = len(all_stems)
    if n < 2:
        raise ValueError(
            "Need at least 2 clips for train/val split; "
            f"use --which all with validation disabled, or add clips (found {n})."
        )
    ratio = float(split_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"validation.split_ratio must be in (0, 1), got {ratio}")
    rng = random.Random(int(seed))
    stems_shuffled = sorted(all_stems)
    rng.shuffle(stems_shuffled)
    n_val = int(round(n * ratio))
    n_val = max(1, min(n - 1, n_val))
    val_stems = stems_shuffled[:n_val]
    train_stems = stems_shuffled[n_val:]
    return train_stems, val_stems


def _merge_cfg_from_checkpoint(
    ckpt: Dict[str, Any], config_path: Optional[str], merge_disk_infer_keys: bool
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(ckpt.get("config") or {})
    if config_path:
        override = load_yaml(Path(config_path))
        cfg.update(override)
    elif merge_disk_infer_keys:
        disk_cfg = _PKG / "config.yaml"
        if disk_cfg.is_file():
            disk = load_yaml(disk_cfg)
            for key in ("threshold", "min_event_gap_sec", "activation", "multi_label", "postprocess"):
                if key in disk:
                    cfg[key] = disk[key]
    if not cfg:
        raise RuntimeError("No config in checkpoint; pass --config path/to/config.yaml")
    return cfg


def _infer_one_clip(
    model: torch.nn.Module,
    device: torch.device,
    vpre: VideoPreprocessConfig,
    pp_cfg: Any,
    activation: str,
    video_path: Path,
) -> List[Dict[str, Any]]:
    clip = preprocess_clip_to_tensor(str(video_path), vpre, device=device).unsqueeze(0)
    with torch.inference_mode():
        logits = model(clip)
    events = postprocess_clip(logits, pp_cfg)
    for e in events:
        e["time"] = round(float(e["time"]), 4)
        e["confidence"] = round(float(e["confidence"]), 4)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer all train/val clips and score predictions vs label JSON (same split as training)."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default="videos")
    parser.add_argument("--labels_dir", type=str, default="labels")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional stem list file under data_root (same as train.py --split). Default: scan all clips.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML merged on top of checkpoint config (postprocess, etc.).",
    )
    parser.add_argument(
        "--which",
        type=str,
        choices=("all", "train", "val"),
        default="all",
        help="Which stems to run: all clips, or only the train or val split (same logic as train.py).",
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=None,
        help="Override config training_environments (e.g. --environments night).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analyze_infer",
        help="Directory for predictions/, per_clip.jsonl, summary.json",
    )
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=1.0,
        help="Seconds: match pred/GT when same class and |Δt| < this (passed to score_events).",
    )
    parser.add_argument(
        "--gt-classes",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only score these GT classes (e.g. pass). Recommended for pass-only models.",
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write predictions/<stem>.json per clip (default: true).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu (default: checkpoint inference/training device or cuda if available).",
    )
    parser.add_argument(
        "--no-merge-disk-config",
        action="store_true",
        help="Do not merge repo config.yaml infer keys when --config is omitted (infer.py merges by default).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm clip progress bar (default: show bar if tqdm is installed).",
    )
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = _merge_cfg_from_checkpoint(
        ckpt,
        args.config,
        merge_disk_infer_keys=not args.no_merge_disk_config and args.config is None,
    )
    cfg.setdefault("validation", {})
    val_cfg = {**_DEFAULT_VALIDATION, **(cfg.get("validation") or {})}

    if args.environments is not None:
        cfg["training_environments"] = list(args.environments)

    data_root = Path(args.data_root)
    video_dir_p = data_root / args.video_dir
    labels_dir_p = data_root / args.labels_dir
    stem_to_rel = load_stem_to_relpath(data_root)
    allowed_env = parse_allowed_environments(cfg)

    all_items = discover_clip_items(
        data_root,
        args.split,
        video_dir_p,
        labels_dir_p,
        stem_to_rel,
        allowed_environments=allowed_env,
    )
    stem_to_paths: Dict[str, Tuple[Path, Path]] = {}
    for vp, lp in all_items:
        stem_to_paths[vp.stem] = (vp, lp)
    all_stems = list(stem_to_paths.keys())

    validation_enabled = bool(val_cfg.get("enabled", True))
    train_stems: List[str] = []
    val_stems: List[str] = []
    if validation_enabled and len(all_stems) >= 2:
        train_stems, val_stems = _deterministic_train_val_stems(
            all_stems,
            split_ratio=float(val_cfg["split_ratio"]),
            seed=int(val_cfg["seed"]),
        )
    else:
        train_stems = list(all_stems)
        val_stems = []

    if args.which == "train":
        work = [(s, "train") for s in train_stems]
    elif args.which == "val":
        if not validation_enabled:
            print("ERROR: --which val requires validation.enabled true in config.", file=sys.stderr)
            sys.exit(2)
        work = [(s, "val") for s in val_stems]
    else:
        if validation_enabled and val_stems:
            work = [(s, "train") for s in train_stems] + [(s, "val") for s in val_stems]
        else:
            work = [(s, "train") for s in train_stems]

    gt_filter = None
    if args.gt_classes:
        gt_filter = {str(x).strip() for x in args.gt_classes if str(x).strip()}

    out_root = Path(args.output_dir)
    pred_dir = out_root / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "per_clip.jsonl"

    dev_str = args.device
    if dev_str is None:
        dev_str = str(
            cfg.get("inference", {}).get("device", cfg.get("training", {}).get("device", "cuda"))
        )
    device = torch.device(dev_str if torch.cuda.is_available() else "cpu")

    model = build_event_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    vpre = VideoPreprocessConfig(
        fps=int(cfg["fps"]),
        num_frames=int(cfg["num_frames"]),
        backend=str(cfg.get("video_backend", "opencv")),  # type: ignore[arg-type]
    )
    activation = str(cfg.get("activation", "sigmoid"))
    pp_cfg = postprocess_config_from_cfg(cfg)

    totals = {
        "n_clips": 0,
        "n_pred_events": 0,
        "n_gt_events_scored": 0,
        "matched": 0,
        "pred_only": 0,
        "gt_only": 0,
        "sum_conf_all": 0.0,
        "n_conf_all": 0,
        "sum_conf_matched": 0.0,
        "n_conf_matched": 0,
    }
    by_split: Dict[str, Dict[str, Any]] = {
        "train": {k: 0 for k in ("n_clips", "matched", "pred_only", "gt_only")},
        "val": {k: 0 for k in ("n_clips", "matched", "pred_only", "gt_only")},
    }

    print(
        f"Clips discovered: {len(all_stems)}  |  work list: {len(work)}  |  "
        f"validation.enabled={validation_enabled}  |  device={device}",
        file=sys.stderr,
    )

    use_pbar = not args.no_progress_bar and tqdm_auto is not None
    if not args.no_progress_bar and tqdm_auto is None:
        print(
            "tqdm is not installed; run `pip install tqdm` for a clip progress bar.",
            file=sys.stderr,
        )

    pbar_ctx: Any
    if use_pbar:
        pbar_ctx = tqdm_auto(
            work,
            desc="Batch infer+score",
            total=len(work),
            unit="clip",
            dynamic_ncols=True,
            leave=True,
        )
    else:
        pbar_ctx = nullcontext(work)

    with pbar_ctx as iterator:
        with jsonl_path.open("w", encoding="utf-8") as jf:
            for stem, split_tag in iterator:
                vp, lp = stem_to_paths[stem]
                try:
                    events = _infer_one_clip(model, device, vpre, pp_cfg, activation, vp)
                except Exception as exc:  # pragma: no cover
                    row = {
                        "stem": stem,
                        "split": split_tag,
                        "video": str(vp),
                        "label": str(lp),
                        "error": repr(exc),
                    }
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    msg = f"FAIL {stem}: {exc}"
                    if use_pbar:
                        tqdm_auto.write(msg, file=sys.stderr)
                    else:
                        print(msg, file=sys.stderr)
                    continue

                if args.save_predictions:
                    pj = pred_dir / f"{stem}.json"
                    pj.write_text(
                        json.dumps(events, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                gt_events = load_ground_truth_events(lp)
                res = match_prediction_to_gt(
                    events,
                    gt_events,
                    tolerance_sec=float(args.score_tolerance),
                    gt_class_filter=gt_filter,
                )

                n_gt_scored = 0
                for ev in gt_events:
                    parsed = _normalize_gt_event(ev)
                    if parsed is None:
                        continue
                    if gt_filter is not None and parsed[1] not in gt_filter:
                        continue
                    n_gt_scored += 1

                row = {
                    "stem": stem,
                    "split": split_tag,
                    "video": str(vp),
                    "label": str(lp),
                    "n_pred_events": len(events),
                    "n_gt_events_scored": n_gt_scored,
                    "matched": res.n_matched,
                    "pred_only": res.n_pred_only,
                    "gt_only": res.n_gt_only,
                    "mean_conf_all": res.mean_conf_all,
                    "mean_conf_matched": res.mean_conf_matched,
                }
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

                totals["n_clips"] += 1
                totals["n_pred_events"] += len(events)
                totals["n_gt_events_scored"] += n_gt_scored
                totals["matched"] += res.n_matched
                totals["pred_only"] += res.n_pred_only
                totals["gt_only"] += res.n_gt_only
                if res.mean_conf_all is not None and len(events) > 0:
                    totals["sum_conf_all"] += float(res.mean_conf_all) * len(events)
                    totals["n_conf_all"] += len(events)
                if res.mean_conf_matched is not None and res.n_matched > 0:
                    totals["sum_conf_matched"] += float(res.mean_conf_matched) * res.n_matched
                    totals["n_conf_matched"] += res.n_matched

                bs = by_split.setdefault(
                    split_tag, {"n_clips": 0, "matched": 0, "pred_only": 0, "gt_only": 0}
                )
                bs["n_clips"] += 1
                bs["matched"] += res.n_matched
                bs["pred_only"] += res.n_pred_only
                bs["gt_only"] += res.n_gt_only

                if use_pbar:
                    iterator.set_postfix(
                        tag=split_tag,
                        match=res.n_matched,
                        pred=len(events),
                        stem=stem[:20],
                    )

    mean_conf_all = totals["sum_conf_all"] / totals["n_conf_all"] if totals["n_conf_all"] else None
    mean_conf_matched = (
        totals["sum_conf_matched"] / totals["n_conf_matched"] if totals["n_conf_matched"] else None
    )

    summary = {
        "data_root": str(data_root.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "which": args.which,
        "score_tolerance_sec": float(args.score_tolerance),
        "gt_class_filter": sorted(gt_filter) if gt_filter else None,
        "training_environments": sorted(allowed_env) if allowed_env else None,
        "validation": {
            "enabled": validation_enabled,
            "split_ratio": float(val_cfg.get("split_ratio", 0.2)),
            "seed": int(val_cfg.get("seed", 42)),
        },
        "totals": {
            **{k: totals[k] for k in ("n_clips", "n_pred_events", "n_gt_events_scored", "matched", "pred_only", "gt_only")},
            "mean_confidence_all_predictions": mean_conf_all,
            "mean_confidence_matched": mean_conf_matched,
        },
        "by_split": by_split,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {jsonl_path.resolve()} ({totals['n_clips']} clips)")
    print(f"Wrote {summary_path.resolve()}")
    if mean_conf_all is not None:
        mc = f"{mean_conf_matched:.4f}" if mean_conf_matched is not None else "n/a"
        print(
            f"Totals: matched={totals['matched']}  pred_only={totals['pred_only']}  "
            f"gt_only={totals['gt_only']}  mean_conf(all preds)={mean_conf_all:.4f}  "
            f"mean_conf(matched)={mc}"
        )
    else:
        print(
            f"Totals: matched={totals['matched']}  pred_only={totals['pred_only']}  "
            f"gt_only={totals['gt_only']}  mean_conf=n/a (no confidence on predictions)"
        )


if __name__ == "__main__":
    main()

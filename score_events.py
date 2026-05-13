#!/usr/bin/env python3
"""
Compare inference JSON to a ground-truth label JSON.

Counts:
  - Matched: same event class and |time_pred - time_gt| < tolerance (default 1.0 s), one-to-one (greedy by smallest gap).
  - Only in prediction: no matching GT event.
  - Only in ground truth: no matching prediction.

Also prints mean confidence over all predicted events (and mean over matched only).

Prediction format: list of {time, event, confidence, ...} or {"events": [...]} with the same fields.
Ground truth: {"events": [...]} or list; time via time_sec / time / t; class via class / event / label / name.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from utils.labels import _event_class_name, _event_time_seconds  # noqa: SLF001


def load_prediction_events(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "events" in data:
        return list(data["events"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported prediction format in {path}")


def load_ground_truth_events(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "events" in data:
        return list(data["events"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported ground-truth format in {path}")


def _pred_time_seconds(ev: Dict[str, Any]) -> float:
    for key in ("time_sec", "time", "t", "timestamp_sec"):
        if key in ev:
            return float(ev[key])
    raise KeyError(f"Prediction event has no time field: {ev}")


def _pred_class_name(ev: Dict[str, Any]) -> str:
    for key in ("event", "class", "label", "name"):
        if key in ev:
            return str(ev[key])
    raise KeyError(f"Prediction event has no class/event field: {ev}")


def _pred_confidence(ev: Dict[str, Any]) -> Optional[float]:
    if "confidence" in ev:
        return float(ev["confidence"])
    if "prob" in ev:
        return float(ev["prob"])
    return None


def _normalize_gt_event(ev: Dict[str, Any]) -> Optional[Tuple[float, str]]:
    try:
        t = _event_time_seconds(ev)
        c = _event_class_name(ev)
    except KeyError:
        return None
    return t, c


def _normalize_pred_event(ev: Dict[str, Any]) -> Optional[Tuple[float, str, Optional[float]]]:
    try:
        t = _pred_time_seconds(ev)
        c = _pred_class_name(ev)
    except KeyError:
        return None
    conf = _pred_confidence(ev)
    return t, c, conf


@dataclass
class MatchResult:
    n_matched: int
    n_pred_only: int
    n_gt_only: int
    mean_conf_all: Optional[float]
    mean_conf_matched: Optional[float]
    matched_pairs: List[Dict[str, Any]]


def match_prediction_to_gt(
    pred_events: Sequence[Dict[str, Any]],
    gt_events: Sequence[Dict[str, Any]],
    tolerance_sec: float,
    gt_class_filter: Optional[set[str]] = None,
) -> MatchResult:
    """
    Greedy one-to-one matching: sort valid (dt, pred_i, gt_j) by dt ascending, assign if class equal and dt < tol.
    If ``gt_class_filter`` is set, only GT events whose class is in the set participate in matching.
    """
    preds: List[Tuple[int, float, str, Optional[float]]] = []
    for i, ev in enumerate(pred_events):
        parsed = _normalize_pred_event(ev)
        if parsed is None:
            continue
        t, c, conf = parsed
        preds.append((i, t, c, conf))

    gts: List[Tuple[int, float, str]] = []
    for j, ev in enumerate(gt_events):
        parsed = _normalize_gt_event(ev)
        if parsed is None:
            continue
        t, c = parsed
        if gt_class_filter is not None and c not in gt_class_filter:
            continue
        gts.append((j, t, c))

    candidates: List[Tuple[float, int, int]] = []
    for pi, tp, cp, _ in preds:
        for gj, tg, cg in gts:
            if cp != cg:
                continue
            d = abs(tp - tg)
            if d < tolerance_sec:
                candidates.append((d, pi, gj))

    candidates.sort(key=lambda x: x[0])
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matched_pairs: List[Dict[str, Any]] = []

    for d, pi, gj in candidates:
        if pi in used_pred or gj in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gj)
        _, tp, cp, conf = next(x for x in preds if x[0] == pi)
        _, tg, cg = next(x for x in gts if x[0] == gj)
        matched_pairs.append(
            {
                "pred_index": pi,
                "gt_index": gj,
                "class": cp,
                "time_pred": tp,
                "time_gt": tg,
                "time_diff_sec": d,
                "confidence": conf,
            }
        )

    n_pred_only = len(preds) - len(used_pred)
    n_gt_only = len(gts) - len(used_gt)

    confs_all = [p[3] for p in preds if p[3] is not None]
    mean_all = sum(confs_all) / len(confs_all) if confs_all else None

    confs_matched = [m["confidence"] for m in matched_pairs if m.get("confidence") is not None]
    mean_matched = sum(confs_matched) / len(confs_matched) if confs_matched else None

    return MatchResult(
        n_matched=len(matched_pairs),
        n_pred_only=n_pred_only,
        n_gt_only=n_gt_only,
        mean_conf_all=mean_all,
        mean_conf_matched=mean_matched,
        matched_pairs=matched_pairs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score inference JSON against ground-truth label JSON (time + class matching)."
    )
    parser.add_argument("--predicted", type=str, required=True, help="Path to inference output JSON")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to label / GT JSON")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Max |Δt| in seconds for a match (default: 1.0)",
    )
    parser.add_argument(
        "--gt-classes",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only consider these GT event classes (e.g. pass). Default: all GT classes.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write a JSON summary of counts and matched pairs",
    )
    args = parser.parse_args()

    pred_path = Path(args.predicted)
    gt_path = Path(args.ground_truth)
    if not pred_path.is_file():
        print(f"ERROR: predicted file not found: {pred_path}", file=sys.stderr)
        sys.exit(2)
    if not gt_path.is_file():
        print(f"ERROR: ground-truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(2)

    pred_events = load_prediction_events(pred_path)
    gt_events = load_ground_truth_events(gt_path)

    filt: Optional[set[str]] = None
    if args.gt_classes:
        filt = {str(x).strip() for x in args.gt_classes if str(x).strip()}

    result = match_prediction_to_gt(
        pred_events,
        gt_events,
        tolerance_sec=float(args.tolerance),
        gt_class_filter=filt,
    )

    print(f"Predicted: {pred_path}  ({len(pred_events)} raw events)")
    print(f"Ground truth: {gt_path}  ({len(gt_events)} raw events)")
    if filt:
        print(f"GT class filter: {sorted(filt)}")
    print(f"Time tolerance: {args.tolerance} s (same class, |Δt| < tolerance)")
    print()
    print(f"  Matched (TP pairs):     {result.n_matched}")
    print(f"  Only in prediction:     {result.n_pred_only}  (no same-class GT within {args.tolerance}s)")
    print(f"  Only in ground truth:   {result.n_gt_only}  (no same-class pred within {args.tolerance}s)")
    print()
    if result.mean_conf_all is not None:
        print(f"  Mean confidence (all predictions):   {result.mean_conf_all:.4f}")
    else:
        print("  Mean confidence (all predictions):   n/a (no confidence fields)")
    if result.mean_conf_matched is not None:
        print(f"  Mean confidence (matched only):      {result.mean_conf_matched:.4f}")
    elif result.n_matched > 0:
        print("  Mean confidence (matched only):      n/a (no confidence on matched preds)")
    else:
        print("  Mean confidence (matched only):      n/a (no matches)")

    if args.json_out:
        out = {
            "predicted_path": str(pred_path.resolve()),
            "ground_truth_path": str(gt_path.resolve()),
            "tolerance_sec": float(args.tolerance),
            "gt_class_filter": sorted(filt) if filt else None,
            "n_matched": result.n_matched,
            "n_pred_only": result.n_pred_only,
            "n_gt_only": result.n_gt_only,
            "mean_confidence_all": result.mean_conf_all,
            "mean_confidence_matched": result.mean_conf_matched,
            "matched_pairs": result.matched_pairs,
        }
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print()
        print(f"Wrote summary to {outp.resolve()}")


if __name__ == "__main__":
    main()

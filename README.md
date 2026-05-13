# Video-classification--action-spotting-model

PyTorch pipeline for per-frame multi-label soccer clip events (fixed length, e.g. 750 frames at 25 FPS), BCE-with-logits training, and JSON event inference.

## Label radius (per class)

`label_radius_frames` controls how many frames around each annotated event time are set to positive for that class.

- **Legacy int:** `label_radius_frames: 10` uses half-width 10 frames for every class.
- **Preferred dict:** requires a `default` key plus optional overrides per `class_names` entry, for example:

```yaml
label_radius_frames:
  default: 10
  pass: 8
  goal: 15
```

Implementation: `utils/labels.parse_label_radius_frames` and `events_to_frame_labels` (per-event radius via `radius_frames_for_class`).

## Strict label checking

`strict_labels: true` causes unknown strings in the JSON `class` / `event` / `label` / `name` fields to raise `ValueError` with the file path and valid class names. When `false`, a warning is logged and those events are skipped (they are never stamped silently). The dataset passes the label JSON path into parsing for useful errors.

## BCE `pos_weight` (class imbalance)

Under `loss:`:

- `use_pos_weight: true` with `pos_weight_mode: auto` scans **label JSON only** (same clips as the training split / dataset discovery) and builds per-class counts. For each class `c`, `pos_weight[c] = neg_count[c] / max(pos_count[c], 1)`, then clipped to `pos_weight_clip_max`. The tensor is moved to the training device and passed to `binary_cross_entropy_with_logits(..., pos_weight=...)`.
- `manual_pos_weight: [ ... ]` must have length `num_classes` and overrides auto.
- `multi_label: false` keeps softmax + cross-entropy unchanged (`pos_weight` is ignored with a stderr note).

Training prints a per-class table when `use_pos_weight` is enabled.

## Postprocessing (per-class threshold / gap, smoothing, peaks)

Legacy globals still work: top-level `threshold` and `min_event_gap_sec`.

Optional nested `postprocess:` block (see `config.yaml`):

- `thresholds` and `min_gap_sec` accept a scalar for all classes or a dict with `default` plus per-class keys matching `class_names`.
- `smoothing.enabled` + `window_frames`: temporal moving average per class on probabilities before detection (even window sizes use asymmetric edge padding; see `postprocess.temporal_moving_average_probs`).
- `peak_picking.enabled`: only local maxima along time (ties allowed) above the per-class threshold are candidates.
- NMS uses per-class minimum gaps in seconds (`min_gap_sec` → frames via `fps`).
- `top_k_per_class` / `top_k_total`: optional caps after NMS (int or per-class dict with `default` for `top_k_per_class`).

Inference JSON remains `{frame, time, event, confidence}`.

## Sanity script

```bash
python sanity_check_labels.py --data_root <folder> --config config.yaml --split train.txt
```

Reports raw JSON event counts, unknown strings, positive frame counts using the same radius logic as training, approximate `pos_weight`, and exits with code 1 when `strict_labels` is true and unknown labels exist.

## Tests

```bash
python -m unittest discover -s tests -v
```

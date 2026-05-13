"""
Train temporal stack + prediction head on fixed-length clips.

Feature extractor is frozen by default (config: freeze_feature_extractor).

Data: clips are loaded in batches (training.batch_size). Each optimizer step
processes one batch of up to batch_size videos stacked as [B,T,3,H,W].

Logging: training.log_each_step (default true) logs loss and video paths
after every batch. Set log_each_step: false and tune log_every for sparser logs.

Progress: training.progress_bar (default true) shows a tqdm bar per epoch
(requires `pip install tqdm`, listed in requirements.txt).

Resume: pass ``--resume path/to.pt`` (e.g. ``checkpoints/last.pt`` or
``checkpoints/best_model.pt``) to load ``model_state_dict`` and, if compatible,
``optimizer_state_dict``. Training runs epochs ``(checkpoint_epoch + 1) .. training.num_epochs``
using the current ``--config`` (learning rate, data paths, etc.); only weights and
optimizer buffers are restored from the file.

Validation / early stopping: see ``validation`` and ``early_stopping`` in ``config.yaml``.
When ``early_stopping.save_best`` is true, the best epoch (by any monitored metric when
``monitors`` is set) is written to ``early_stopping.best_checkpoint_path`` in addition to
per-epoch and ``last.pt`` checkpoints.

Train/val stem lists: use ``--export-split DIR`` to write ``train_stems.txt`` and ``val_stems.txt``
(see ``_deterministic_train_val_stems`` and ``validation`` keys in config).
"""

from __future__ import annotations

import argparse
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm as tqdm_auto
except ImportError:  # pragma: no cover
    tqdm_auto = None  # type: ignore[misc, assignment]

_TQDM_MISSING_NOTIFIED = False

# Allow `python path/to/train.py` from any working directory
_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from dataset import SoccerClipDataset, discover_clip_items, parse_allowed_environments
from models.event_model import build_event_model
from utils.checkpoint import load_checkpoint, prune_epoch_checkpoints, save_checkpoint
from utils.dataset_video import load_stem_to_relpath
from utils.early_stopping import EarlyStoppingMulti, EarlyStopper, build_early_stopper
from utils.label_stats import compute_auto_pos_weight_numpy


DEFAULT_VALIDATION: Dict[str, Any] = {
    "enabled": True,
    "split_ratio": 0.2,
    "seed": 42,
    "threshold": 0.5,
}

DEFAULT_EARLY_STOPPING: Dict[str, Any] = {
    "enabled": True,
    "monitor": "val_f1",
    "mode": "max",
    "patience": 8,
    "min_delta": 0.0005,
    "save_best": True,
    "best_checkpoint_path": "checkpoints/best_model.pt",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    videos = torch.stack([b["video"] for b in batch], dim=0)  # [B,T,3,H,W]
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    paths = [b["video_path"] for b in batch]
    return {"video": videos, "labels": labels, "video_path": paths}


def _build_pos_weight_tensor(
    cfg: Dict[str, Any],
    data_root: str,
    split: Optional[str],
    loss_cfg: Dict[str, Any],
    multi_label: bool,
    video_dir: str,
    labels_dir: str,
    stems: Optional[List[str]] = None,
) -> Optional[torch.Tensor]:
    if not multi_label:
        if bool(loss_cfg.get("use_pos_weight", False)):
            print(
                "loss.use_pos_weight is ignored when multi_label is false (cross-entropy).",
                file=sys.stderr,
            )
        return None
    if str(loss_cfg.get("type", "bce")) != "bce":
        return None
    if not bool(loss_cfg.get("use_pos_weight", False)):
        return None

    clip_max = float(loss_cfg.get("pos_weight_clip_max", 20.0))
    manual = loss_cfg.get("manual_pos_weight")
    num_classes = int(cfg["num_classes"])
    if manual is not None:
        lst = list(manual)
        if len(lst) != num_classes:
            raise ValueError(
                f"manual_pos_weight must have length num_classes={num_classes}, got {len(lst)}"
            )
        return torch.tensor(lst, dtype=torch.float32)

    if str(loss_cfg.get("pos_weight_mode", "auto")) != "auto":
        raise ValueError(f"Unknown pos_weight_mode: {loss_cfg.get('pos_weight_mode')!r}")

    arr = compute_auto_pos_weight_numpy(
        cfg,
        data_root,
        split,
        clip_max=clip_max,
        video_dir=video_dir,
        labels_dir=labels_dir,
        stems=stems,
    )
    return torch.from_numpy(arr)


def _print_pos_weight_table(class_names: List[str], w: torch.Tensor) -> None:
    print("Per-class BCE pos_weight (on device at train time):")
    for i, name in enumerate(class_names):
        print(f"  {name:20s}  {float(w[i].item()):.6f}")


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
    num_epochs: int,
    pos_weight: Optional[torch.Tensor],
) -> float:
    global _TQDM_MISSING_NOTIFIED
    model.train()
    multi_label = bool(cfg.get("multi_label", True))
    tcfg = cfg.get("training", {})
    log_every = max(1, int(tcfg.get("log_every", 10)))
    log_each_step = bool(tcfg.get("log_each_step", True))
    want_pbar = bool(tcfg.get("progress_bar", True))
    use_pbar = want_pbar and tqdm_auto is not None
    if want_pbar and tqdm_auto is None and not _TQDM_MISSING_NOTIFIED:
        print(
            "tqdm is not installed; run `pip install tqdm` for epoch progress bars.",
            file=sys.stderr,
        )
        _TQDM_MISSING_NOTIFIED = True

    num_batches = len(loader)
    bs = getattr(loader, "batch_size", None)

    pbar_ctx: Any
    if use_pbar:
        pbar_ctx = tqdm_auto(
            loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            total=num_batches,
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
    else:
        pbar_ctx = nullcontext(loader)
        print(f"epoch {epoch}/{num_epochs}: {num_batches} batches (batch_size={bs})")

    running = 0.0
    n = 0

    with pbar_ctx as iterator:
        for step, batch in enumerate(iterator):
            x = batch["video"].to(device)
            y = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)  # [B,T,C]

            if multi_label:
                if pos_weight is not None:
                    pw = pos_weight.to(device=device, dtype=logits.dtype)
                    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, y)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            loss.backward()
            optimizer.step()

            running += float(loss.item())
            n += 1
            if use_pbar:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

            should_log = log_each_step or (step % log_every == 0)
            if should_log:
                paths = batch.get("video_path", [])
                path_str = "; ".join(str(p) for p in paths) if paths else ""
                msg = (
                    f"epoch {epoch}/{num_epochs} step {step + 1}/{num_batches} "
                    f"loss {loss.item():.4f} | {path_str}"
                )
                if use_pbar:
                    tqdm_auto.write(msg)
                else:
                    print(msg)

    return running / max(n, 1)


def _deterministic_train_val_stems(
    all_stems: List[str], split_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    n = len(all_stems)
    if n < 2:
        raise ValueError(
            "validation.enabled requires at least 2 clips for a train/validation split; "
            f"found {n}."
        )
    ratio = float(split_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"validation.split_ratio must be in (0, 1), got {ratio}")
    rng = random.Random(int(seed))
    stems_shuffled = sorted(all_stems)
    rng.shuffle(stems_shuffled)
    n_val = int(round(n * ratio))
    n_val = max(1, min(n - 1, n_val))
    if n_val <= 0 or n_val >= n:
        raise ValueError(
            "validation split produced an empty validation or training set "
            f"(n={n}, n_val={n_val}). Adjust validation.split_ratio."
        )
    val_stems = stems_shuffled[:n_val]
    train_stems = stems_shuffled[n_val:]
    return train_stems, val_stems


def write_train_val_stem_lists(out_dir: Path, train_stems: List[str], val_stems: List[str]) -> Tuple[Path, Path]:
    """Write one stem per line (same convention as ``--split`` / ``train.txt``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_stems.txt"
    val_path = out_dir / "val_stems.txt"
    train_path.write_text("\n".join(train_stems) + ("\n" if train_stems else ""), encoding="utf-8")
    val_path.write_text("\n".join(val_stems) + ("\n" if val_stems else ""), encoding="utf-8")
    return train_path, val_path


def _print_validation_per_class_table(class_names: List[str], metrics: Dict[str, Any]) -> None:
    precs = metrics.get("val_per_class_precision")
    recs = metrics.get("val_per_class_recall")
    f1s = metrics.get("val_per_class_f1")
    if not isinstance(precs, list) or not isinstance(recs, list) or not isinstance(f1s, list):
        return
    if not precs:
        return
    print("  Per-class validation (P / R / F1):")
    print(f"  {'class':22s}  {'P':>5}  {'R':>5}  {'F1':>5}")
    for name, p, r, f1 in zip(class_names, precs, recs, f1s):
        print(f"  {name:22s}  {p:5.2f}  {r:5.2f}  {f1:5.2f}")


def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    pos_weight: Optional[torch.Tensor],
    threshold: float,
) -> Dict[str, Any]:
    model.eval()
    class_names: List[str] = list(cfg["class_names"])
    num_classes = len(class_names)
    multi_label = bool(cfg.get("multi_label", True))
    tcfg = cfg.get("training", {})
    want_pbar = bool(tcfg.get("progress_bar", True))
    use_pbar = want_pbar and tqdm_auto is not None

    sum_loss = 0.0
    n_loss_elems = 0
    tp_c = torch.zeros(num_classes)
    fp_c = torch.zeros(num_classes)
    fn_c = torch.zeros(num_classes)

    with torch.no_grad():
        iterator: Any = loader
        ctx = (
            tqdm_auto(iterator, desc="Validation", leave=False, dynamic_ncols=True)
            if use_pbar
            else nullcontext(iterator)
        )
        with ctx as it:
            for batch in it:
                x = batch["video"].to(device)
                y = batch["labels"].to(device)
                logits = model(x)

                if multi_label:
                    if pos_weight is not None:
                        pw = pos_weight.to(device=device, dtype=logits.dtype)
                        loss_red = F.binary_cross_entropy_with_logits(
                            logits, y, pos_weight=pw, reduction="sum"
                        )
                    else:
                        loss_red = F.binary_cross_entropy_with_logits(
                            logits, y, reduction="sum"
                        )
                    ne = logits.numel()
                    sum_loss += float(loss_red.item())
                    n_loss_elems += int(ne)

                    probs = torch.sigmoid(logits)
                    pred = (probs >= float(threshold)).to(dtype=logits.dtype)
                    y_bin = y.to(dtype=logits.dtype)
                else:
                    loss_red = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
                    )
                    n_el = int(y.numel())
                    sum_loss += float(loss_red.item())
                    n_loss_elems += n_el

                    pred_cls = logits.argmax(dim=-1)
                    y_bin = F.one_hot(y.long().view(-1), num_classes).view(
                        y.shape[0], y.shape[1], num_classes
                    ).to(device=device, dtype=logits.dtype)
                    pred = F.one_hot(pred_cls.long().view(-1), num_classes).view_as(y_bin).to(
                        dtype=logits.dtype
                    )

                tp_c += (pred * y_bin).sum(dim=(0, 1)).detach().cpu()
                fp_c += (pred * (1.0 - y_bin)).sum(dim=(0, 1)).detach().cpu()
                fn_c += ((1.0 - pred) * y_bin).sum(dim=(0, 1)).detach().cpu()

    eps = 1.0e-8
    val_loss = sum_loss / max(n_loss_elems, 1)
    tp = float(tp_c.sum())
    fp = float(fp_c.sum())
    fn = float(fn_c.sum())
    prec = tp / max(tp + fp, eps)
    rec = tp / max(tp + fn, eps)
    f1 = (2.0 * prec * rec) / max(prec + rec, eps)

    prec_pc: List[float] = []
    rec_pc: List[float] = []
    f1_pc: List[float] = []
    for c in range(num_classes):
        tpc = float(tp_c[c])
        fpc = float(fp_c[c])
        fnc = float(fn_c[c])
        pc = tpc / max(tpc + fpc, eps)
        rc = tpc / max(tpc + fnc, eps)
        f1c = (2.0 * pc * rc) / max(pc + rc, eps)
        prec_pc.append(pc)
        rec_pc.append(rc)
        f1_pc.append(f1c)

    out: Dict[str, Any] = {
        "val_loss": float(val_loss),
        "val_precision": float(prec),
        "val_recall": float(rec),
        "val_f1": float(f1),
        "val_per_class_precision": prec_pc,
        "val_per_class_recall": rec_pc,
        "val_per_class_f1": f1_pc,
    }
    for name, p, r, f1v in zip(class_names, prec_pc, rec_pc, f1_pc):
        out[f"val_precision_{name}"] = float(p)
        out[f"val_recall_{name}"] = float(r)
        out[f"val_f1_{name}"] = float(f1v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(_PKG / "config.yaml"))
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Dataset root (videos/ + labels/, or flat dir like dataset/examples, or "
            "per-clip folders like dataset/private with --video_dir . --labels_dir .)"
        ),
    )
    parser.add_argument("--split", type=str, default=None, help="Optional split list file under data_root")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="Subfolder of data_root with clip videos (use 'examples' for dataset/examples layout)",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="labels",
        help="Subfolder of data_root with per-clip label JSON (same as video_dir when JSON sits next to mp4)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint .pt to continue training (loads weights; restores AdamW state if shapes match)",
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Override config training_environments for this run (e.g. --environments night). "
            "Omit to use config.yaml training_environments (null = all)."
        ),
    )
    parser.add_argument(
        "--export-split",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "If validation is enabled, write train_stems.txt and val_stems.txt under this directory "
            "(one clip stem per line; split is deterministic from validation.seed and split_ratio)."
        ),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    cfg.setdefault("training", {})
    cfg.setdefault("loss", {})
    cfg["validation"] = {**DEFAULT_VALIDATION, **(cfg.get("validation") or {})}
    cfg["early_stopping"] = {**DEFAULT_EARLY_STOPPING, **(cfg.get("early_stopping") or {})}

    if args.environments is not None:
        cfg["training_environments"] = list(args.environments)

    device_str = str(cfg["training"].get("device", "cuda"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    val_cfg = cfg["validation"]
    es_cfg = cfg["early_stopping"]
    validation_enabled = bool(val_cfg.get("enabled", True))
    early_stopping_requested = bool(es_cfg.get("enabled", True))

    backend = str(cfg.get("video_backend", "opencv"))
    data_root = Path(args.data_root)
    video_dir_p = data_root / args.video_dir
    labels_dir_p = data_root / args.labels_dir
    stem_to_rel = load_stem_to_relpath(data_root)
    allowed_env = parse_allowed_environments(cfg)
    if allowed_env is not None:
        print(f"training_environments (clip filter): {sorted(allowed_env)}")
    all_items = discover_clip_items(
        data_root,
        args.split,
        video_dir_p,
        labels_dir_p,
        stem_to_rel,
        allowed_environments=allowed_env,
    )
    print(f"clips used for train/val split: {len(all_items)}")
    all_stems = [vp.stem for vp, _ in all_items]

    train_stems: Optional[List[str]] = None
    val_stems: Optional[List[str]] = None
    val_loader: Optional[DataLoader] = None
    val_threshold = float(val_cfg.get("threshold", 0.5))

    if validation_enabled:
        train_stems, val_stems = _deterministic_train_val_stems(
            all_stems,
            split_ratio=float(val_cfg["split_ratio"]),
            seed=int(val_cfg["seed"]),
        )
        if not val_stems:
            raise RuntimeError(
                "validation split produced an empty validation set. "
                "Increase the number of clips or adjust validation.split_ratio."
            )
        export_dir = args.export_split
        if export_dir:
            tp, vp = write_train_val_stem_lists(Path(export_dir), train_stems, val_stems)
            print(f"Wrote train/val stem lists: {tp.resolve()} ({len(train_stems)} stems)")
            print(f"                             {vp.resolve()} ({len(val_stems)} stems)")
        val_ds = SoccerClipDataset(
            args.data_root,
            cfg,
            split=None,
            stems=val_stems,
            video_dir=args.video_dir,
            labels_dir=args.labels_dir,
            video_backend=backend,
        )
        if len(val_ds) == 0:
            raise RuntimeError("Validation dataset has length 0 (empty validation set).")
        nw = int(cfg["training"].get("num_workers", 2))
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_batch,
            pin_memory=device.type == "cuda",
            persistent_workers=nw > 0,
            prefetch_factor=4 if nw > 0 else None,
        )

    pos_weight_split: Optional[str] = args.split
    pos_weight_stems: Optional[List[str]] = None
    if validation_enabled and train_stems is not None:
        pos_weight_split = None
        pos_weight_stems = train_stems

    ds = SoccerClipDataset(
        args.data_root,
        cfg,
        split=None if train_stems is not None else args.split,
        stems=train_stems,
        video_dir=args.video_dir,
        labels_dir=args.labels_dir,
        video_backend=backend,
    )
    nw = int(cfg["training"].get("num_workers", 2))
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    multi_label = bool(cfg.get("multi_label", True))
    loss_cfg = cfg.get("loss") or {}
    pos_weight_cpu = _build_pos_weight_tensor(
        cfg,
        args.data_root,
        pos_weight_split,
        loss_cfg,
        multi_label,
        video_dir=args.video_dir,
        labels_dir=args.labels_dir,
        stems=pos_weight_stems,
    )
    if pos_weight_cpu is not None:
        _print_pos_weight_table(list(cfg["class_names"]), pos_weight_cpu)

    early_stopper: Optional[EarlyStopper] = None
    if early_stopping_requested and val_loader is None:
        print(
            "early_stopping.enabled is true but validation is disabled or no validation loader is "
            "available; early stopping is disabled and training will run for training.num_epochs. "
            "Enable validation (validation.enabled: true) to monitor val_f1 or val_loss.",
            file=sys.stderr,
        )
    elif early_stopping_requested and val_loader is not None:
        early_stopper = build_early_stopper(es_cfg)

    model = build_event_model(cfg).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-5)),
    )

    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    num_epochs = int(cfg["training"]["num_epochs"])

    start_epoch = 1
    if args.resume:
        rpath = Path(args.resume)
        if not rpath.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {rpath}")
        ckpt = load_checkpoint(rpath, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        od = ckpt.get("optimizer_state_dict")
        if od is not None:
            try:
                optimizer.load_state_dict(od)
            except (ValueError, KeyError, RuntimeError) as exc:
                print(
                    f"Warning: could not load optimizer state from {rpath} ({exc!r}); "
                    "continuing with a fresh optimizer (new AdamW moments).",
                    file=sys.stderr,
                )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"Resumed from {rpath.resolve()} after epoch {start_epoch - 1}; "
            f"training epochs {start_epoch}..{num_epochs} (num_epochs is the final epoch index)."
        )

    if start_epoch > num_epochs:
        print(
            f"Nothing to train: resume starts at epoch {start_epoch} but training.num_epochs is {num_epochs}. "
            "Increase num_epochs in config or use an earlier checkpoint.",
            file=sys.stderr,
        )
        return

    last_completed_epoch = start_epoch - 1
    stop_early = False
    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            cfg,
            epoch,
            num_epochs,
            pos_weight_cpu,
        )

        val_metrics: Optional[Dict[str, Any]] = None
        if val_loader is not None:
            val_metrics = validate_one_epoch(
                model,
                val_loader,
                device,
                cfg,
                pos_weight_cpu,
                val_threshold,
            )
            print(
                f"Epoch {epoch} mean_train_loss {avg_loss:.4f} | "
                f"val_loss {val_metrics['val_loss']:.4f} | "
                f"val_precision {val_metrics['val_precision']:.2f} | "
                f"val_recall {val_metrics['val_recall']:.2f} | "
                f"val_f1 {val_metrics['val_f1']:.2f}"
            )
            _print_validation_per_class_table(list(cfg["class_names"]), val_metrics)
        else:
            print(f"Epoch {epoch} mean_train_loss {avg_loss:.4f}")

        if early_stopper is not None and val_metrics is not None:
            patience = int(es_cfg["patience"])
            save_best = bool(es_cfg.get("save_best", True))

            if isinstance(early_stopper, EarlyStoppingMulti):
                should_stop, improved, improved_names = early_stopper.step(val_metrics)
                if improved:
                    parts = [
                        f"{n}={float(val_metrics[n]):.4f}"
                        for n in improved_names
                    ]
                    trend = (
                        f"improved: {', '.join(parts)} | bests "
                        + ", ".join(
                            f"{n}={float(early_stopper.bests[n]):.4f}"
                            for n, _ in early_stopper.monitors
                            if early_stopper.bests.get(n) is not None
                        )
                    )
                    extra_save = ""
                    if save_best:
                        best_path = Path(es_cfg["best_checkpoint_path"])
                        best_path.parent.mkdir(parents=True, exist_ok=True)
                        save_checkpoint(
                            best_path,
                            model_state=model.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            epoch=epoch,
                            config=cfg,
                            extra={
                                "best_metrics": {
                                    k: float(v)
                                    for k, v in early_stopper.bests.items()
                                    if v is not None
                                },
                                "monitors": [
                                    {"metric": n, "mode": m} for n, m in early_stopper.monitors
                                ],
                            },
                        )
                        extra_save = f" Saving best checkpoint to {best_path}"
                    print(f"EarlyStopping: {trend}.{extra_save}")
                else:
                    nb = early_stopper.epochs_without_improvement
                    bests_str = ", ".join(
                        f"{n}={float(early_stopper.bests[n]):.4f}"
                        for n, _ in early_stopper.monitors
                        if early_stopper.bests.get(n) is not None
                    )
                    print(
                        f"EarlyStopping: no improvement in any monitored metric for {nb}/{patience} epochs. "
                        f"Bests: {bests_str}"
                    )
                if should_stop:
                    print(
                        f"EarlyStopping triggered at epoch {epoch}. "
                        f"Bests: {', '.join(f'{n}={float(early_stopper.bests[n]):.4f}' for n, _ in early_stopper.monitors if early_stopper.bests.get(n) is not None)}"
                    )
                    stop_early = True
            else:
                prev_best = early_stopper.best
                should_stop, improved = early_stopper.step(val_metrics)
                mon = str(es_cfg["monitor"])
                cur = float(val_metrics[mon])
                if improved:
                    if prev_best is None:
                        trend = f"{mon} reached {cur:.4f} (new best)"
                    else:
                        trend = f"{mon} improved from {prev_best:.4f} to {cur:.4f}"
                    extra_save = ""
                    if save_best:
                        best_path = Path(es_cfg["best_checkpoint_path"])
                        best_path.parent.mkdir(parents=True, exist_ok=True)
                        save_checkpoint(
                            best_path,
                            model_state=model.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            epoch=epoch,
                            config=cfg,
                            extra={
                                "best_metric": float(early_stopper.best),
                                "monitor": mon,
                            },
                        )
                        extra_save = f" Saving best checkpoint to {best_path}"
                    print(f"EarlyStopping: {trend}.{extra_save}")
                else:
                    nb = early_stopper.epochs_without_improvement
                    best_v = float(early_stopper.best) if early_stopper.best is not None else cur
                    print(
                        f"EarlyStopping: no improvement in {mon} for {nb}/{patience} epochs. "
                        f"Best {mon}: {best_v:.4f}"
                    )
                if should_stop:
                    print(
                        f"EarlyStopping triggered at epoch {epoch}. "
                        f"Best {mon}: {float(early_stopper.best):.4f}"
                    )
                    stop_early = True

        save_checkpoint(
            ckpt_dir / f"epoch_{epoch:03d}.pt",
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            config=cfg,
        )
        prune_epoch_checkpoints(ckpt_dir, keep=2)
        last_completed_epoch = epoch

        if stop_early:
            break

    save_checkpoint(
        ckpt_dir / "last.pt",
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        epoch=last_completed_epoch,
        config=cfg,
    )
    print(f"Done. Checkpoints in {ckpt_dir.resolve()}")


if __name__ == "__main__":
    main()

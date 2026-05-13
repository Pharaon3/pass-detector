"""Early stopping on a monitored validation metric."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union


class EarlyStopping:
    """
    Track best ``monitor`` value from ``metrics_dict`` each epoch.

    Returns ``should_stop`` when there have been ``patience`` consecutive epochs
    without meaningful improvement (see ``min_delta``).
    """

    def __init__(
        self,
        monitor: str,
        mode: str,
        patience: int,
        min_delta: float,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f'EarlyStopping mode must be "min" or "max", got {mode!r}')
        if patience < 1:
            raise ValueError(f"EarlyStopping patience must be >= 1, got {patience}")

        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self._epochs_without_improvement = 0

    def step(self, metrics_dict: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Args:
            metrics_dict: must contain ``self.monitor`` as a float-like value.

        Returns:
            should_stop: True if patience was exceeded this epoch.
            improved: True if this epoch set a new best (including the first epoch).
        """
        if self.monitor not in metrics_dict:
            raise KeyError(
                f"EarlyStopping monitor {self.monitor!r} missing from metrics; "
                f"keys present: {sorted(metrics_dict.keys())}"
            )
        current = float(metrics_dict[self.monitor])

        if self.best is None:
            self.best = current
            self._epochs_without_improvement = 0
            return False, True

        improved = False
        if self.mode == "max":
            if current > self.best + self.min_delta:
                improved = True
        else:
            if current < self.best - self.min_delta:
                improved = True

        if improved:
            self.best = current
            self._epochs_without_improvement = 0
            return False, True

        self._epochs_without_improvement += 1
        should_stop = self._epochs_without_improvement >= self.patience
        return should_stop, False

    @property
    def epochs_without_improvement(self) -> int:
        return self._epochs_without_improvement


class EarlyStoppingMulti:
    """
    Same patience counter, reset when **any** monitored metric improves.

    Use when training should continue if e.g. ``val_loss`` drops even when ``val_f1`` is flat.
    """

    def __init__(
        self,
        monitors: List[Tuple[str, str]],
        patience: int,
        min_delta: float,
    ) -> None:
        if not monitors:
            raise ValueError("EarlyStoppingMulti requires at least one (metric, mode) pair")
        if patience < 1:
            raise ValueError(f"EarlyStoppingMulti patience must be >= 1, got {patience}")
        for name, mode in monitors:
            if mode not in ("min", "max"):
                raise ValueError(f'EarlyStoppingMulti mode must be "min" or "max", got {mode!r} for {name!r}')
        self.monitors: List[Tuple[str, str]] = list(monitors)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.bests: Dict[str, Optional[float]] = {name: None for name, _ in self.monitors}
        self._epochs_without_improvement = 0

    @property
    def best(self) -> float | None:
        """Best value of the first monitor (for logging / checkpoint metadata)."""
        first, _ = self.monitors[0]
        return self.bests.get(first)

    @property
    def epochs_without_improvement(self) -> int:
        return self._epochs_without_improvement

    def step(self, metrics_dict: Dict[str, Any]) -> Tuple[bool, bool, List[str]]:
        """
        Returns:
            should_stop, improved_any, improved_metric_names (subset of monitors that improved).
        """
        improved_names: List[str] = []
        for name, mode in self.monitors:
            if name not in metrics_dict:
                raise KeyError(
                    f"EarlyStoppingMulti missing {name!r} in metrics; "
                    f"keys present: {sorted(metrics_dict.keys())}"
                )
            current = float(metrics_dict[name])
            best = self.bests[name]
            if best is None:
                self.bests[name] = current
                improved_names.append(name)
                continue
            if mode == "max":
                if current > best + self.min_delta:
                    self.bests[name] = current
                    improved_names.append(name)
            else:
                if current < best - self.min_delta:
                    self.bests[name] = current
                    improved_names.append(name)

        if improved_names:
            self._epochs_without_improvement = 0
            return False, True, improved_names

        self._epochs_without_improvement += 1
        should_stop = self._epochs_without_improvement >= self.patience
        return should_stop, False, []


EarlyStopper = Union[EarlyStopping, EarlyStoppingMulti]


def build_early_stopper(es_cfg: Dict[str, Any]) -> EarlyStopper:
    """
    Build from ``early_stopping`` config block.

    - If ``monitors`` is a non-empty list of ``{metric, mode}``, use :class:`EarlyStoppingMulti`.
    - Otherwise use legacy ``monitor`` + ``mode`` (:class:`EarlyStopping`).
    """
    patience = int(es_cfg["patience"])
    min_delta = float(es_cfg["min_delta"])
    raw_m = es_cfg.get("monitors")
    if raw_m is not None and isinstance(raw_m, (list, tuple)) and len(raw_m) > 0:
        rules: List[Tuple[str, str]] = []
        for i, item in enumerate(raw_m):
            if not isinstance(item, dict):
                raise TypeError(
                    f"early_stopping.monitors[{i}] must be a dict with 'metric' and 'mode', "
                    f"got {type(item).__name__}"
                )
            rules.append((str(item["metric"]), str(item["mode"])))
        return EarlyStoppingMulti(rules, patience, min_delta)
    return EarlyStopping(
        str(es_cfg["monitor"]),
        str(es_cfg["mode"]),
        patience,
        min_delta,
    )

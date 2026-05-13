"""Early stopping on a monitored validation metric."""

from __future__ import annotations

from typing import Any, Dict, Tuple


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

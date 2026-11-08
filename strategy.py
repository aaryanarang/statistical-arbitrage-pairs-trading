"""A single-pair finite-state strategy. Decisions are close-t timestamps."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    entry: float = 2.0
    exit: float = 0.0
    stop: float = 3.5
    window: int = 60
    cost_bps: float = 5.0
    initial_capital: float = 100_000.0
    require_cointegration: bool = True

    def __post_init__(self):
        if not (np.isfinite([self.entry, self.exit, self.stop, self.cost_bps, self.initial_capital]).all()
                and 0 <= self.exit < self.entry < self.stop):
            raise ValueError("Require 0 ≤ exit < entry < stop, with finite parameters.")
        if not isinstance(self.window, int) or self.window < 5:
            raise ValueError("Rolling window must be an integer of at least five observations.")
        if not 0 <= self.cost_bps <= 100 or self.initial_capital <= 0:
            raise ValueError("Use costs from 0 to 100 bps and positive starting capital.")


def generate_signals(zscore: pd.Series, config: StrategyConfig, allowed: bool = True) -> pd.DataFrame:
    """Desired state AFTER today's decision, to be executed next aligned close.

    +1 = long A / short beta B; -1 = short A / long beta B. Exits get
    priority; an exit never reverses directly into a new position that day.
    """
    state = 0
    rows = []
    for z in zscore:
        reason = "hold" if state else "flat"
        if not allowed:
            state, reason = 0, "suppressed"
        elif not np.isfinite(z):
            # Do not enter without a valid rolling window. Flatten an existing
            # trade next session if the signal becomes unusable.
            state, reason = 0, "invalid_signal" if state else "warmup"
        elif state == 1:
            if z <= -config.stop:
                state, reason = 0, "stop"
            elif z >= -config.exit:
                state, reason = 0, "mean_reversion"
        elif state == -1:
            if z >= config.stop:
                state, reason = 0, "stop"
            elif z <= config.exit:
                state, reason = 0, "mean_reversion"
        elif -config.stop < z < -config.entry:
            state, reason = 1, "enter_long"
        elif config.entry < z < config.stop:
            state, reason = -1, "enter_short"
        rows.append((state, reason))
    return pd.DataFrame(rows, columns=["target", "reason"], index=zscore.index)

"""Training-only estimation; trailing transformations never look forward."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class PairModel:
    alpha: float
    beta: float
    correlation: float
    pvalue: float
    test_statistic: float
    half_life: float | None
    train_end: pd.Timestamp
    observations: int
    warning: str = ""

    @property
    def qualified(self) -> bool:
        return bool(np.isfinite(self.pvalue) and self.pvalue < 0.05 and not self.warning)


def chronological_split(prices: pd.DataFrame, fraction: float = 0.6) -> int:
    if not 0 < fraction < 1:
        raise ValueError("Training fraction must be strictly between zero and one.")
    split = int(len(prices) * fraction)
    if split < 60 or len(prices) - split < 20:
        raise ValueError("Not enough rows for training (60 minimum) and testing (20 minimum).")
    return split


def estimate_half_life(spread: pd.Series) -> float | None:
    values = spread.dropna().to_numpy(dtype=float)
    if len(values) < 20 or np.std(values[:-1]) < 1e-10:
        return None
    fit = sm.OLS(values[1:], sm.add_constant(values[:-1], has_constant="add")).fit()
    phi = float(fit.params[1])
    # Oscillatory, unit-root and explosive estimates do not have this half-life.
    return float(-np.log(2) / np.log(phi)) if np.isfinite(phi) and 0 < phi < 1 else None


def fit_pair(training: pd.DataFrame) -> PairModel:
    if len(training) < 60 or training.shape[1] != 2:
        raise ValueError("Fit requires at least 60 observations and exactly two securities.")
    a, b = (training.iloc[:, i].to_numpy(dtype=float) for i in range(2))
    if not np.isfinite(np.column_stack([a, b])).all() or min(np.std(a), np.std(b)) < 1e-10:
        raise ValueError("Training prices must be finite and nonconstant.")
    fit = sm.OLS(a, sm.add_constant(b, has_constant="add")).fit()
    alpha, beta = map(float, fit.params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statistic, pvalue, _ = coint(a, b, trend="c", autolag="aic")
    diagnostic = "; ".join(str(w.message) for w in caught)
    if not np.isfinite(statistic) and not diagnostic:
        diagnostic = "Cointegration test was numerically degenerate; qualification suppressed."
    spread = training.iloc[:, 0] - alpha - beta * training.iloc[:, 1]
    return PairModel(alpha, beta, float(np.corrcoef(a, b)[0, 1]), float(pvalue), float(statistic),
                     estimate_half_life(spread), training.index[-1], len(training), diagnostic)


def spread_and_zscore(prices: pd.DataFrame, model: PairModel, window: int = 60) -> pd.DataFrame:
    if window < 5:
        raise ValueError("Rolling window must be at least five observations.")
    spread = prices.iloc[:, 0] - model.alpha - model.beta * prices.iloc[:, 1]
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std(ddof=1)
    z = (spread - mean) / std.where(std > 1e-10)
    return pd.DataFrame({"spread": spread, "rolling_mean": mean, "rolling_std": std, "zscore": z})

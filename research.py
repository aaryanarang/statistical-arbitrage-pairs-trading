"""One reusable end-to-end research pipeline for the UI, CLI, and tests."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backtest import BacktestResult, run_backtest
from statistics import PairModel, chronological_split, fit_pair, spread_and_zscore
from strategy import StrategyConfig, generate_signals


@dataclass
class ResearchResult:
    model: PairModel
    split: int
    features: pd.DataFrame
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    status: str


def run_research(prices: pd.DataFrame, config: StrategyConfig) -> ResearchResult:
    split = chronological_split(prices)
    if split < config.window:
        raise ValueError("The training sample is shorter than the rolling window. Increase the lookback.")
    model = fit_pair(prices.iloc[:split])
    features = spread_and_zscore(prices, model, config.window)
    allowed = model.beta > 0 and (model.qualified or not config.require_cointegration)
    if model.beta <= 0:
        status = "Unsupported nonpositive hedge ratio · trades suppressed"
    elif model.warning:
        status = "Cointegration test inconclusive · " + ("trades suppressed" if not allowed else "exploratory trades enabled")
    elif not model.qualified:
        status = "Cointegration failed (p ≥ 0.05) · " + ("trades suppressed" if not allowed else "exploratory trades enabled")
    else:
        status = "Training cointegration passed (p < 0.05)"
    results = []
    for section in (slice(None, split), slice(split, None)):
        sample = prices.iloc[section]
        signals = generate_signals(features.zscore.iloc[section], config, allowed)
        # A suppressed, unsupported beta has no holdings. A positive dummy is
        # used only to run the engine's all-cash path; it is never a hedge fit.
        results.append(run_backtest(sample, signals, model.beta if model.beta > 0 else 1.0, config))
    return ResearchResult(model, split, features, *results, status)

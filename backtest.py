"""Self-financing, two-leg adjusted-close accounting with one-session delay."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from strategy import StrategyConfig


TRADE_COLUMNS = ["side", "entry_signal_date", "entry_date", "exit_signal_date", "exit_date",
                 "exit_reason", "holding_sessions", "gross_pnl", "costs", "net_pnl", "return_on_entry_equity"]


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    open_trade: dict | None


def performance_metrics(daily: pd.DataFrame, trades: pd.DataFrame, capital: float) -> dict:
    returns = daily["return"].iloc[1:]
    n = len(returns)
    total = float(daily.equity.iloc[-1] / capital - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if n > 1 else None
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if n > 1 and returns.std(ddof=1) > 1e-12 else None
    return {
        "total_return": total,
        "annualized_return": float((1 + total) ** (252 / n) - 1) if n else None,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(daily.drawdown.min()),
        "completed_trades": len(trades),
        "win_rate": float((trades.net_pnl > 0).mean()) if len(trades) else None,
        "average_holding_sessions": float(trades.holding_sessions.mean()) if len(trades) else None,
        "dollar_turnover": float(daily.traded_notional.sum()),
        "turnover_multiple": float(daily.traded_notional.sum() / capital),
        "total_costs": float(daily.cost.sum()),
        "ending_equity": float(daily.equity.iloc[-1]),
        "open_positions": int(daily.position.iloc[-1] != 0),
    }


def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame, beta: float,
                 config: StrategyConfig) -> BacktestResult:
    """Each row marks old holdings, then executes yesterday's target at close.

    Active holdings are rebalanced daily to 100% of AFTER-cost equity gross.
    Fractional adjusted-price units are a total-return approximation. Cash
    earns zero; no borrow, financing, taxes, margin, or separate dividends.
    Final open positions remain marked to market, with no invented exit.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("This opposite-direction pairs engine requires a positive finite hedge ratio.")
    if len(prices) < 2 or prices.shape[1] != 2 or not prices.index.equals(signals.index):
        raise ValueError("Supply at least two aligned price/signal rows and exactly two price columns.")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Prices must have a sorted, unique chronological index.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Clean missing, invalid, or nonpositive prices before backtesting.")
    if not signals.target.isin([-1, 0, 1]).all():
        raise ValueError("Signal targets must be -1, 0 or +1.")
    rate = config.cost_bps / 10_000
    cash = previous_equity = config.initial_capital
    quantity = np.zeros(2)
    previous_prices = prices.iloc[0].to_numpy(dtype=float)
    position = 0
    active = None
    rows, trades = [], []

    for i, (day, row) in enumerate(prices.iterrows()):
        px = row.to_numpy(dtype=float)
        market_pnl = float(quantity @ (px - previous_prices)) if i else 0.0
        equity_before = float(cash + quantity @ px)
        if equity_before <= 0:
            raise ValueError("Strategy equity is depleted; this prototype does not simulate bankruptcy or margin calls.")
        if active is not None:
            active["gross_pnl"] += market_pnl
        target = int(signals.target.iloc[i - 1]) if i else 0
        reason = str(signals.reason.iloc[i - 1]) if i else "initial_cash"
        if position * target == -1:
            raise ValueError("Direct reversals are unsupported: insert a flat decision between opposing trades.")

        # q_A = s E/(A+beta B), q_B = -s beta E/(A+beta B).
        # Solve E_after + cost(q_after - q_before) = E_before, so fees
        # cannot inadvertently push gross exposure above the stated 100%.
        coeff = target * np.array([1.0, -beta]) / (px[0] + beta * px[1])
        low, high = 0.0, equity_before
        if rate * float(np.abs(quantity) @ px) >= equity_before:
            raise ValueError("Not enough equity to pay modeled liquidation costs.")
        for _ in range(55):
            mid = (low + high) / 2
            residual = mid + rate * float(np.abs(coeff * mid - quantity) @ px) - equity_before
            if residual > 0:
                high = mid
            else:
                low = mid
        desired = coeff * ((low + high) / 2)
        delta = desired - quantity
        notional = float(np.abs(delta) @ px)
        fee = rate * notional
        cash -= float(delta @ px) + fee
        quantity = desired
        equity = float(cash + quantity @ px)
        event = "rebalance" if target else "cash"

        if position and not target:
            event = "exit"
            active["costs"] += fee
            active.update(exit_signal_date=prices.index[i - 1], exit_date=day, exit_reason=reason,
                          holding_sessions=i - active["entry_i"])
            active["net_pnl"] = active["gross_pnl"] - active["costs"]
            active["return_on_entry_equity"] = active["net_pnl"] / active["entry_equity"]
            trades.append({key: active[key] for key in TRADE_COLUMNS})
            active = None
        elif not position and target:
            event = "entry"
            active = {"side": "Long spread" if target == 1 else "Short spread",
                      "entry_signal_date": prices.index[i - 1], "entry_date": day,
                      "entry_i": i, "entry_equity": equity_before, "gross_pnl": 0.0, "costs": fee}
        elif active is not None:
            active["costs"] += fee

        gross = float(np.abs(quantity) @ px)
        rows.append({"date": day, "equity": equity, "cash": cash, "quantity_a": quantity[0],
                     "quantity_b": quantity[1], "position": target, "gross_exposure": gross / equity,
                     "net_exposure": float(quantity @ px) / equity,
                     "market_pnl": market_pnl, "cost": fee, "net_pnl": market_pnl - fee,
                     "traded_notional": notional, "return": equity / previous_equity - 1,
                     "event": event, "execution_reason": reason,
                     "signal_date": prices.index[i - 1] if i else pd.NaT})
        position, previous_prices, previous_equity = target, px, equity

    daily = pd.DataFrame(rows).set_index("date")
    peak = daily.equity.cummax().clip(lower=config.initial_capital)
    daily["drawdown"] = daily.equity / peak - 1
    completed = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    if active is not None:
        active = {**active, "net_pnl": active["gross_pnl"] - active["costs"],
                  "holding_sessions": len(prices) - 1 - active["entry_i"]}
    return BacktestResult(daily, completed, performance_metrics(daily, completed, config.initial_capital), active)

"""Reproducible command-line research run, with JSON/CSV/HTML artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, timedelta
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_loader import DataError, completed_session_end, download_prices, snapshot_prices, synthetic_prices
from research import run_research
from strategy import StrategyConfig


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs=2, default=["SPY", "IVV"])
    parser.add_argument("--data", choices=["live", "snapshot", "synthetic"], default="live")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--end", type=date.fromisoformat, help="Inclusive end date (default: latest completed session)")
    parser.add_argument("--cost-bps", type=float, default=5)
    parser.add_argument("--allow-unqualified", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/run_latest"))
    args = parser.parse_args()
    end = min(args.end + timedelta(days=1), completed_session_end()) if args.end else completed_session_end()
    start = (pd.Timestamp(end) - pd.DateOffset(years=args.years)).date()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.data == "live":
            bundle = download_prices(*args.tickers, start, end)
        elif args.data == "snapshot":
            bundle = snapshot_prices(*args.tickers, start, end)
        else:
            bundle = synthetic_prices(*args.tickers, observations=args.years * 252)
        config = StrategyConfig(cost_bps=args.cost_bps, require_cointegration=not args.allow_unqualified)
        result = run_research(bundle.prices, config)
    except (DataError, ValueError) as exc:
        failure = {"status": "failed", "requested_data": args.data, "error": str(exc)}
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2
    bundle.prices.to_csv(args.output / "adjusted_prices.csv")
    result.features.to_csv(args.output / "spread_and_zscore.csv")
    report = {"source": bundle.source, "synthetic": bundle.synthetic, "note": bundle.note,
              "tickers": list(bundle.prices.columns), "requested_start": start if not bundle.synthetic else None,
              "requested_end_exclusive": end if not bundle.synthetic else None,
              "first_observation": bundle.prices.index[0], "last_observation": bundle.prices.index[-1],
              "aligned_observations": len(bundle.prices), "dropped_rows": bundle.dropped_rows,
              "training_observations": result.split, "test_start": bundle.prices.index[result.split],
              "configuration": asdict(config), "model": asdict(result.model), "status": result.status,
              "in_sample_fitted_diagnostic": result.in_sample.metrics, "out_of_sample": result.out_of_sample.metrics}
    for name, bt in [("in_sample", result.in_sample), ("out_of_sample", result.out_of_sample)]:
        bt.daily.to_csv(args.output / f"{name}_daily.csv")
        bt.trades.to_csv(args.output / f"{name}_trades.csv", index=False)
        report[f"{name}_open_trade"] = bt.open_trade
    (args.output / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    fig = go.Figure(go.Scatter(x=result.out_of_sample.daily.index, y=result.out_of_sample.daily.equity,
                               name="Net equity", line_color="#4ae2be"))
    fig.update_layout(title=("SYNTHETIC DEMO — " if bundle.synthetic else "Yahoo adjusted prices — ") +
                      f"{' / '.join(args.tickers)} · held-out net equity", template="plotly_dark", yaxis_title="USD")
    fig.write_html(args.output / "equity.html", include_plotlyjs=True)
    print(json.dumps(json_safe(report), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

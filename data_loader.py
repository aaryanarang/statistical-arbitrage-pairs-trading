"""Adjusted daily prices, explicit cleaning, and an opt-in artificial demo."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import re

import numpy as np
import pandas as pd


class DataError(ValueError):
    """A readable error that the dashboard can display without a traceback."""


@dataclass
class DataBundle:
    prices: pd.DataFrame
    source: str
    dropped_rows: int = 0
    note: str = ""
    synthetic: bool = False


def validate_tickers(a: str, b: str) -> tuple[str, str]:
    pair = tuple(t.strip().upper() for t in (a, b))
    if any(not re.fullmatch(r"[A-Z0-9.^=\-]{1,15}", t) for t in pair):
        raise DataError("Use two valid Yahoo Finance ticker symbols.")
    if pair[0] == pair[1]:
        raise DataError("Choose two different securities.")
    return pair


def completed_session_end(now: datetime | None = None) -> date:
    """Exclusive date cutoff: omit today's potentially incomplete daily candle.

    A 16:15 New York buffer is conservative on ordinary US sessions. Weekends
    and holidays have no Yahoo observations. This is not an exchange calendar.
    """
    ny = ZoneInfo("America/New_York")
    now = datetime.now(ny) if now is None else now.astimezone(ny)
    return now.date() + timedelta(days=int(now.time() >= time(16, 15)))


def clean_prices(frame: pd.DataFrame, tickers: tuple[str, str]) -> tuple[pd.DataFrame, int]:
    if frame is None or frame.empty:
        raise DataError("No prices returned. Check the symbols, date range, and internet connection.")
    if not set(tickers).issubset(frame.columns):
        raise DataError("The data source did not return both tickers.")
    df = frame.loc[:, list(tickers)].copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    n = len(df)
    df = df.loc[~df.index.isna()]
    # Keep the first duplicate rather than introducing a later revision silently.
    df = df.loc[~df.index.duplicated(keep="first")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.where(df > 0).dropna(how="any")
    df.index.name = "Date"
    if len(df) < 100:
        raise DataError("Fewer than 100 aligned valid observations. Increase the lookback or check the pair.")
    return df.astype(float), n - len(df)


def download_prices(a: str, b: str, start: date, end: date) -> DataBundle:
    """Download adjusted closes. end is exclusive; missing prices are never filled."""
    import yfinance as yf

    pair = validate_tickers(a, b)
    if start >= end:
        raise DataError("The start date must precede the end date.")
    # Keep yfinance's cache within this project instead of a system directory.
    yf.set_tz_cache_location(str(Path(__file__).parent / ".cache" / "yfinance"))
    try:
        raw = yf.download(
            list(pair), start=start.isoformat(), end=end.isoformat(), interval="1d",
            auto_adjust=True, actions=False, progress=False, threads=False,
            group_by="column", timeout=20,
        )
        if raw is None or raw.empty:
            raise DataError("Yahoo Finance returned no data (it may be rate-limited or unreachable).")
        if isinstance(raw.columns, pd.MultiIndex):
            frame = raw.xs("Close", axis=1, level=0)
        else:
            raise DataError("Unexpected Yahoo response: adjusted closes for both tickers were not found.")
        frame = frame.loc[(frame.index.date >= start) & (frame.index.date < end)]
        prices, dropped = clean_prices(frame, pair)
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"Could not download prices: {type(exc).__name__}: {exc}") from exc
    lag = (end - prices.index[-1].date()).days
    note = "Rows missing either leg are dropped; no prices are forward-filled."
    if lag > 5:
        note += f" Data are stale relative to the requested end (last row {prices.index[-1].date()})."
    return DataBundle(prices, "Yahoo Finance via yfinance · adjusted daily close", dropped, note)


def synthetic_prices(a: str = "SPY", b: str = "IVV", observations: int = 1260, seed: int = 42) -> DataBundle:
    """Deterministic artificial random walk plus AR(1) residual; never market data."""
    pair = validate_tickers(a, b)
    rng = np.random.default_rng(seed)
    base = 180 + np.cumsum(rng.normal(0.025, 0.7, observations))
    residual = np.zeros(observations)
    for t in range(1, observations):
        residual[t] = 0.86 * residual[t - 1] + rng.normal(0, 0.6)
    frame = pd.DataFrame(
        {pair[0]: 12 + 1.15 * base + residual, pair[1]: base},
        index=pd.bdate_range("2020-01-02", periods=observations),
    )
    prices, dropped = clean_prices(frame, pair)
    return DataBundle(prices, "SYNTHETIC DEMO · artificial prices, not historical market data", dropped,
                      "Ticker names are labels only. Dates are artificial weekdays, not exchange sessions.", True)


def snapshot_prices(a: str, b: str, start: date, end: date) -> DataBundle:
    """Load an explicitly selected, checksum-verified historical download."""
    pair = validate_tickers(a, b)
    directory = Path(__file__).parent / "data"
    key = "_".join(pair).lower()
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        record = manifest["snapshots"][key]
        path = directory / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise DataError("Snapshot checksum mismatch. Restore the original data file or use Yahoo Finance.")
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise DataError("No bundled snapshot for that ticker order. Use a preset pair or select Yahoo Finance.") from exc
    frame = frame.loc[(frame.index.date >= start) & (frame.index.date < end)]
    prices, dropped = clean_prices(frame, pair)
    note = (f"Historical snapshot, not refreshed live. Downloaded {manifest['download_date_new_york']}; "
            f"stored range {record['first_date']} to {record['last_date']}. "
            "Requested dates are restricted to the available snapshot; no missing prices are filled.")
    return DataBundle(prices, "Yahoo Finance via yfinance · bundled historical snapshot", dropped, note)

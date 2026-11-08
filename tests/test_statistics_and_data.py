from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from data_loader import DataError, clean_prices, completed_session_end, download_prices, snapshot_prices, synthetic_prices, validate_tickers
from research import run_research
from statistics import estimate_half_life, fit_pair, spread_and_zscore
from strategy import StrategyConfig


def test_cleaning_drops_invalid_and_missing_rows_without_filling():
    p = synthetic_prices(observations=200).prices
    p.iloc[5, 0] = np.nan
    p.iloc[8, 1] = -1
    p.iloc[9, 0] = np.inf
    messy = pd.concat([p.iloc[::-1], p.iloc[[10]]])
    cleaned, removed = clean_prices(messy, ("SPY", "IVV"))
    assert removed == 4
    assert len(cleaned) == 197
    assert p.index[5] not in cleaned.index
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.has_duplicates


def test_training_estimates_and_oos_prefix_unchanged_by_future_data():
    p = synthetic_prices(observations=400).prices
    first = run_research(p, StrategyConfig())
    altered = p.copy()
    altered.iloc[-40:, 0] *= 2
    second = run_research(altered, StrategyConfig())
    assert first.model == second.model
    pd.testing.assert_frame_equal(first.features.iloc[:-40], second.features.iloc[:-40])
    pd.testing.assert_frame_equal(first.out_of_sample.daily.iloc[:-40], second.out_of_sample.daily.iloc[:-40])


def test_rolling_warmup_trailing_math_and_zero_variance():
    p = synthetic_prices(observations=200).prices
    model = fit_pair(p.iloc[:120])
    f = spread_and_zscore(p, model, 20)
    assert f.zscore.iloc[:19].isna().all()
    window = f.spread.iloc[:20]
    assert f.zscore.iloc[19] == pytest.approx((window.iloc[-1] - window.mean()) / window.std(ddof=1))
    constant = p.copy()
    constant.iloc[:, 0] = model.alpha + model.beta * constant.iloc[:, 1]
    assert spread_and_zscore(constant, model, 20).zscore.isna().all()


def test_half_life_valid_and_undefined_cases():
    assert estimate_half_life(pd.Series([1.0] * 100)) is None
    assert estimate_half_life(pd.Series(1.05 ** np.arange(100))) is None
    rng = np.random.default_rng(3)
    ar = np.zeros(3000)
    for i in range(1, len(ar)):
        ar[i] = .8 * ar[i - 1] + rng.normal()
    assert estimate_half_life(pd.Series(ar)) == pytest.approx(-np.log(2) / np.log(.8), rel=.15)


def test_completed_cutoff_omits_intraday_candle():
    ny = ZoneInfo("America/New_York")
    assert str(completed_session_end(datetime(2026, 11, 6, 15, 0, tzinfo=ny))) == "2026-11-06"
    assert str(completed_session_end(datetime(2026, 11, 6, 17, 0, tzinfo=ny))) == "2026-11-07"


def test_download_extracts_adjusted_close_and_passes_explicit_flags(monkeypatch, tmp_path):
    import yfinance as yf
    p = synthetic_prices(observations=200).prices
    raw = pd.concat({"Close": p, "Open": p}, axis=1)
    calls = {}
    def fake(*args, **kwargs):
        calls.update(kwargs)
        return raw
    monkeypatch.setattr(yf, "download", fake)
    monkeypatch.setattr(yf, "set_tz_cache_location", lambda path: None)
    bundle = download_prices("SPY", "IVV", p.index[0].date(), (p.index[-1] + pd.Timedelta(days=1)).date())
    assert calls["auto_adjust"] is True
    assert calls["threads"] is False
    assert bundle.prices.to_numpy() == pytest.approx(p.to_numpy())
    assert bundle.synthetic is False


def test_download_failure_never_silently_uses_synthetic(monkeypatch):
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(yf, "set_tz_cache_location", lambda path: None)
    with pytest.raises(DataError):
        download_prices("SPY", "IVV", pd.Timestamp("2020-01-01").date(), pd.Timestamp("2024-01-01").date())


@pytest.mark.parametrize("pair", [("SPY", "SPY"), ("bad ticker", "IVV"), ("", "IVV")])
def test_bad_tickers(pair):
    with pytest.raises(DataError):
        validate_tickers(*pair)


def test_negative_hedge_is_labeled_and_suppressed():
    p = synthetic_prices(observations=300).prices
    p.iloc[:, 0] = 1000 - p.iloc[:, 0]
    r = run_research(p, StrategyConfig(require_cointegration=False))
    assert r.model.beta < 0
    assert "Unsupported" in r.status
    assert r.out_of_sample.daily.position.eq(0).all()


@pytest.mark.parametrize("pair", [("SPY", "IVV"), ("GLD", "IAU"), ("KO", "PEP")])
def test_bundled_real_snapshots_run_full_pipeline(pair):
    b = snapshot_prices(*pair, pd.Timestamp("2021-11-07").date(), pd.Timestamp("2026-11-07").date())
    assert b.synthetic is False
    assert len(b.prices) == 1255
    r = run_research(b.prices, StrategyConfig())
    assert r.model.train_end < r.out_of_sample.daily.index[0]
    assert np.isfinite(r.model.pvalue)
    if not r.model.qualified:
        assert r.out_of_sample.daily.position.eq(0).all()


def test_real_snapshot_cost_sensitivity_with_gate_explicitly_disabled():
    b = snapshot_prices("SPY", "IVV", pd.Timestamp("2021-11-07").date(), pd.Timestamp("2026-11-07").date())
    zero = run_research(b.prices, StrategyConfig(cost_bps=0, require_cointegration=False))
    paid = run_research(b.prices, StrategyConfig(cost_bps=5, require_cointegration=False))
    assert paid.out_of_sample.metrics["completed_trades"] > 0
    assert paid.out_of_sample.metrics["ending_equity"] < zero.out_of_sample.metrics["ending_equity"]
    assert "exploratory" in paid.status

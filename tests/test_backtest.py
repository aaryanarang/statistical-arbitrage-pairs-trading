from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from backtest import run_backtest
from data_loader import synthetic_prices
from research import run_research
from strategy import StrategyConfig, generate_signals


def example(prices=None, z=None, cost=0.0, beta=1.0):
    if prices is None:
        prices = [[100, 100], [100, 100], [110, 100], [112, 100], [111, 100]]
    index = pd.bdate_range("2024-01-02", periods=len(prices))
    p = pd.DataFrame(prices, index=index, columns=["A", "B"], dtype=float)
    config = StrategyConfig(cost_bps=cost)
    s = generate_signals(pd.Series(z if z is not None else [-2.5, -2.4, 0.1, 0.2, 0.2], index=index), config)
    return run_backtest(p, s, beta, config)


def test_hand_calculated_pnl_and_next_session_execution():
    r = example()
    assert r.daily.position.tolist() == [0, 1, 1, 0, 0]
    assert r.daily.equity.tolist() == pytest.approx([100000, 100000, 105000, 106000, 106000])
    assert len(r.trades) == 1
    trade = r.trades.iloc[0]
    assert trade.net_pnl == pytest.approx(6000)
    assert trade.holding_sessions == 2
    assert trade.entry_signal_date < trade.entry_date
    assert trade.exit_signal_date < trade.exit_date


def test_no_pnl_from_price_move_before_entry_execution():
    r = example([[100, 100], [150, 100], [150, 100], [150, 100], [150, 100]])
    assert r.metrics["total_return"] == pytest.approx(0)
    assert r.daily.market_pnl.eq(0).all()


def test_costs_strictly_reduce_controlled_roundtrip_results():
    flat = [[100, 100]] * 5
    zero, low, high = (example(flat, cost=c) for c in [0, 5, 25])
    assert zero.metrics["ending_equity"] > low.metrics["ending_equity"] > high.metrics["ending_equity"]
    assert high.metrics["total_costs"] > low.metrics["total_costs"] > 0
    assert low.daily.equity.iloc[1] == pytest.approx(100000 / 1.0005)
    assert high.trades.iloc[0].net_pnl == pytest.approx(-high.metrics["total_costs"])


def test_beta_share_ratio_gross_exposure_and_cash_identity():
    r = example(cost=5, beta=1.5)
    d = r.daily
    active = d.position.ne(0)
    assert d.loc[active, "gross_exposure"].to_numpy() == pytest.approx(np.ones(active.sum()))
    assert d.loc[active, "quantity_b"].to_numpy() == pytest.approx(-1.5 * d.loc[active, "quantity_a"].to_numpy())
    assert (d.market_pnl - d.cost).sum() == pytest.approx(d.equity.iloc[-1] - 100000)
    px = np.array([[100, 100], [100, 100], [110, 100], [112, 100], [111, 100]])
    identity = d.cash + d.quantity_a * px[:, 0] + d.quantity_b * px[:, 1]
    assert identity.to_numpy() == pytest.approx(d.equity.to_numpy())
    assert d.loc[~active, "cash"].to_numpy() == pytest.approx(d.loc[~active, "equity"].to_numpy())


def test_rebalancing_turnover_is_charged_on_both_legs():
    r = example([[100, 100], [100, 100], [110, 110], [110, 110], [110, 110]], cost=5)
    assert r.daily.event.iloc[2] == "rebalance"
    assert r.daily.cost.iloc[2] > 0
    assert r.daily.cost.to_numpy() == pytest.approx(r.daily.traded_notional.to_numpy() * .0005)


def test_no_trade_metrics_and_last_signal_not_executed():
    r = example(z=[0, 0, 0, 0, -2.5])
    assert r.metrics["total_return"] == pytest.approx(0)
    assert r.metrics["completed_trades"] == 0
    assert r.metrics["win_rate"] is None
    assert r.metrics["average_holding_sessions"] is None
    assert r.metrics["sharpe"] is None
    assert r.metrics["total_costs"] == 0


def test_open_position_is_marked_and_not_counted_as_completed():
    r = example(z=[-2.5] * 5, cost=5)
    assert r.open_trade is not None
    assert r.trades.empty
    assert r.open_trade["net_pnl"] == pytest.approx(r.metrics["ending_equity"] - 100000)


def test_closed_and_open_trade_pnl_reconciles_to_equity():
    p = synthetic_prices(observations=500).prices
    r = run_research(p, StrategyConfig()).out_of_sample
    open_pnl = r.open_trade["net_pnl"] if r.open_trade else 0
    assert r.trades.net_pnl.sum() + open_pnl == pytest.approx(r.daily.equity.iloc[-1] - 100000)


def test_future_prices_cannot_change_past_holdings_or_pnl():
    first = example()
    changed = example([[100, 100], [100, 100], [110, 100], [112, 100], [1000, 1000]])
    pd.testing.assert_frame_equal(first.daily.iloc[:4], changed.daily.iloc[:4])


def test_stop_exits_only_at_following_close():
    r = example(z=[-2.1, -3.6, -4, 0, 0])
    assert r.daily.position.tolist() == [0, 1, 0, 0, 0]
    assert r.trades.iloc[0].exit_reason == "stop"


@pytest.mark.parametrize("beta", [0, -1, np.nan])
def test_unsupported_beta_rejected(beta):
    with pytest.raises(ValueError, match="positive finite"):
        example(beta=beta)


def test_engine_rejects_missing_prices_instead_of_inventing_prices():
    with pytest.raises(ValueError, match="Clean missing"):
        example([[100, 100], [100, 100], [np.nan, 100], [110, 100], [111, 100]])


def test_gate_creates_all_cash_when_training_pair_fails(monkeypatch):
    import research
    p = synthetic_prices(observations=300).prices
    original = research.fit_pair
    monkeypatch.setattr(research, "fit_pair", lambda train: replace(original(train), pvalue=0.8))
    r = research.run_research(p, StrategyConfig())
    assert r.out_of_sample.daily.position.eq(0).all()
    assert r.out_of_sample.metrics["total_costs"] == 0
    assert "suppressed" in r.status

import numpy as np
import pandas as pd
import pytest

from strategy import StrategyConfig, generate_signals


def signals(values, **kwargs):
    return generate_signals(pd.Series(values), StrategyConfig(**kwargs))


def test_default_long_short_exits_and_no_same_day_reversal():
    s = signals([np.nan, -2.1, -2.5, 2.6, 2.2, -0.1])
    assert s.target.tolist() == [0, 1, 1, 0, -1, 0]
    assert s.reason.tolist() == ["warmup", "enter_long", "hold", "mean_reversion", "enter_short", "mean_reversion"]


@pytest.mark.parametrize("values", [[-2.1, -3.5], [2.1, 3.5]])
def test_adverse_stop(values):
    s = signals(values)
    assert s.target.iloc[1] == 0
    assert s.reason.iloc[1] == "stop"


def test_does_not_enter_at_threshold_or_beyond_stop():
    assert signals([-2.0, 2.0, -3.5, 3.5, -4.0, 4.0]).target.eq(0).all()


def test_configurable_exit_band():
    s = signals([-2.1, -0.6, -0.4, 2.1, 0.4], exit=0.5)
    assert s.target.tolist() == [1, 1, 0, -1, 0]


def test_invalid_signal_requests_flat_and_gate_suppresses_all():
    assert signals([-2.1, np.nan]).target.tolist() == [1, 0]
    gated = generate_signals(pd.Series([-2.5, 0, 2.5]), StrategyConfig(), allowed=False)
    assert gated.target.eq(0).all()


def test_decisions_do_not_depend_on_future_values():
    prefix = pd.Series([np.nan, -2.1, -2.8, -1.0, 0.2, 2.4])
    combined = pd.concat([prefix, pd.Series([100, -100, 0])], ignore_index=True)
    pd.testing.assert_frame_equal(signals(prefix), signals(combined).iloc[:len(prefix)])


@pytest.mark.parametrize("kwargs", [{"exit": 2}, {"entry": 4}, {"window": 3}, {"cost_bps": -1}, {"initial_capital": 0}, {"stop": np.nan}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        StrategyConfig(**kwargs)

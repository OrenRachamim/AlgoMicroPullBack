import numpy as np
import pandas as pd
import pytest

from micro_pullback.indicators import add_indicators
from micro_pullback.strategy import (
    StrategyParams, compute_setup, consecutive_down_days,
    entry_price_for, initial_stop_target,
)


def test_consecutive_down_days():
    c = pd.Series([10, 9, 8, 9, 8, 7, 6, 7], dtype=float)
    out = consecutive_down_days(c)
    assert list(out) == [0, 1, 2, 0, 1, 2, 3, 0]


def synthetic_pullback_frame():
    """Uptrend for ~200 days then a 3-day shallow dip, ending at the dip low."""
    rng = pd.bdate_range("2023-01-02", periods=203)
    base = np.linspace(100, 200, 200)
    dip = [base[-1] * 0.99, base[-1] * 0.975, base[-1] * 0.965]  # -3.5% total
    closes = np.concatenate([base, dip])
    df = pd.DataFrame({
        "open": np.r_[closes[0], closes[:-1]],
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": 5_000_000,
    }, index=rng)
    return add_indicators(df)


def test_setup_fires_on_micro_pullback():
    df = synthetic_pullback_frame()
    p = StrategyParams(min_dollar_vol=1e6)
    setup = compute_setup(df, p)
    assert bool(setup.iloc[-1]), "3-day shallow dip in strong uptrend should trigger setup"
    # no setup while making fresh highs (day 150: no pullback)
    assert not bool(setup.iloc[150])


def test_setup_rejects_deep_pullback():
    df = synthetic_pullback_frame().copy()
    p = StrategyParams(min_dollar_vol=1e6, max_pullback_pct=0.02)  # dip is 3.5% -> too deep
    setup = compute_setup(df, p)
    assert not bool(setup.iloc[-1])


def test_setup_requires_trend():
    # downtrend with a small bounce-dip should never fire
    rng = pd.bdate_range("2023-01-02", periods=203)
    closes = np.linspace(200, 100, 203)
    df = pd.DataFrame({
        "open": np.r_[closes[0], closes[:-1]],
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": 5_000_000,
    }, index=rng)
    df = add_indicators(df)
    setup = compute_setup(df, StrategyParams(min_dollar_vol=1e6))
    assert not setup.any()


def test_entry_breakout_mode():
    p = StrategyParams(entry_mode="breakout", breakout_buffer=0.0)
    row = pd.Series({"open": 100.0, "high": 103.0, "low": 99.0, "close": 102.0})
    # prev high 101 -> trigger 101, open below trigger -> fill at trigger
    assert entry_price_for(row, 101.0, p) == pytest.approx(101.0)
    # gap above trigger -> fill at open
    row2 = pd.Series({"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.5})
    assert entry_price_for(row2, 101.0, p) == pytest.approx(105.0)
    # day never reaches trigger -> no entry
    row3 = pd.Series({"open": 100.0, "high": 100.5, "low": 99.0, "close": 100.2})
    assert entry_price_for(row3, 101.0, p) is None


def test_entry_open_mode():
    p = StrategyParams(entry_mode="open")
    row = pd.Series({"open": 100.0, "high": 100.5, "low": 99.0, "close": 100.2})
    assert entry_price_for(row, 999.0, p) == pytest.approx(100.0)


def test_stop_target_computation():
    p = StrategyParams(stop_atr_mult=2.0, stop_pct=0.05, target_pct=0.04, target_atr_mult=0.0)
    stop, target = initial_stop_target(100.0, 2.0, p)
    # ATR stop = 96, pct stop = 95 -> tighter (higher) wins
    assert stop == pytest.approx(96.0)
    assert target == pytest.approx(104.0)
    p2 = StrategyParams(stop_atr_mult=0.0, stop_pct=0.0, target_pct=0.0, target_atr_mult=0.0)
    stop2, target2 = initial_stop_target(100.0, 2.0, p2)
    assert stop2 == -np.inf and target2 == np.inf

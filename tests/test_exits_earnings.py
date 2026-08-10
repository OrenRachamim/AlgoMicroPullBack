import numpy as np
import pandas as pd
import pytest

from micro_pullback.backtest import BacktestConfig, run_backtest
from micro_pullback.earnings import risk_days
from micro_pullback.indicators import add_indicators
from micro_pullback.strategy import StrategyParams

from tests.test_backtest import make_universe_with_pullback


def test_risk_days_mapping():
    cal = pd.bdate_range("2024-01-01", periods=20)
    # filing on a Saturday snaps to next trading day; window = 1 before + 1 after
    days = risk_days(["2024-01-13"], cal, buffer_before=1, buffer_after=1)
    assert pd.Timestamp("2024-01-12") in days   # Friday before
    assert pd.Timestamp("2024-01-15") in days   # Monday (snap)
    assert pd.Timestamp("2024-01-16") in days   # day after
    assert len(days) == 3
    assert risk_days([], cal) == set()


def test_earnings_blocks_entry():
    data = make_universe_with_pullback()
    p = StrategyParams(min_dollar_vol=1e6)
    base_trades, _, _ = run_backtest(data, p, BacktestConfig())
    assert base_trades, "sanity: without earnings there is a trade"
    entry_day = base_trades[0].entry_date
    earnings = {"TEST": {entry_day}}
    trades, _, _ = run_backtest(data, p, BacktestConfig(), earnings=earnings)
    assert all(t.entry_date != entry_day for t in trades)


def test_earnings_forces_exit_before_risk_day():
    data = make_universe_with_pullback()
    p = StrategyParams(min_dollar_vol=1e6)
    base_trades, _, _ = run_backtest(data, p, BacktestConfig())
    tr = base_trades[0]
    df = data["TEST"]
    # put an earnings risk day in the middle of the base trade's holding period
    i_entry = df.index.get_loc(tr.entry_date)
    risk_day = df.index[i_entry + 2]
    trades, _, _ = run_backtest(data, p, BacktestConfig(),
                                earnings={"TEST": {risk_day}})
    tr2 = next(t for t in trades if t.entry_date == tr.entry_date)
    assert tr2.exit_reason == "earnings"
    assert tr2.exit_date == df.index[i_entry + 1]  # closed the day before


def test_earnings_filter_can_be_disabled():
    data = make_universe_with_pullback()
    p = StrategyParams(min_dollar_vol=1e6, avoid_earnings=False)
    base, _, _ = run_backtest(data, p, BacktestConfig())
    entry_day = base[0].entry_date
    trades, _, _ = run_backtest(data, p, BacktestConfig(),
                                earnings={"TEST": {entry_day}})
    assert any(t.entry_date == entry_day for t in trades)


def _trend_reversal_universe():
    """Rise for 200d, shallow dip, strong rally 8d, then hard sell-off 10d."""
    rng_len = 200 + 3 + 8 + 10
    rng = pd.bdate_range("2022-06-01", periods=rng_len)
    base = np.linspace(100, 200, 200)
    top = base[-1]
    dip = top * np.array([0.99, 0.975, 0.965])
    rally = dip[-1] * np.linspace(1.03, 1.15, 8)
    sell = rally[-1] * np.array([0.97 ** i for i in range(1, 11)])
    closes = np.concatenate([base, dip, rally, sell])
    prev = np.r_[closes[0], closes[:-1]]
    df = pd.DataFrame({
        "open": prev,
        "high": np.maximum(closes, prev) * 1.006,
        "low": np.minimum(closes, prev) * 0.994,
        "close": closes,
        "volume": 5_000_000,
    }, index=rng)
    return {"REV": add_indicators(df)}


def test_trailing_atr_stop_locks_profit():
    data = _trend_reversal_universe()
    p = StrategyParams(min_dollar_vol=1e6, stop_atr_mult=0.0, exit_rsi_min=0.0,
                       max_hold_days=40, trail_atr_mult=2.0)
    trades, _, _ = run_backtest(data, p, BacktestConfig())
    assert trades
    tr = trades[0]
    assert tr.exit_reason == "trail_stop"
    assert tr.ret_pct > 0, "trail should lock in rally profit before the sell-off erases it"


def test_trailing_pct_stop():
    data = _trend_reversal_universe()
    p = StrategyParams(min_dollar_vol=1e6, stop_atr_mult=0.0, exit_rsi_min=0.0,
                       max_hold_days=40, trail_pct=0.05)
    trades, _, _ = run_backtest(data, p, BacktestConfig())
    assert trades and trades[0].exit_reason == "trail_stop"
    # exit within ~5-6% of the peak close (gap slack)
    df = data["REV"]
    tr = trades[0]
    peak = df.loc[tr.entry_date:tr.exit_date, "close"].max()
    assert tr.exit_price >= peak * 0.93


def test_breakeven_stop():
    """After the rally arms breakeven, the sell-off must not turn the trade into a loss."""
    data = _trend_reversal_universe()
    p = StrategyParams(min_dollar_vol=1e6, stop_atr_mult=0.0, exit_rsi_min=0.0,
                       max_hold_days=40, breakeven_atr=1.0)
    trades, _, _ = run_backtest(data, p, BacktestConfig())
    assert trades
    tr = trades[0]
    assert tr.exit_reason == "trail_stop"
    assert tr.exit_price >= tr.entry_price * 0.98  # near-breakeven worst case (gap slack)


def test_exit_below_ema10():
    data = _trend_reversal_universe()
    p = StrategyParams(min_dollar_vol=1e6, stop_atr_mult=0.0, exit_rsi_min=0.0,
                       max_hold_days=40, exit_below_ema10=True)
    trades, _, _ = run_backtest(data, p, BacktestConfig())
    assert trades and trades[0].exit_reason == "below_ema10"

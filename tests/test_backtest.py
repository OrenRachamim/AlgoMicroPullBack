import numpy as np
import pandas as pd
import pytest

from micro_pullback.backtest import BacktestConfig, run_backtest, compute_stats, Trade
from micro_pullback.indicators import add_indicators
from micro_pullback.strategy import StrategyParams


def make_universe_with_pullback(n_pre=200, dip_days=3, recovery_days=10):
    """One ticker: long uptrend, shallow dip, then recovery — should produce >=1 trade."""
    total = n_pre + dip_days + recovery_days
    rng = pd.bdate_range("2022-06-01", periods=total)
    base = np.linspace(100, 200, n_pre)
    top = base[-1]
    dip = top * np.array([0.99, 0.975, 0.965][:dip_days])
    rec = dip[-1] * np.linspace(1.025, 1.12, recovery_days)
    closes = np.concatenate([base, dip, rec])
    df = pd.DataFrame({
        "open": np.r_[closes[0], closes[:-1]],
        "high": np.maximum(closes, np.r_[closes[0], closes[:-1]]) * 1.006,
        "low": np.minimum(closes, np.r_[closes[0], closes[:-1]]) * 0.994,
        "close": closes,
        "volume": 5_000_000,
    }, index=rng)
    return {"TEST": add_indicators(df)}


def test_backtest_executes_round_trip():
    data = make_universe_with_pullback()
    p = StrategyParams(min_dollar_vol=1e6)
    cfg = BacktestConfig(initial_capital=100_000, max_positions=2)
    trades, equity, stats = run_backtest(data, p, cfg)
    assert len(trades) >= 1
    tr = trades[0]
    assert tr.exit_date > tr.entry_date
    assert tr.bars_held <= p.max_hold_days
    assert stats["n_trades"] == len(trades)
    # recovery scenario should be profitable
    assert tr.pnl > 0


def test_no_lookahead_entry_after_setup():
    """Entry date must be strictly after the setup bar."""
    data = make_universe_with_pullback()
    p = StrategyParams(min_dollar_vol=1e6)
    from micro_pullback.strategy import compute_setup
    setup = compute_setup(data["TEST"], p)
    setup_dates = set(setup.index[setup])
    trades, _, _ = run_backtest(data, p, BacktestConfig())
    for tr in trades:
        prev_idx = data["TEST"].index.get_loc(tr.entry_date) - 1
        assert data["TEST"].index[prev_idx] in setup_dates


def test_stop_loss_enforced():
    """Crash after the dip: the ATR stop must cap the loss."""
    n_pre, dip_days, crash_days = 200, 3, 8
    total = n_pre + dip_days + crash_days
    rng = pd.bdate_range("2022-06-01", periods=total)
    base = np.linspace(100, 200, n_pre)
    top = base[-1]
    dip = top * np.array([0.99, 0.975, 0.965])
    # one bounce day (triggers breakout entry), then a crash
    crash = dip[-1] * np.array([1.02] + [1.02 * (0.95 ** i) for i in range(1, crash_days)])
    closes = np.concatenate([base, dip, crash])
    df = pd.DataFrame({
        "open": np.r_[closes[0], closes[:-1]],
        "high": np.maximum(closes, np.r_[closes[0], closes[:-1]]) * 1.006,
        "low": np.minimum(closes, np.r_[closes[0], closes[:-1]]) * 0.994,
        "close": closes,
        "volume": 5_000_000,
    }, index=rng)
    data = {"CRASH": add_indicators(df)}
    p = StrategyParams(min_dollar_vol=1e6, stop_atr_mult=1.5, max_hold_days=30, exit_rsi_min=0)
    trades, _, _ = run_backtest(data, p, BacktestConfig())
    assert len(trades) >= 1
    assert any(t.exit_reason == "stop" for t in trades)
    stop_trade = next(t for t in trades if t.exit_reason == "stop")
    assert stop_trade.ret_pct > -0.25  # loss bounded (gap risk aside)


def test_cash_and_position_accounting():
    data = make_universe_with_pullback()
    p = StrategyParams(min_dollar_vol=1e6)
    cfg = BacktestConfig(initial_capital=50_000, max_positions=1)
    trades, equity, _ = run_backtest(data, p, cfg)
    # final equity = initial + sum of pnl (single ticker, all closed)
    assert equity.iloc[-1] == pytest.approx(50_000 + sum(t.pnl for t in trades), rel=1e-9)


def test_max_positions_respected():
    dfs = make_universe_with_pullback()
    base = dfs["TEST"]
    data = {f"T{i}": base.copy() for i in range(5)}
    p = StrategyParams(min_dollar_vol=1e6)
    cfg = BacktestConfig(max_positions=2)
    trades, _, _ = run_backtest(data, p, cfg)
    # count concurrent holdings per day
    from collections import Counter
    open_days = Counter()
    for t in trades:
        for d in pd.bdate_range(t.entry_date, t.exit_date):
            open_days[d] += 1
    assert max(open_days.values()) <= 2


def test_compute_stats_basic():
    eq = pd.Series([100_000, 101_000, 100_500, 102_000],
                   index=pd.bdate_range("2024-01-01", periods=4))
    trades = [
        Trade("A", eq.index[0], eq.index[1], 10, 11, 100, 100.0, 0.10, 2, "target"),
        Trade("B", eq.index[1], eq.index[2], 10, 9.5, 100, -50.0, -0.05, 2, "stop"),
    ]
    s = compute_stats(trades, eq, 100_000)
    assert s["n_trades"] == 2
    assert s["win_rate_pct"] == 50.0
    assert s["profit_factor"] == 2.0
    assert s["total_return_pct"] == 2.0

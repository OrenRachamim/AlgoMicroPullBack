import numpy as np
import pandas as pd

from micro_pullback.indicators import add_indicators
from micro_pullback.strategy import StrategyParams, compute_setup


def make_fresh_cross_frame(n_below=300, n_up=80):
    """Long sideways/down base below the MAs, then a breakout above all MAs,
    then a shallow 3-day dip inside the fresh-uptrend window."""
    rng = pd.bdate_range("2022-01-03", periods=n_below + n_up + 3)
    # base: slow decline keeps price below the long MAs
    base = np.linspace(150, 100, n_below)
    # breakout: strong rally above every MA
    up = np.linspace(102, 160, n_up)
    top = up[-1]
    dip = top * np.array([0.99, 0.975, 0.965])
    closes = np.concatenate([base, up, dip])
    prev = np.r_[closes[0], closes[:-1]]
    df = pd.DataFrame({
        "open": prev,
        "high": np.maximum(closes, prev) * 1.005,
        "low": np.minimum(closes, prev) * 0.995,
        "close": closes,
        "volume": 5_000_000,
    }, index=rng)
    return add_indicators(df)


def fresh_params(**kw):
    base = dict(min_dollar_vol=1e6, require_sma50_above_sma100=False,
                require_rising_sma50=False, min_ret63=0.001,
                fresh_cross_lookback=60, fresh_min_days_below=126,
                fresh_max_prior_above=5)
    base.update(kw)
    return StrategyParams(**base)


def test_above_all_ma_column():
    df = make_fresh_cross_frame()
    # late in the rally the price is above all MAs; deep in the base it is not
    assert bool(df["above_all_ma"].iloc[-5])
    assert not bool(df["above_all_ma"].iloc[200])


def test_fresh_setup_fires_after_breakout_dip():
    df = make_fresh_cross_frame()
    p = fresh_params()
    setup = compute_setup(df, p)
    assert bool(setup.iloc[-1]), "dip shortly after a fresh cross should fire"


def test_fresh_filter_blocks_old_uptrends():
    """A stock that has been above all MAs for years never qualifies."""
    rng = pd.bdate_range("2022-01-03", periods=500)
    closes = np.linspace(100, 300, 500)
    # add the dip at the end
    closes[-3:] = closes[-4] * np.array([0.99, 0.975, 0.965])
    prev = np.r_[closes[0], closes[:-1]]
    df = pd.DataFrame({
        "open": prev,
        "high": np.maximum(closes, prev) * 1.005,
        "low": np.minimum(closes, prev) * 0.995,
        "close": closes,
        "volume": 5_000_000,
    }, index=rng)
    df = add_indicators(df)
    p = fresh_params()
    setup = compute_setup(df, p)
    assert not setup.iloc[-40:].any(), "long-established uptrend must not count as fresh"


def test_fresh_lookback_window_expires():
    df = make_fresh_cross_frame(n_below=300, n_up=80)
    # with a tiny lookback the dip (3 bars after an 80-bar rally) is too late
    p = fresh_params(fresh_cross_lookback=10)
    setup = compute_setup(df, p)
    assert not bool(setup.iloc[-1])


def test_moc_entry_fills_on_signal_day():
    """entry_mode='moc' enters at the signal day's close, not the next open."""
    from micro_pullback.backtest import BacktestConfig, run_backtest
    from tests.test_backtest import make_universe_with_pullback
    data = make_universe_with_pullback()
    p_open = StrategyParams(min_dollar_vol=1e6, entry_mode="open")
    p_moc = StrategyParams(min_dollar_vol=1e6, entry_mode="moc")
    tr_open, _, _ = run_backtest(data, p_open, BacktestConfig())
    tr_moc, _, _ = run_backtest(data, p_moc, BacktestConfig())
    assert tr_open and tr_moc
    df = data["TEST"]
    i_open = df.index.get_loc(tr_open[0].entry_date)
    i_moc = df.index.get_loc(tr_moc[0].entry_date)
    assert i_moc == i_open - 1, "MOC entry should be one bar earlier"
    assert tr_moc[0].entry_price == float(df.iloc[i_moc]["close"])


def test_fresh_ma_sma200_mode():
    """fresh_ma='sma200' keys the cross off the 200-day line alone."""
    df = make_fresh_cross_frame()
    p = fresh_params(fresh_ma="sma200", rsi_entry_max=30.0,
                     min_pullback_pct=0.005, max_pullback_pct=0.05)
    setup = compute_setup(df, p)
    assert setup.any(), "sma200-only cross should qualify the breakout stock"


def test_retake_entry_triggers_on_peak_recovery():
    """Retake mode: entry only when price takes out the pre-pullback peak."""
    import numpy as np
    from micro_pullback.backtest import BacktestConfig, run_backtest
    from micro_pullback.indicators import add_indicators
    # base below MAs, breakout rally, small 2-day dip, slow recovery that
    # crosses the pre-dip peak on the 3rd recovery day
    n_below, n_up = 300, 40
    base = np.linspace(150, 100, n_below)
    up = np.linspace(102, 140, n_up)
    top = up[-1]
    dip = top * np.array([0.99, 0.975])
    rec = top * np.array([0.985, 0.995, 1.02, 1.04, 1.05, 1.06])
    closes = np.concatenate([base, up, dip, rec])
    prev = np.r_[closes[0], closes[:-1]]
    rng = pd.bdate_range("2022-01-03", periods=len(closes))
    df = pd.DataFrame({"open": prev,
                       "high": np.maximum(closes, prev) * 1.004,
                       "low": np.minimum(closes, prev) * 0.996,
                       "close": closes, "volume": 5_000_000}, index=rng)
    data = {"RTK": add_indicators(df)}
    p = fresh_params(fresh_ma="sma200", entry_mode="retake",
                     pullback_within_days=5, rsi_entry_max=101.0,
                     min_pullback_pct=0.005, max_pullback_pct=0.04)
    trades, _, _ = run_backtest(data, p, BacktestConfig())
    assert trades, "recovery through the peak should trigger a retake entry"
    tr = trades[0]
    peak_before = data["RTK"].loc[:tr.entry_date, "res10"].iloc[-1]
    assert tr.entry_price >= peak_before, "fill must be at/above the retaken peak"


def test_one_trade_per_cycle():
    """With one_trade_per_cycle, a ticker trades at most once per breakout cycle."""
    import numpy as np
    from micro_pullback.backtest import BacktestConfig, run_backtest
    from micro_pullback.indicators import add_indicators
    # breakout, dip+recovery (trade 1), then another dip+recovery in the SAME cycle
    n_below, n_up = 300, 30
    base = np.linspace(150, 100, n_below)
    up = np.linspace(102, 140, n_up)
    seg = lambda top: top * np.array([0.99, 0.975, 0.985, 1.0, 1.02, 1.03])
    s1 = seg(up[-1]); s2 = seg(s1[-1]); s3 = seg(s2[-1])
    closes = np.concatenate([base, up, s1, s2, s3])
    prev = np.r_[closes[0], closes[:-1]]
    rng = pd.bdate_range("2022-01-03", periods=len(closes))
    df = pd.DataFrame({"open": prev,
                       "high": np.maximum(closes, prev) * 1.004,
                       "low": np.minimum(closes, prev) * 0.996,
                       "close": closes, "volume": 5_000_000}, index=rng)
    data = {"ONE": add_indicators(df)}
    kw = dict(fresh_ma="sma200", entry_mode="retake", pullback_within_days=5,
              rsi_entry_max=101.0, min_pullback_pct=0.005, max_pullback_pct=0.04,
              exit_rsi_min=0.0, max_hold_days=3)
    many, _, _ = run_backtest(data, fresh_params(**kw), BacktestConfig())
    once, _, _ = run_backtest(data, fresh_params(one_trade_per_cycle=True, **kw),
                              BacktestConfig())
    assert len(many) >= 2, "sanity: without the flag there are repeat trades"
    assert len(once) == 1, "with the flag only the first trade of the cycle happens"
    assert once[0].entry_date == many[0].entry_date


def test_ma_retest_setup_and_entry():
    """MA-retest: touch of SMA20 inside the fresh window + breakout reclaim."""
    import numpy as np
    from micro_pullback.backtest import BacktestConfig, run_backtest
    from micro_pullback.indicators import add_indicators
    # base below MAs -> breakout rally -> drift down to SMA20 -> reclaim up
    n_below, n_up = 300, 35
    base = np.linspace(150, 100, n_below)
    up = np.linspace(102, 135, n_up)
    top = up[-1]
    drift = top * np.array([0.995, 0.985, 0.975, 0.968, 0.966])  # back to ~SMA20
    rec = drift[-1] * np.linspace(1.015, 1.08, 6)
    closes = np.concatenate([base, up, drift, rec])
    prev = np.r_[closes[0], closes[:-1]]
    rng = pd.bdate_range("2022-01-03", periods=len(closes))
    df = pd.DataFrame({"open": prev,
                       "high": np.maximum(closes, prev) * 1.004,
                       "low": np.minimum(closes, prev) * 0.994,
                       "close": closes, "volume": 5_000_000}, index=rng)
    data = {"RTST": add_indicators(df)}
    p = fresh_params(ma_touch_col="sma20", entry_mode="breakout",
                     rsi_entry_max=101.0, min_down_days=0,
                     require_rsi14_min=0.0, exit_rsi_min=0.0, max_hold_days=5)
    trades, _, _ = run_backtest(data, p, BacktestConfig())
    assert trades, "touch of SMA20 followed by reclaim should produce a trade"

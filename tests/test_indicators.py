import numpy as np
import pandas as pd
import pytest

from micro_pullback.indicators import (
    add_indicators, atr, ema, macd, rsi, sma, support_level, resistance_level,
)


def make_ohlc(closes):
    closes = pd.Series(closes, index=pd.bdate_range("2023-01-02", periods=len(closes)))
    return pd.DataFrame({
        "open": closes.shift(1).fillna(closes.iloc[0]),
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": 1_000_000,
    })


def test_sma_matches_manual():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_converges_to_constant():
    s = pd.Series([10.0] * 50)
    out = ema(s, 10)
    assert out.iloc[-1] == pytest.approx(10.0)


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(100, 200, 60))
    down = pd.Series(np.linspace(200, 100, 60))
    rsi_up = rsi(up, 14).iloc[-1]
    rsi_down = rsi(down, 14).iloc[-1]
    assert 99.0 <= rsi_up <= 100.0     # pure uptrend -> RSI ~100
    assert 0.0 <= rsi_down <= 1.0      # pure downtrend -> RSI ~0
    mixed = pd.Series(100 + np.sin(np.arange(100)))
    vals = rsi(mixed, 14).dropna()
    assert ((vals >= 0) & (vals <= 100)).all()


def test_rsi_no_lookahead():
    s = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 100)))
    full = rsi(s, 14)
    partial = rsi(s.iloc[:60], 14)
    pd.testing.assert_series_equal(full.iloc[:60], partial)


def test_macd_sign_in_trend():
    up = pd.Series(np.linspace(100, 300, 120))
    line, sig, hist = macd(up)
    assert line.iloc[-1] > 0
    down = pd.Series(np.linspace(300, 100, 120))
    line_d, _, _ = macd(down)
    assert line_d.iloc[-1] < 0


def test_atr_positive_and_scales():
    df = make_ohlc(np.linspace(100, 120, 60))
    a = atr(df, 14).dropna()
    assert (a > 0).all()
    df2 = df.copy()
    for col in ("open", "high", "low", "close"):
        df2[col] *= 10
    a2 = atr(df2, 14).dropna()
    assert a2.iloc[-1] == pytest.approx(10 * a.iloc[-1], rel=1e-6)


def test_support_resistance_shifted():
    df = make_ohlc(np.arange(100, 160, dtype=float))
    sup = support_level(df, 20)
    res = resistance_level(df, 20)
    # support/resistance on bar t must ignore bar t itself (shift by 1)
    assert sup.iloc[25] == pytest.approx(df["low"].iloc[5:25].min())
    assert res.iloc[25] == pytest.approx(df["high"].iloc[5:25].max())


def test_add_indicators_columns():
    df = make_ohlc(100 + np.cumsum(np.random.default_rng(1).normal(0.1, 1, 200)))
    out = add_indicators(df)
    for col in ["sma20", "sma50", "sma100", "ema20", "rsi2", "rsi3", "rsi14",
                "macd", "macd_signal", "macd_hist", "atr14", "hi20",
                "support20", "resistance20", "ret63", "dollar_vol20"]:
        assert col in out.columns
        assert out[col].notna().any()

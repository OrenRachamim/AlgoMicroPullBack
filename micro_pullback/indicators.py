"""Technical indicators implemented with pandas (no external TA library).

All functions take/return pandas Series aligned to the input index and use
only past data at each point (no lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.fillna(100.0).where(avg_gain.notna() & avg_loss.notna(), np.nan)
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range from an OHLC frame."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).max()


def rolling_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).min()


def support_level(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Nearest support: rolling minimum of lows over `window` days (shifted 1 day,
    so today's bar never defines its own support)."""
    return df["low"].rolling(window, min_periods=window).min().shift(1)


def resistance_level(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Nearest resistance: rolling maximum of highs over `window` days (shifted 1 day)."""
    return df["high"].rolling(window, min_periods=window).max().shift(1)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the indicator columns used by the strategies."""
    out = df.copy()
    c = out["close"]
    out["sma20"] = sma(c, 20)
    out["sma50"] = sma(c, 50)
    out["sma100"] = sma(c, 100)
    out["ema10"] = ema(c, 10)
    out["ema20"] = ema(c, 20)
    out["rsi2"] = rsi(c, 2)
    out["rsi3"] = rsi(c, 3)
    out["rsi14"] = rsi(c, 14)
    macd_line, signal_line, hist = macd(c)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    out["atr14"] = atr(out, 14)
    out["hi10"] = rolling_high(c, 10).shift(1)   # 10-day closing high as of yesterday
    out["hi20"] = rolling_high(c, 20).shift(1)
    out["support20"] = support_level(out, 20)
    out["resistance20"] = resistance_level(out, 20)
    out["hi252"] = rolling_high(c, 252).shift(1)  # 52-week closing high as of yesterday
    out["ret63"] = c.pct_change(63)              # ~3-month momentum
    out["ret126"] = c.pct_change(126)            # ~6-month momentum
    out["dollar_vol20"] = (c * out["volume"]).rolling(20, min_periods=20).mean()
    return out

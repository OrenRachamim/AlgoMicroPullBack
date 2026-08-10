"""Micro-pullback strategy: setup detection and exit rules.

Core idea
---------
1. Trend filter  – the stock is in a strong uptrend (price above rising moving
   averages, positive multi-month momentum, MACD bullish).
2. Micro pullback – a shallow, short dip: a few red days / short-term RSI
   oversold while the longer RSI stays healthy, price still above support.
3. Entry trigger – the dip stops: next day price takes out the previous day's
   high (buy-the-resumption), or simply next open (configurable).
4. Exit          – small profit target / protective stop (ATR based) /
   strength exit (short RSI overbought) / time stop after a few days.

Everything is parameterized through StrategyParams so backtest iterations can
sweep variants without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    name: str = "base"

    # --- trend filter ---
    require_above_sma50: bool = True
    require_sma50_above_sma100: bool = True
    require_rising_sma50: bool = True        # sma50 today > sma50 10 days ago
    min_ret63: float = 0.05                  # >=5% over ~3 months
    macd_mode: str = "line_positive"         # "line_positive": macd>0 | "above_signal" | "none"
    require_rsi14_min: float = 40.0          # longer-term RSI still healthy
    min_pct_of_52w_high: float = 0.0         # close >= X * 52-week high (0 disables)

    # --- pullback definition ---
    rsi_entry_col: str = "rsi3"              # short RSI used for the dip
    rsi_entry_max: float = 30.0              # dip: short RSI below this
    min_down_days: int = 2                   # at least N consecutive red closes
    max_pullback_pct: float = 0.10           # not more than 10% off the 20d high
    min_pullback_pct: float = 0.01           # at least a real dip (1%)
    require_above_support: bool = True       # close above 20d support level
    max_dist_ema20: float = 0.03             # close within 3% of / above EMA20 - support zone

    # --- entry ---
    entry_mode: str = "breakout"             # "breakout": next day takes out prev high; "open": next open
    breakout_buffer: float = 0.001           # 0.1% above previous high

    # --- exit ---
    stop_atr_mult: float = 1.5               # stop = entry - mult*ATR (0 disables)
    stop_pct: float = 0.0                    # fixed % stop (0 disables)
    stop_close_pct: float = 0.0              # exit at close when close <= entry*(1-x) (0 disables)
    target_pct: float = 0.0                  # fixed % profit target (0 disables)
    target_atr_mult: float = 0.0             # target = entry + mult*ATR (0 disables)
    exit_rsi_col: str = "rsi3"
    exit_rsi_min: float = 75.0               # strength exit when short RSI overbought (0 disables)
    max_hold_days: int = 5                   # time stop (trading days after entry)

    # --- risk guards ---
    max_atr_pct: float = 0.0                 # skip stocks whose ATR14/close exceeds this (0 disables)

    # --- liquidity / universe hygiene ---
    min_price: float = 5.0
    min_dollar_vol: float = 20e6             # 20M$ average daily dollar volume

    def to_dict(self) -> dict:
        return asdict(self)


def consecutive_down_days(close: pd.Series) -> pd.Series:
    """Number of consecutive down closes ending at each bar."""
    down = (close.diff() < 0).astype(int)
    # reset cumulative count at every up/flat day
    grp = (down == 0).cumsum()
    return down.groupby(grp).cumsum()


def compute_setup(df: pd.DataFrame, p: StrategyParams) -> pd.Series:
    """Boolean Series: True on bars where the micro-pullback setup is complete.

    The signal on bar t means: conditions verified with data through t's close;
    any entry happens on bar t+1 (handled by the backtester).
    """
    c = df["close"]
    ok = pd.Series(True, index=df.index)

    # liquidity
    ok &= c >= p.min_price
    ok &= df["dollar_vol20"] >= p.min_dollar_vol

    # trend filter
    if p.require_above_sma50:
        ok &= c > df["sma50"]
    if p.require_sma50_above_sma100:
        ok &= df["sma50"] > df["sma100"]
    if p.require_rising_sma50:
        ok &= df["sma50"] > df["sma50"].shift(10)
    if p.min_ret63 > 0:
        ok &= df["ret63"] >= p.min_ret63
    if p.macd_mode == "line_positive":
        ok &= df["macd"] > 0
    elif p.macd_mode == "above_signal":
        ok &= df["macd"] > df["macd_signal"]
    if p.require_rsi14_min > 0:
        ok &= df["rsi14"] >= p.require_rsi14_min
    if p.min_pct_of_52w_high > 0:
        ok &= c >= df["hi252"] * p.min_pct_of_52w_high

    # pullback
    ok &= df[p.rsi_entry_col] <= p.rsi_entry_max
    if p.min_down_days > 0:
        ok &= consecutive_down_days(c) >= p.min_down_days
    drawdown_from_hi = 1.0 - c / df["hi20"]
    ok &= drawdown_from_hi.between(p.min_pullback_pct, p.max_pullback_pct)
    if p.require_above_support:
        ok &= c > df["support20"]
    if p.max_dist_ema20 is not None:
        # close not too far below EMA20: sitting on/near dynamic support
        ok &= c >= df["ema20"] * (1.0 - p.max_dist_ema20)

    # volatility guard: avoid falling knives / crash-regime names
    if p.max_atr_pct > 0:
        ok &= (df["atr14"] / c) <= p.max_atr_pct

    # need valid indicators
    needed = ["sma50", "sma100", "rsi14", "macd", "macd_signal", "atr14",
              "hi20", "support20", "ema20", "dollar_vol20", p.rsi_entry_col]
    for col in set(needed):
        ok &= df[col].notna()

    return ok.fillna(False)


@dataclass
class OpenPosition:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop: float
    target: float
    bars_held: int = 0


def entry_price_for(df_row: pd.Series, prev_high: float, p: StrategyParams) -> float | None:
    """Given the bar after a setup, return the fill price or None if no entry.

    breakout mode: enter only if the day's high exceeds prev_high*(1+buffer);
    fill at the breakout level (or the open when it gaps above it).
    open mode: always enter at the open.
    """
    o, h = df_row["open"], df_row["high"]
    if p.entry_mode == "open":
        return float(o)
    trigger = prev_high * (1.0 + p.breakout_buffer)
    if h >= trigger:
        return float(max(o, trigger))
    return None


def initial_stop_target(entry: float, atr_value: float, p: StrategyParams) -> tuple[float, float]:
    """Compute (stop, target); 0/absent legs return -inf / +inf respectively."""
    stops = []
    if p.stop_atr_mult > 0 and np.isfinite(atr_value):
        stops.append(entry - p.stop_atr_mult * atr_value)
    if p.stop_pct > 0:
        stops.append(entry * (1.0 - p.stop_pct))
    stop = max(stops) if stops else -np.inf

    targets = []
    if p.target_pct > 0:
        targets.append(entry * (1.0 + p.target_pct))
    if p.target_atr_mult > 0 and np.isfinite(atr_value):
        targets.append(entry + p.target_atr_mult * atr_value)
    target = min(targets) if targets else np.inf
    return stop, target

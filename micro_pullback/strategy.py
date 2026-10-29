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

    # --- fresh-breakout regime (0 disables) ---
    # Restrict setups to stocks that recently crossed above ALL moving
    # averages (SMA10/20/50/100/200) for the first time after a long stretch
    # of not being above them: the start of a new uptrend.
    fresh_cross_lookback: int = 0            # setup allowed within N trading days after a qualifying cross
    fresh_min_days_below: int = 126          # prior stretch (trading days) in which the stock was (almost) never above all MAs
    fresh_max_prior_above: int = 5           # tolerance: max days above-all inside that prior stretch
    fresh_ma: str = "all"                    # which line defines the cross: "all" (SMA10..200) or "sma200" only
    one_trade_per_cycle: bool = False        # take only the FIRST trade per breakout cycle; the ticker re-arms only after a new qualifying cross (long stretch below the line again)

    # --- pullback definition ---
    rsi_entry_col: str = "rsi3"              # short RSI used for the dip
    rsi_entry_max: float = 30.0              # dip: short RSI below this
    min_down_days: int = 2                   # at least N consecutive red closes
    max_pullback_pct: float = 0.10           # not more than 10% off the 20d high
    min_pullback_pct: float = 0.01           # at least a real dip (1%)
    require_above_support: bool = True       # close above 20d support level
    max_dist_ema20: float = 0.03             # close within 3% of / above EMA20 - support zone

    # --- entry ---
    entry_mode: str = "breakout"             # "breakout": next day takes out prev high | "open": next open | "moc": signal day's close | "retake": price retakes the pre-pullback peak
    breakout_buffer: float = 0.001           # 0.1% above previous high
    pullback_within_days: int = 0            # >0: pullback may have occurred up to N days ago (retake pattern)

    # --- exit ---
    stop_atr_mult: float = 1.5               # stop = entry - mult*ATR (0 disables)
    stop_pct: float = 0.0                    # fixed % stop (0 disables)
    stop_close_pct: float = 0.0              # exit at close when close <= entry*(1-x) (0 disables)
    target_pct: float = 0.0                  # fixed % profit target (0 disables)
    target_atr_mult: float = 0.0             # target = entry + mult*ATR (0 disables)
    exit_rsi_col: str = "rsi3"
    exit_rsi_min: float = 75.0               # strength exit when short RSI overbought (0 disables)
    max_hold_days: int = 5                   # time stop (trading days after entry)
    trail_atr_mult: float = 0.0              # chandelier trail: highest-high-since-entry - mult*ATR (0 disables)
    trail_pct: float = 0.0                   # trail: peak close since entry * (1-pct) (0 disables)
    breakeven_atr: float = 0.0               # once high >= entry + X*ATR, raise stop to entry (0 disables)
    exit_below_ema10: bool = False           # exit at close when close < EMA10

    # --- earnings avoidance ---
    avoid_earnings: bool = True              # skip entries on earnings-risk days; exit before them

    # --- risk guards ---
    # --- MA-retest pattern (0/"" disables) ---
    # The pullback is defined as price coming back DOWN to a moving average
    # (the day's low touches it within a tolerance) instead of by depth from
    # the recent high. Pair with entry_mode="breakout" so the entry fires only
    # when price turns back UP through the previous day's high (the reclaim).
    ma_touch_col: str = ""                   # e.g. "sma20" or "sma50"
    ma_touch_tol: float = 0.01               # low <= MA*(1+tol) counts as a touch
    ma_touch_floor: str = ""                 # "sma100" | "sma200" | "both": close must also hold above these on the touch day
    rising_sma20_days: int = 0               # >0: SMA20 must be HIGHER than it was N days ago (buy only while the line slopes up)

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
    if p.rising_sma20_days > 0:
        ok &= df["sma20"] > df["sma20"].shift(p.rising_sma20_days)
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

    # fresh-breakout regime: a qualifying cross above all MAs happened within
    # the last `fresh_cross_lookback` days, after >= `fresh_min_days_below`
    # days of (almost) never being above them.
    if p.fresh_cross_lookback > 0:
        if p.fresh_ma == "sma200":
            above = (c > df["sma200"]).fillna(False)
        else:
            above = df["above_all_ma"].fillna(False)
        prior_above = above.shift(1).rolling(p.fresh_min_days_below,
                                             min_periods=p.fresh_min_days_below).sum()
        cross = above & above.shift(1).notna() & (prior_above <= p.fresh_max_prior_above)
        # also require the MAs themselves to be valid (enough history)
        cross &= df["sma200"].notna()
        fresh = cross.rolling(p.fresh_cross_lookback, min_periods=1).max().astype(bool)
        ok &= fresh

    # pullback — collected separately so it can be applied either same-day
    # (default) or "happened within the last N days" (retake pattern).
    pull = pd.Series(True, index=df.index)
    if p.ma_touch_col:
        # MA-retest: the day's low reaches the chosen MA (within tolerance)
        # while the close holds above the slower structure (SMA50 zone).
        ma = df[p.ma_touch_col]
        pull &= df["low"] <= ma * (1.0 + p.ma_touch_tol)
        pull &= c > df["sma50"] * 0.99
        pull &= ma.notna()
        if p.rsi_entry_max < 100.0:  # optional confluence: deep dip AT the MA
            pull &= df[p.rsi_entry_col] <= p.rsi_entry_max
        if p.ma_touch_floor in ("sma100", "both"):
            pull &= c > df["sma100"]
        if p.ma_touch_floor in ("sma200", "both"):
            pull &= c > df["sma200"]
    else:
        pull &= df[p.rsi_entry_col] <= p.rsi_entry_max
        if p.min_down_days > 0:
            pull &= consecutive_down_days(c) >= p.min_down_days
        drawdown_from_hi = 1.0 - c / df["hi20"]
        pull &= drawdown_from_hi.between(p.min_pullback_pct, p.max_pullback_pct)
        if p.require_above_support:
            pull &= c > df["support20"]
        if p.max_dist_ema20 is not None:
            # close not too far below EMA20: sitting on/near dynamic support
            pull &= c >= df["ema20"] * (1.0 - p.max_dist_ema20)

    if p.pullback_within_days > 0:
        # retake pattern: a qualifying pullback occurred within the last N
        # days and the pre-pullback peak (10d high of highs) has not been
        # retaken yet at today's close — the entry trigger is that peak.
        recent_pull = pull.rolling(p.pullback_within_days, min_periods=1).max().astype(bool)
        ok &= recent_pull & (c < df["res10"]) & df["res10"].notna()
    else:
        ok &= pull

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
    # "breakout" and "retake" share the same mechanics; only the trigger level
    # differs (previous day's high vs the pre-pullback peak, chosen by caller).
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


def compute_cycle_id(df: pd.DataFrame, p: StrategyParams) -> pd.Series:
    """Breakout-cycle id per bar (0 before the first cycle).

    A cycle starts at a qualifying fresh cross (first close above the line
    after >= fresh_min_days_below days with at most fresh_max_prior_above days
    above). Qualifying crosses within 10 trading days of the previous one are
    tolerance artifacts of the same breakout, not a new cycle. A new cycle can
    only begin after the price has again spent the long stretch below the
    line — so "one trade per cycle" means: trade the first setup after the
    cross, then wait for the stock to reset like a new one.
    """
    c = df["close"]
    if p.fresh_ma == "sma200":
        above = (c > df["sma200"]).fillna(False)
    else:
        above = df["above_all_ma"].fillna(False)
    prior_above = above.shift(1).rolling(p.fresh_min_days_below,
                                         min_periods=p.fresh_min_days_below).sum()
    cross = above & above.shift(1).notna() & (prior_above <= p.fresh_max_prior_above)
    cross &= df["sma200"].notna()
    # cluster: a cross with no cross in the prior 10 bars starts a new cycle
    recent = cross.shift(1).rolling(10, min_periods=1).sum().fillna(0)
    start = cross & (recent == 0)
    return start.cumsum()

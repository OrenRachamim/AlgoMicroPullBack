#!/usr/bin/env python3
"""Event study: follow-through after a small dip or a rising-slowdown.

Within the fresh-breakout window (the official strategy's eligibility regime),
define several light-touch events on bar t and measure forward returns from
the NEXT day's open (consistent with live execution) to the close 1, 3 and 5
trading days later.

Events:
  dip1       - exactly 1 red day, total decline from the prior close peak 0.3%-3%
  dip2       - exactly 2 consecutive red days, same small-decline band
  slow       - deceleration: two consecutive up closes, today's gain < yesterday's
  slow_hard  - strong slowdown: both up, today's gain <= half of yesterday's
  deep_ref   - the official trigger (RSI2<10, dip 1%-7%) for reference
  all_days   - every eligible day in the fresh window (baseline drift)

Usage:
  python analyze_followthrough.py --universe broad --start 2023-10-19 --end 2026-10-19
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from micro_pullback.data import fetch_daily, load_universe
from micro_pullback.indicators import add_indicators
from micro_pullback.strategy import StrategyParams, compute_setup, consecutive_down_days
from micro_pullback.universe import EXTENDED_UNIVERSE, FULL_UNIVERSE, NASDAQ_UNIVERSE, SMALL_UNIVERSE
from micro_pullback.variants import VARIANTS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
HORIZONS = [1, 3, 5]


def eligibility_mask(df: pd.DataFrame, spy_ok: pd.Series) -> pd.Series:
    """The fresh-breakout eligibility of the official strategy, WITHOUT any
    pullback condition: fresh cross window + trend/hygiene gates + SPY regime."""
    p = VARIANTS["final"]
    neutral = StrategyParams(
        name="eligibility", entry_mode="open",
        require_above_sma50=p.require_above_sma50,
        require_sma50_above_sma100=p.require_sma50_above_sma100,
        require_rising_sma50=p.require_rising_sma50,
        min_ret63=p.min_ret63, macd_mode=p.macd_mode,
        require_rsi14_min=p.require_rsi14_min,
        fresh_cross_lookback=p.fresh_cross_lookback,
        fresh_min_days_below=p.fresh_min_days_below,
        fresh_max_prior_above=p.fresh_max_prior_above,
        fresh_ma=p.fresh_ma,
        max_atr_pct=p.max_atr_pct,
        min_price=p.min_price, min_dollar_vol=p.min_dollar_vol,
        # neutralize the pullback gates
        rsi_entry_col="rsi2", rsi_entry_max=101.0, min_down_days=0,
        min_pullback_pct=-1.0, max_pullback_pct=10.0,
        require_above_support=False, max_dist_ema20=None,
    )
    ok = compute_setup(df, neutral)
    return ok & spy_ok.reindex(df.index).fillna(False)


def forward_returns(df: pd.DataFrame, events: pd.Series) -> dict[int, np.ndarray]:
    """Entry at next open after event day t; return to close of t+h."""
    o, c = df["open"].values, df["close"].values
    n = len(df)
    out = {h: [] for h in HORIZONS}
    for i in np.flatnonzero(events.values):
        e = i + 1  # entry bar
        if e >= n:
            continue
        entry = o[e]
        if not np.isfinite(entry) or entry <= 0:
            continue
        for h in HORIZONS:
            j = e + h - 1  # close h trading days after entry day open (1d = entry day close)
            if j < n:
                out[h].append(c[j] / entry - 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["small", "extended", "nasdaq", "full", "broad"], default="broad")
    ap.add_argument("--start", default="2023-10-19")
    ap.add_argument("--end", default="2026-10-19")
    ap.add_argument("--broad-limit", type=int, default=600)
    ap.add_argument("--warmup-days", type=int, default=700)
    args = ap.parse_args()

    if args.universe == "broad":
        from micro_pullback.universe import get_broad_universe
        tickers = get_broad_universe(limit=args.broad_limit)
    else:
        tickers = {"small": SMALL_UNIVERSE, "extended": EXTENDED_UNIVERSE,
                   "nasdaq": NASDAQ_UNIVERSE, "full": FULL_UNIVERSE}[args.universe]
    fetch_start = (dt.date.fromisoformat(args.start) - dt.timedelta(days=args.warmup_days)).isoformat()
    print(f"Loading {len(tickers)} tickers ...")
    raw = load_universe(tickers, fetch_start, args.end, verbose=False)
    print(f"Loaded {len(raw)}")

    spy = fetch_daily("SPY", (dt.date.fromisoformat(args.start) - dt.timedelta(days=500)).isoformat(), args.end)
    spy_ok = (spy["close"] > spy["close"].rolling(200, min_periods=200).mean()).shift(1).fillna(False)

    deep = VARIANTS["final"]
    agg: dict[str, dict[int, list]] = {}

    for t, df0 in raw.items():
        df = add_indicators(df0)
        elig = eligibility_mask(df, spy_ok)
        win = df.index >= pd.Timestamp(args.start)
        elig = elig & win
        if not elig.any():
            continue

        c = df["close"]
        ret = c.pct_change()
        down = consecutive_down_days(c)
        # decline from the highest close of the last 5 days (small-dip size)
        peak5 = c.rolling(5, min_periods=1).max()
        dd = 1.0 - c / peak5

        small = dd.between(0.003, 0.03)
        events = {
            "dip1": elig & (down == 1) & small,
            "dip2": elig & (down == 2) & small,
            "slow": elig & (ret > 0) & (ret.shift(1) > 0) & (ret < ret.shift(1)),
            "slow_hard": elig & (ret > 0) & (ret.shift(1) > 0) & (ret <= 0.5 * ret.shift(1)),
            "deep_ref": compute_setup(df, deep) & spy_ok.reindex(df.index).fillna(False) & win,
            "all_days": elig,
        }
        for name, ev in events.items():
            fr = forward_returns(df, ev)
            slot = agg.setdefault(name, {h: [] for h in HORIZONS})
            for h in HORIZONS:
                slot[h].extend(fr[h])

    rows = []
    for name in ["dip1", "dip2", "slow", "slow_hard", "deep_ref", "all_days"]:
        row = {"event": name, "n": len(agg[name][1])}
        for h in HORIZONS:
            a = np.array(agg[name][h])
            row[f"avg_{h}d_pct"] = round(100 * a.mean(), 3)
            row[f"med_{h}d_pct"] = round(100 * np.median(a), 3)
            row[f"win_{h}d_pct"] = round(100 * (a > 0).mean(), 1)
        rows.append(row)

    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"followthrough_{args.universe}.json")
    with open(path, "w") as f:
        json.dump({"start": args.start, "end": args.end, "universe": args.universe,
                   "rows": rows}, f, indent=1)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()

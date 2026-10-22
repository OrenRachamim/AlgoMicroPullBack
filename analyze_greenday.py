#!/usr/bin/env python3
"""Green-day probability zones + the intraday-else-hold hybrid.

For every condition ("zone") measurable at the OPEN of day t (yesterday's
close-based indicators + today's opening gap), estimate:
  - P(green): probability the day closes above its open
  - avg/median open->close return
  - the user's hybrid: take close-open if the day is green, otherwise KEEP the
    position and exit at the close 5 / 21 trading days later
  - the pure alternatives: always exit at close (intraday) / always hold 5d

Zones (within SPY>SMA200, liquidity/price hygiene):
  all_days      - every eligible day (baseline)
  fresh_all     - inside the fresh above-all-MAs window (official regime)
  deep_dip      - yesterday RSI2<10 inside the fresh window (official entry day)
  deep_gapdown  - deep_dip AND today opens below yesterday's close
  deep_gapup    - deep_dip AND today opens above yesterday's close
  down3         - 3+ consecutive red closes (any eligible stock)
  post_cross    - first 3 days after a qualifying SMA200 cross
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
from micro_pullback.variants import VARIANTS
from analyze_followthrough import eligibility_mask

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-10-19")
    ap.add_argument("--end", default="2026-10-19")
    ap.add_argument("--broad-limit", type=int, default=600)
    args = ap.parse_args()

    from micro_pullback.universe import get_broad_universe
    tickers = get_broad_universe(limit=args.broad_limit)
    fetch_start = (dt.date.fromisoformat(args.start) - dt.timedelta(days=700)).isoformat()
    print(f"Loading {len(tickers)} tickers ...")
    raw = load_universe(tickers, fetch_start, args.end, verbose=False)
    print(f"Loaded {len(raw)}")

    spy = fetch_daily("SPY", (dt.date.fromisoformat(args.start) - dt.timedelta(days=500)).isoformat(), args.end)
    spy_ok = (spy["close"] > spy["close"].rolling(200, min_periods=200).mean()).shift(1).fillna(False)

    deep = VARIANTS["final"]  # official deep-dip setup (fresh all-MA window)

    zones = ["all_days", "fresh_all", "deep_dip", "deep_gapdown", "deep_gapup",
             "down3", "post_cross"]
    agg = {z: {"intra": [], "hold5": [], "hold21": []} for z in zones}

    for t, df0 in raw.items():
        df = add_indicators(df0)
        win = df.index >= pd.Timestamp(args.start)
        elig = eligibility_mask(df, spy_ok)
        # basic hygiene without the fresh window for the generic zones
        basic = (df["close"] >= 5.0) & (df["dollar_vol20"] >= 20e6) \
            & spy_ok.reindex(df.index).fillna(False) & df["sma200"].notna()

        deep_setup = compute_setup(df, deep)
        o, c, pc = df["open"], df["close"], df["close"].shift(1)
        gap_dn = o < pc
        down = consecutive_down_days(df["close"])

        above = (c > df["sma200"]).fillna(False)
        prior = above.shift(1).rolling(126, min_periods=126).sum()
        cross = above & (prior <= 5) & df["sma200"].notna()
        post_cross = cross.rolling(3, min_periods=1).max().astype(bool)

        # condition is evaluated for trading ON day t (signal from t-1 + today's gap)
        conds = {
            "all_days": basic & win,
            "fresh_all": elig & win,
            "deep_dip": deep_setup.shift(1).fillna(False) & win,
            "deep_gapdown": deep_setup.shift(1).fillna(False) & gap_dn & win,
            "deep_gapup": deep_setup.shift(1).fillna(False) & ~gap_dn & win,
            "down3": basic & (down.shift(1) >= 3) & win,
            "post_cross": basic & post_cross.shift(1).fillna(False) & win,
        }

        ov, cv = o.values, c.values
        n = len(df)
        for z, mask in conds.items():
            for i in np.flatnonzero(mask.values):
                if not np.isfinite(ov[i]) or ov[i] <= 0:
                    continue
                intra = cv[i] / ov[i] - 1.0
                agg[z]["intra"].append(intra)
                for name, h in (("hold5", 5), ("hold21", 21)):
                    j = min(n - 1, i + h - 1)
                    hybrid = intra if intra > 0 else cv[j] / ov[i] - 1.0
                    agg[z][name].append(hybrid)

    rows = []
    for z in zones:
        a = np.array(agg[z]["intra"])
        h5 = np.array(agg[z]["hold5"])
        h21 = np.array(agg[z]["hold21"])
        losers5 = h5[a <= 0]
        rows.append({
            "zone": z, "n": len(a),
            "p_green_pct": round(100 * (a > 0).mean(), 1),
            "avg_intraday_pct": round(100 * a.mean(), 3),
            "med_intraday_pct": round(100 * np.median(a), 3),
            "hybrid5_avg_pct": round(100 * h5.mean(), 3),
            "hybrid21_avg_pct": round(100 * h21.mean(), 3),
            "fallback_book_avg_pct": round(100 * losers5.mean(), 3) if len(losers5) else None,
        })
    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "greenday_zones.json"), "w") as f:
        json.dump({"start": args.start, "end": args.end, "rows": rows}, f, indent=1)
    print("\nSaved -> results/greenday_zones.json")


if __name__ == "__main__":
    main()

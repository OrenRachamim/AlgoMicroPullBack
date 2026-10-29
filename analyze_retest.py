#!/usr/bin/env python3
"""Count MA-retest events inside fresh-breakout windows and measure outcomes.

Event: inside the fresh window (above-all-MAs cross), after at least 3 days
with lows ABOVE SMA20, the low touches SMA20 (within 1%). Touch days cluster
into one event. Outcomes per event:
  - reclaimed within 5 days? (a later day's high exceeds the touch-cluster high)
  - forward return 5 trading days from the reclaim entry price
"""
import datetime as dt, json, os
import numpy as np, pandas as pd
from micro_pullback.data import fetch_daily, load_universe
from micro_pullback.indicators import add_indicators
from micro_pullback.universe import get_broad_universe
from micro_pullback.variants import VARIANTS
from micro_pullback.strategy import compute_setup, StrategyParams
from analyze_followthrough import eligibility_mask

START, END = "2018-01-02", "2026-10-19"
tickers = get_broad_universe(limit=600)
fetch_start = (dt.date.fromisoformat(START) - dt.timedelta(days=700)).isoformat()
raw = load_universe(tickers, fetch_start, END, verbose=False)
print("loaded", len(raw))
spy = fetch_daily("SPY", "2015-01-01", END)
spy_ok = (spy["close"] > spy["close"].rolling(200, min_periods=200).mean()).shift(1).fillna(False)

n_events = n_reclaimed = 0
rets5, rets10 = [], []
per_year = {}
groups = {"above_100_200": {"n":0,"rec":0,"r5":[]}, "below_either": {"n":0,"rec":0,"r5":[]}}
for t, df0 in raw.items():
    df = add_indicators(df0)
    elig = eligibility_mask(df, spy_ok) & (df.index >= pd.Timestamp(START))
    low, high, close, sma20 = df["low"].values, df["high"].values, df["close"].values, df["sma20"].values
    touch = (df["low"] <= df["sma20"] * 1.01) & elig & (df["close"] > df["sma50"] * 0.99)
    above = df["low"] > df["sma20"]
    # cluster: touch day preceded by >=3 days of lows above SMA20
    prior_above = above.shift(1).rolling(3).min().fillna(0).astype(bool)
    starts = np.flatnonzero((touch & prior_above).values)
    n = len(df)
    sma100 = df["sma100"].values
    sma200v = df["sma200"].values
    for i in starts:
        n_events += 1
        y = df.index[i].year
        per_year[y] = per_year.get(y, 0) + 1
        gkey = "above_100_200" if (close[i] > sma100[i] and close[i] > sma200v[i]) else "below_either"
        groups[gkey]["n"] += 1
        # cluster high = max high over touch cluster (walk forward while touching)
        j = i
        while j + 1 < n and low[j + 1] <= sma20[j + 1] * 1.01:
            j += 1
        trigger = max(high[i:j + 1])
        # reclaim within 5 days after cluster end
        rec = None
        for k in range(j + 1, min(n, j + 6)):
            if high[k] >= trigger * 1.001:
                rec = k; break
        if rec is None:
            continue
        n_reclaimed += 1
        groups[gkey]["rec"] += 1
        entry = max(df["open"].values[rec], trigger * 1.001)
        for hzn, bucket in ((5, rets5), (10, rets10)):
            m = min(n - 1, rec + hzn - 1)
            bucket.append(close[m] / entry - 1)
        groups[gkey]["r5"].append(close[min(n - 1, rec + 4)] / entry - 1)

r5, r10 = np.array(rets5), np.array(rets10)
out = {
  "events": n_events, "reclaimed_within_5d": n_reclaimed,
  "reclaim_rate_pct": round(100 * n_reclaimed / max(1, n_events), 1),
  "fwd5_avg_pct": round(100 * r5.mean(), 3), "fwd5_win_pct": round(100 * (r5 > 0).mean(), 1),
  "fwd10_avg_pct": round(100 * r10.mean(), 3), "fwd10_win_pct": round(100 * (r10 > 0).mean(), 1),
  "events_per_year": {str(k): v for k, v in sorted(per_year.items())},
  "by_structure": {k: {"events": v["n"],
                        "reclaim_pct": round(100*v["rec"]/max(1,v["n"]),1),
                        "fwd5_avg_pct": round(100*float(np.mean(v["r5"])),3) if v["r5"] else None}
                   for k, v in groups.items()},
}
print(json.dumps(out, indent=1))
json.dump(out, open("results/retest_count.json", "w"), indent=1)

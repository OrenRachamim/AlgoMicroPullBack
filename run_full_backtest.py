#!/usr/bin/env python3
"""Comprehensive backtest of the official strategy (final: fresh breakout +
deep micro-pullback) — one run that produces everything:

  1. Full-period results (2018->today, broad universe) + SPY benchmark
  2. Yearly breakdown and sub-period splits
  3. Sensitivity: costs, position count, RSI threshold, hold days,
     fresh-window length, regime filter, earnings filter
  4. Trade distribution: percentiles, worst trades, streaks, exposure
  5. Monte-Carlo (block bootstrap of daily returns): return/DD distributions

Output: results/full_backtest.json + printed summary.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from micro_pullback.backtest import BacktestConfig, run_backtest
from micro_pullback.data import fetch_daily, load_universe
from micro_pullback.earnings import load_earnings_risk
from micro_pullback.indicators import add_indicators
from micro_pullback.universe import get_broad_universe
from micro_pullback.variants import VARIANTS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
START, END = "2018-01-02", "2026-10-19"
WARMUP_DAYS = 700
CAPITAL = 100_000.0


def spy_regime(end: str) -> pd.Series:
    spy = fetch_daily("SPY", "2015-01-01", end)
    return spy["close"] > spy["close"].rolling(200, min_periods=200).mean(), spy


def summarize(trades, equity, label, spy=None):
    from micro_pullback.backtest import compute_stats
    stats = compute_stats(trades, equity, CAPITAL)
    stats["label"] = label
    if spy is not None and not equity.empty:
        w = spy["close"].reindex(equity.index).ffill().dropna()
        stats["benchmark_spy_pct"] = round(100 * (w.iloc[-1] / w.iloc[0] - 1), 2)
    return stats


def main() -> None:
    tickers = get_broad_universe(limit=600)
    fetch_start = (dt.date.fromisoformat(START) - dt.timedelta(days=WARMUP_DAYS)).isoformat()
    print(f"Loading {len(tickers)} tickers {fetch_start} -> {END} ...")
    raw = load_universe(tickers, fetch_start, END)
    print(f"Loaded {len(raw)} tickers")
    data = {t: add_indicators(df) for t, df in raw.items()}

    regime, spy = spy_regime(END)
    calendar = pd.DatetimeIndex(sorted(set().union(*[df.index for df in data.values()])))
    print("Loading earnings (SEC EDGAR) ...")
    earnings = load_earnings_risk(list(data.keys()), calendar,
                                  buffer_before=0, buffer_after=0)

    base_params = VARIANTS["final"]
    base_cfg = BacktestConfig(initial_capital=CAPITAL, max_positions=5, rank_by="momentum")
    out: dict = {"start": START, "end": END, "n_tickers": len(data), "runs": []}

    def run(label, params=None, cfg=None, use_regime=True, use_earn=True,
            start=START, end=END, keep=False):
        p = params or base_params
        c = cfg or base_cfg
        tr, eq, _ = run_backtest(data, p, c, start=start, end=end,
                                 regime=regime if use_regime else None,
                                 earnings=earnings if use_earn else None)
        s = summarize(tr, eq, label, spy)
        out["runs"].append(s)
        print(f"  done: {label} -> {s.get('total_return_pct')}% "
              f"(sharpe {s.get('sharpe')}, dd {s.get('max_drawdown_pct')}%)")
        return (tr, eq) if keep else None

    # ---- 1. base run (keep trades/equity for deep analysis) ----
    print("\n[1] base full-period run")
    trades, equity = run("base_full_period", keep=True)

    # ---- 2. yearly breakdown + splits ----
    eq = equity
    yearly = {}
    for y, grp in eq.groupby(eq.index.year):
        prev = eq[eq.index.year < y]
        start_val = prev.iloc[-1] if len(prev) else CAPITAL
        yearly[int(y)] = round(100 * (grp.iloc[-1] / start_val - 1), 2)
    out["yearly_pct"] = yearly
    # SPY yearly for comparison
    spy_w = spy["close"].reindex(eq.index).ffill()
    out["spy_yearly_pct"] = {int(y): round(100 * (g.iloc[-1] / (spy_w[spy_w.index.year < y].iloc[-1] if len(spy_w[spy_w.index.year < y]) else g.iloc[0]) - 1), 2)
                             for y, g in spy_w.groupby(spy_w.index.year)}
    print("\n[2] sub-period splits")
    run("split_2018_2020", start="2018-01-02", end="2020-12-31")
    run("split_2021_2023", start="2021-01-04", end="2023-12-29")
    run("split_2024_2026", start="2024-01-02", end=END)

    # ---- 3. sensitivity ----
    print("\n[3] sensitivity")
    for bps in (0.0, 0.001, 0.002):
        run(f"cost_{int(bps*10000)}bps_per_side",
            cfg=replace(base_cfg, commission_pct=bps / 2, slippage_pct=bps / 2))
    for mp in (3, 8):
        run(f"positions_{mp}", cfg=replace(base_cfg, max_positions=mp))
    for r in (8.0, 12.0):
        run(f"rsi_entry_{int(r)}", params=replace(base_params, rsi_entry_max=r))
    for h in (4, 6):
        run(f"hold_{h}d", params=replace(base_params, max_hold_days=h))
    for w in (40, 80):
        run(f"window_{w}d", params=replace(base_params, fresh_cross_lookback=w))
    run("no_regime_filter", use_regime=False)
    run("no_earnings_filter", use_earn=False)
    run("rank_rsi", cfg=replace(base_cfg, rank_by="rsi"))

    # ---- 4. trade distribution ----
    print("\n[4] trade distribution")
    rets = np.array([t.ret_pct for t in trades]) * 100
    seq = [t.ret_pct for t in sorted(trades, key=lambda x: x.exit_date)]
    streak, worst_streak = 0, 0
    for r in seq:
        streak = streak + 1 if r <= 0 else 0
        worst_streak = max(worst_streak, streak)
    daily = equity.pct_change().dropna()
    held_days = sum(t.bars_held for t in trades)
    out["trade_stats"] = {
        "n_trades": len(trades),
        "pct_per_trade": {p: round(float(np.percentile(rets, p)), 2)
                          for p in (5, 25, 50, 75, 95)},
        "worst_trade_pct": round(float(rets.min()), 2),
        "best_trade_pct": round(float(rets.max()), 2),
        "max_consecutive_losses": worst_streak,
        "avg_capital_deployed_pct": round(100 * held_days / (len(equity) * 5), 1),
        "days_in_market_pct": round(100 * float((daily != 0).mean()), 1),
        "ann_vol_pct": round(100 * float(daily.std() * np.sqrt(252)), 1),
        "worst_10_trades": [
            {"ticker": t.ticker, "entry": str(t.entry_date.date()),
             "ret_pct": round(100 * t.ret_pct, 2), "reason": t.exit_reason}
            for t in sorted(trades, key=lambda x: x.ret_pct)[:10]
        ],
    }

    # ---- 5. Monte Carlo: 21-day block bootstrap of daily equity returns ----
    print("[5] monte carlo")
    rng = np.random.default_rng(7)
    dr = daily.values
    n = len(dr)
    horizon = 252 * 3
    finals, dds = [], []
    for _ in range(2000):
        blocks = []
        while sum(len(b) for b in blocks) < horizon:
            s = rng.integers(0, max(1, n - 21))
            blocks.append(dr[s:s + 21])
        path = np.concatenate(blocks)[:horizon]
        eq_path = np.cumprod(1 + path)
        finals.append(eq_path[-1] - 1)
        peak = np.maximum.accumulate(eq_path)
        dds.append((eq_path / peak - 1).min())
    out["monte_carlo_3y"] = {
        "iterations": 2000,
        "return_pct": {p: round(100 * float(np.percentile(finals, p)), 1)
                       for p in (5, 25, 50, 75, 95)},
        "max_dd_pct": {p: round(100 * float(np.percentile(dds, p)), 1)
                       for p in (5, 50, 95)},
        "p_loss_pct": round(100 * float(np.mean(np.array(finals) < 0)), 1),
    }

    out["equity_curve"] = {str(k.date()): round(float(v), 2) for k, v in equity.items()}
    path = os.path.join(RESULTS_DIR, "full_backtest.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()

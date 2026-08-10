#!/usr/bin/env python3
"""Run all (or selected) strategy variants over a universe and print a
comparison table. Used for the iteration loop; results append to
results/sweep_<universe>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

from micro_pullback.backtest import BacktestConfig, run_backtest
from micro_pullback.data import load_universe
from micro_pullback.indicators import add_indicators
from micro_pullback.universe import EXTENDED_UNIVERSE, SMALL_UNIVERSE
from micro_pullback.variants import VARIANTS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLS = ["variant", "n_trades", "total_return_pct", "cagr_pct", "max_drawdown_pct",
        "sharpe", "win_rate_pct", "avg_trade_pct", "profit_factor", "avg_hold_days"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["small", "extended"], default="small")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-08-01")
    ap.add_argument("--variants", default=None, help="comma-separated subset, default all")
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--regime", choices=["none", "spy200", "spy100", "spy200+50"], default="none",
                    help="market regime filter: allow entries only when SPY > SMA-N "
                         "(spy200+50: above both SMA200 and SMA50)")
    ap.add_argument("--rank", choices=["rsi", "momentum"], default="rsi")
    args = ap.parse_args()

    tickers = SMALL_UNIVERSE if args.universe == "small" else EXTENDED_UNIVERSE
    fetch_start = (dt.date.fromisoformat(args.start) - dt.timedelta(days=400)).isoformat()
    print(f"Loading {len(tickers)} tickers {fetch_start} -> {args.end} ...")
    raw = load_universe(tickers, fetch_start, args.end)
    print(f"Loaded {len(raw)} tickers\n")
    data = {t: add_indicators(df) for t, df in raw.items()}

    regime = None
    if args.regime != "none":
        from micro_pullback.data import fetch_daily
        spy = fetch_daily("SPY", (dt.date.fromisoformat(args.start) - dt.timedelta(days=500)).isoformat(), args.end)
        sma = lambda w: spy["close"].rolling(w, min_periods=w).mean()
        if args.regime == "spy200+50":
            regime = (spy["close"] > sma(200)) & (spy["close"] > sma(50))
        else:
            win = 200 if args.regime == "spy200" else 100
            regime = spy["close"] > sma(win)

    names = args.variants.split(",") if args.variants else sorted(VARIANTS.keys())
    rows = []
    for name in names:
        params = VARIANTS[name]
        cfg = BacktestConfig(max_positions=args.max_positions, rank_by=args.rank)
        _, _, stats = run_backtest(data, params, cfg, start=args.start, end=args.end,
                                   regime=regime)
        row = {"variant": f"{name}:{params.name}"}
        row.update({k: stats.get(k) for k in COLS[1:]})
        row["exit_reasons"] = stats.get("exit_reasons", {})
        rows.append(row)
        print(f"done {name}")

    # table
    print(f"\n=== Sweep | universe={args.universe} | {args.start}..{args.end} "
          f"| max_pos={args.max_positions} | regime={args.regime} | rank={args.rank} ===")
    header = " | ".join(f"{c:>18}" for c in COLS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '')):>18}" for c in COLS))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"sweep_{args.universe}.json")
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.append({"start": args.start, "end": args.end,
                     "max_positions": args.max_positions,
                     "regime": args.regime, "rank": args.rank, "rows": rows})
    with open(path, "w") as f:
        json.dump(existing, f, indent=1)
    print(f"\nAppended -> {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI runner for the micro-pullback backtest.

Examples:
    python run_backtest.py --universe small --start 2023-01-01 --end 2024-12-31
    python run_backtest.py --universe extended --start 2021-01-01 --end 2025-12-31 --variant v3
"""

from __future__ import annotations

import argparse
import json
import os

from micro_pullback.backtest import BacktestConfig, run_backtest
from micro_pullback.data import load_universe
from micro_pullback.indicators import add_indicators
from micro_pullback.strategy import StrategyParams
from micro_pullback.universe import EXTENDED_UNIVERSE, FULL_UNIVERSE, NASDAQ_UNIVERSE, SMALL_UNIVERSE
from micro_pullback.variants import VARIANTS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main() -> None:
    ap = argparse.ArgumentParser(description="Micro-pullback backtest")
    ap.add_argument("--universe", choices=["small", "extended", "nasdaq", "full"], default="small")
    ap.add_argument("--start", default="2022-01-01", help="backtest start (data warm-up added automatically)")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--variant", default="v1", choices=sorted(VARIANTS.keys()))
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--regime", choices=["none", "spy200", "spy100", "spy200+50"], default="none")
    ap.add_argument("--rank", choices=["rsi", "momentum"], default="rsi")
    ap.add_argument("--no-earnings-filter", action="store_true",
                    help="disable SEC-EDGAR earnings avoidance")
    ap.add_argument("--earn-before", type=int, default=0,
                    help="extra risk days before the earnings filing date")
    ap.add_argument("--earn-after", type=int, default=0,
                    help="extra risk days after the earnings filing date")
    ap.add_argument("--save", default=None, help="save stats+trades JSON under results/<name>.json")
    args = ap.parse_args()

    tickers = {"small": SMALL_UNIVERSE, "extended": EXTENDED_UNIVERSE,
               "nasdaq": NASDAQ_UNIVERSE, "full": FULL_UNIVERSE}[args.universe]
    # warm-up: fetch ~1 extra year of data before the backtest window for indicators
    import datetime as dt
    fetch_start = (dt.date.fromisoformat(args.start) - dt.timedelta(days=400)).isoformat()

    print(f"Loading {len(tickers)} tickers {fetch_start} -> {args.end} ...")
    raw = load_universe(tickers, fetch_start, args.end)
    print(f"Loaded {len(raw)} tickers")

    data = {t: add_indicators(df) for t, df in raw.items()}
    params = VARIANTS[args.variant]
    cfg = BacktestConfig(max_positions=args.max_positions, rank_by=args.rank)

    regime = None
    spy = None
    from micro_pullback.data import fetch_daily
    spy = fetch_daily("SPY", (dt.date.fromisoformat(args.start) - dt.timedelta(days=500)).isoformat(), args.end)
    if args.regime != "none":
        sma = lambda w: spy["close"].rolling(w, min_periods=w).mean()
        if args.regime == "spy200+50":
            regime = (spy["close"] > sma(200)) & (spy["close"] > sma(50))
        else:
            win = 200 if args.regime == "spy200" else 100
            regime = spy["close"] > sma(win)

    earnings = None
    if not args.no_earnings_filter:
        from micro_pullback.earnings import load_earnings_risk
        import pandas as pd
        calendar = pd.DatetimeIndex(sorted(set().union(*[df.index for df in data.values()])))
        print("Loading earnings dates from SEC EDGAR ...")
        earnings = load_earnings_risk(list(data.keys()), calendar,
                                      buffer_before=args.earn_before,
                                      buffer_after=args.earn_after)

    trades, equity, stats = run_backtest(data, params, cfg, start=args.start, end=args.end,
                                         regime=regime, earnings=earnings)

    # SPY buy-and-hold benchmark over the same window
    if spy is not None and not equity.empty:
        spy_win = spy["close"].reindex(equity.index).ffill().dropna()
        if len(spy_win) > 1:
            stats["benchmark_spy_pct"] = round(100 * (spy_win.iloc[-1] / spy_win.iloc[0] - 1.0), 2)

    print(f"\n=== Variant {args.variant} ({params.name}) | universe={args.universe} "
          f"| {args.start}..{args.end} | regime={args.regime} | rank={args.rank} "
          f"| max_pos={args.max_positions} ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = {
            "variant": args.variant,
            "params": params.to_dict(),
            "universe": args.universe,
            "start": args.start,
            "end": args.end,
            "regime": args.regime,
            "rank": args.rank,
            "max_positions": args.max_positions,
            "stats": stats,
            "equity_curve": {str(k.date()): round(float(v), 2) for k, v in equity.items()},
            "trades": [
                {
                    "ticker": t.ticker,
                    "entry_date": str(t.entry_date.date()),
                    "exit_date": str(t.exit_date.date()),
                    "entry": round(t.entry_price, 4),
                    "exit": round(t.exit_price, 4),
                    "ret_pct": round(100 * t.ret_pct, 3),
                    "bars": t.bars_held,
                    "reason": t.exit_reason,
                }
                for t in trades
            ],
        }
        path = os.path.join(RESULTS_DIR, f"{args.save}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()

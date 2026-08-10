"""Strategy variants tried across backtest iterations.

Each iteration of the research loop adds/edits a variant here; results are
logged under results/ and summarized in results/ITERATIONS.md.
"""

from dataclasses import replace

from .strategy import StrategyParams

VARIANTS: dict[str, StrategyParams] = {
    # v1 — baseline: classic micro pullback, breakout re-entry, 1.5 ATR stop,
    # strength exit on RSI3>75, 5-day time stop.
    "v1": StrategyParams(name="baseline"),

    # v2 — wider stop: baseline losses were dominated by a tight 1.5 ATR stop
    # (avg loss > avg win). Give the bounce room: 2.5 ATR stop, earlier
    # strength exit (RSI3>70).
    "v2": StrategyParams(name="wide_stop", stop_atr_mult=2.5, exit_rsi_min=70.0),

    # v3 — Connors-style: no protective stop at all, exit purely on strength
    # (RSI3>65) or 7-day time stop. Mean-reversion systems often degrade with
    # stops; time stop caps the tail.
    "v3": StrategyParams(name="no_stop", stop_atr_mult=0.0, exit_rsi_min=65.0,
                         max_hold_days=7),

    # v4 — enter at next open (no breakout confirmation): catch the bounce a
    # day earlier; deeper oversold requirement compensates (RSI2<15).
    "v4": StrategyParams(name="open_entry_deep", entry_mode="open",
                         rsi_entry_col="rsi2", rsi_entry_max=15.0,
                         stop_atr_mult=2.5, exit_rsi_col="rsi2", exit_rsi_min=80.0),

    # v5 — quick scalp: 1 ATR profit target, 1.5 ATR stop, 4-day time stop.
    "v5": StrategyParams(name="atr_scalp", target_atr_mult=1.0,
                         stop_atr_mult=1.5, exit_rsi_min=0.0, max_hold_days=4),

    # v6 — deeper pullback, wide stop: RSI3<20, 2-4 down days window kept,
    # 2.5 ATR stop, exit RSI3>70.
    "v6": StrategyParams(name="deep_dip", rsi_entry_max=20.0,
                         stop_atr_mult=2.5, exit_rsi_min=70.0),

    # v7 — more setups: relax trend filter (no MACD gate, ret63>=0),
    # 1 down day enough, no-stop exits like v3.
    "v7": StrategyParams(name="loose_trend", macd_mode="none", min_ret63=0.0,
                         min_down_days=1, stop_atr_mult=0.0,
                         exit_rsi_min=65.0, max_hold_days=7),

    # --- iteration 3: combine winners (v4 open-entry + v6 deep dip) ---

    # v8 — v4 with an even deeper dip (RSI2<10), wide 3 ATR stop, let winners
    # run: exit RSI2>85, 6-day time stop.
    "v8": StrategyParams(name="open_deeper_run", entry_mode="open",
                         rsi_entry_col="rsi2", rsi_entry_max=10.0,
                         stop_atr_mult=3.0, exit_rsi_col="rsi2",
                         exit_rsi_min=85.0, max_hold_days=6),

    # v9 — v4 plus an ATR profit target (2 ATR) instead of pure strength exit.
    "v9": StrategyParams(name="open_atr_target", entry_mode="open",
                         rsi_entry_col="rsi2", rsi_entry_max=15.0,
                         stop_atr_mult=2.5, target_atr_mult=2.0,
                         exit_rsi_col="rsi2", exit_rsi_min=90.0, max_hold_days=6),

    # v10 — v4 with 1 down day required only (more setups, RSI2 gate does the work).
    "v10": StrategyParams(name="open_more_setups", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=15.0,
                          min_down_days=1, stop_atr_mult=2.5,
                          exit_rsi_col="rsi2", exit_rsi_min=80.0),

    # v11 — v4 restricted to tight micro pullbacks (<=6% off the 20d high).
    "v11": StrategyParams(name="open_tight_dip", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=15.0,
                          max_pullback_pct=0.06, stop_atr_mult=2.5,
                          exit_rsi_col="rsi2", exit_rsi_min=80.0),

    # v12 — v6 deep dip with open entry: RSI3<20, exit RSI3>70.
    "v12": StrategyParams(name="open_deep_rsi3", entry_mode="open",
                          rsi_entry_col="rsi3", rsi_entry_max=20.0,
                          stop_atr_mult=2.5, exit_rsi_min=70.0),

    # --- iteration 4: merge v10 (setup breadth) with v8 (depth + let-run) ---

    # v13 — 1 down day, RSI2<10, 3 ATR stop, exit RSI2>85, 6-day hold.
    "v13": StrategyParams(name="broad_deep_run", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=3.0,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v14 — v10 exits pushed to v8 style (RSI2>85, 6 days).
    "v14": StrategyParams(name="broad_run", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=15.0,
                          min_down_days=1, stop_atr_mult=2.5,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v15 — v13 without a protective stop (pure mean reversion + time stop).
    "v15": StrategyParams(name="broad_deep_nostop", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=0.0,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v16 — v13 with slightly relaxed trend gate to widen the funnel.
    "v16": StrategyParams(name="broad_deep_loose", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, macd_mode="none", min_ret63=0.02,
                          stop_atr_mult=3.0, exit_rsi_col="rsi2",
                          exit_rsi_min=85.0, max_hold_days=6),

    # --- iteration 7: crash-robustness fixes after 2018-2021 OOS failure ---
    # OOS diagnosis: stops averaged -6.4% when hit, time exits negative,
    # losses clustered at crash onsets (Feb'18, Q4'18, Feb'20) while SPY was
    # still above SMA200. Fixes: drop the hard stop (v15 survived OOS better),
    # add a per-stock volatility guard, pair with the dual SPY regime filter.

    # v17 — v15 + volatility guard (ATR14/close <= 4%).
    "v17": StrategyParams(name="calm_deep_nostop", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=0.0, max_atr_pct=0.04,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v18 — v17 with a shorter leash: 4-day time stop.
    "v18": StrategyParams(name="calm_deep_short", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=0.0, max_atr_pct=0.04,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=4),

    # v19 — v17 with a disaster stop only (very wide, 4 ATR) for gap risk.
    "v19": StrategyParams(name="calm_deep_disaster", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=4.0, max_atr_pct=0.04,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # --- iteration 8: bounce confirmation instead of knife-catching ---

    # v20 — deep dip + breakout confirmation entry (prev high + 0.1%),
    # no hard stop, strength/time exits.
    "v20": StrategyParams(name="confirm_deep", entry_mode="breakout",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=0.0, max_atr_pct=0.04,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v21 — v17 with a tighter volatility guard (ATR14/close <= 3%).
    "v21": StrategyParams(name="calmer_deep", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=0.0, max_atr_pct=0.03,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v22 — v17 with pullback capped at 7% off the 20d high.
    "v22": StrategyParams(name="shallow_only", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # --- iteration 9: close-based stop to cap the event-gap tail ---

    # v23 — v22 + exit at close when down 5% from entry.
    "v23": StrategyParams(name="shallow_closestop5", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07, stop_close_pct=0.05,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v24 — looser close stop (7%).
    "v24": StrategyParams(name="shallow_closestop7", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07, stop_close_pct=0.07,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v25 — v23 + bank profits earlier (RSI2>75).
    "v25": StrategyParams(name="shallow_bank_early", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07, stop_close_pct=0.05,
                          exit_rsi_col="rsi2", exit_rsi_min=75.0, max_hold_days=6),

    # --- iteration 10: entry quality — trade only true momentum leaders ---

    # v26 — v22 + strong 3-month momentum (>=15%).
    "v26": StrategyParams(name="leaders_15", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, min_ret63=0.15,
                          stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v27 — v22 + very strong momentum (>=25%).
    "v27": StrategyParams(name="leaders_25", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, min_ret63=0.25,
                          stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # v29 — middle ground: momentum >=20%.
    "v29": StrategyParams(name="leaders_20", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, min_ret63=0.20,
                          stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),

    # --- iteration 11: give the bounce more time (time exits were ~50% of trades) ---

    # v30 — v29 with 8-day hold, take strength profits a bit earlier (RSI2>80).
    "v30": StrategyParams(name="leaders_20_hold8", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, min_ret63=0.20,
                          stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07,
                          exit_rsi_col="rsi2", exit_rsi_min=80.0, max_hold_days=8),

    # v31 — v29 with 10-day hold.
    "v31": StrategyParams(name="leaders_20_hold10", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, min_ret63=0.20,
                          stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=10),

    # v28 — v22 + near 52-week high (>=90%): dip inside a fresh-high base.
    "v28": StrategyParams(name="near_high", entry_mode="open",
                          rsi_entry_col="rsi2", rsi_entry_max=10.0,
                          min_down_days=1, min_pct_of_52w_high=0.90,
                          stop_atr_mult=0.0, max_atr_pct=0.04,
                          max_pullback_pct=0.07,
                          exit_rsi_col="rsi2", exit_rsi_min=85.0, max_hold_days=6),
}

# The winning configuration after 11 iterations (see results/ITERATIONS.md).
# Run with: --regime spy200 --rank momentum --max-positions 5
VARIANTS["final"] = StrategyParams(name="final_leaders_20", entry_mode="open",
                                   rsi_entry_col="rsi2", rsi_entry_max=10.0,
                                   min_down_days=1, min_ret63=0.20,
                                   stop_atr_mult=0.0, max_atr_pct=0.04,
                                   max_pullback_pct=0.07,
                                   exit_rsi_col="rsi2", exit_rsi_min=85.0,
                                   max_hold_days=6)

# ---------------------------------------------------------------------------
# Exit-strategy research (round 2): entry rules frozen to `final`, exits vary.
# e0 is the reference. All e-variants run with the earnings filter active.
# ---------------------------------------------------------------------------
_E = VARIANTS["final"]

EXIT_VARIANTS: dict[str, StrategyParams] = {
    # reference: strength exit RSI2>85 or 6-day time stop
    "e0": replace(_E, name="ref_rsi85_t6"),
    # pure trailing stops (no RSI exit), longer leash
    "e1": replace(_E, name="trail_atr2", exit_rsi_min=0.0, trail_atr_mult=2.0, max_hold_days=10),
    "e2": replace(_E, name="trail_atr3", exit_rsi_min=0.0, trail_atr_mult=3.0, max_hold_days=10),
    "e3": replace(_E, name="trail_pct5", exit_rsi_min=0.0, trail_pct=0.05, max_hold_days=10),
    # hybrid: strength exit + trailing safety net
    "e4": replace(_E, name="rsi85_trail_atr25", trail_atr_mult=2.5),
    # RSI threshold sweep
    "e5": replace(_E, name="rsi80_t6", exit_rsi_min=80.0),
    "e6": replace(_E, name="rsi90_t6", exit_rsi_min=90.0),
    # time-stop sweep
    "e7": replace(_E, name="rsi85_t5", max_hold_days=5),
    "e8": replace(_E, name="rsi85_t7", max_hold_days=7),
    # moving-average exit
    "e9": replace(_E, name="below_ema10", exit_rsi_min=0.0, exit_below_ema10=True, max_hold_days=10),
    # breakeven protection
    "e10": replace(_E, name="rsi85_breakeven1atr", breakeven_atr=1.0),
    # profit target + trail combo
    "e11": replace(_E, name="target2atr_trail25", exit_rsi_min=0.0, target_atr_mult=2.0,
                   trail_atr_mult=2.5, max_hold_days=8),
    # strength exit + % trail
    "e12": replace(_E, name="rsi85_trailpct7_t8", trail_pct=0.07, max_hold_days=8),
    # slower RSI for the strength exit
    "e13": replace(_E, name="rsi3_80_t6", exit_rsi_col="rsi3", exit_rsi_min=80.0),
    # fixed profit target
    "e14": replace(_E, name="target5pct_t6", target_pct=0.05),

    # --- exit round 2: zoom into short time stops x RSI threshold ---
    "e15": replace(_E, name="rsi85_t4", max_hold_days=4),
    "e16": replace(_E, name="rsi80_t5", exit_rsi_min=80.0, max_hold_days=5),
    "e17": replace(_E, name="rsi85_t3", max_hold_days=3),
    "e18": replace(_E, name="rsi80_t4", exit_rsi_min=80.0, max_hold_days=4),
    "e19": replace(_E, name="rsi75_t5", exit_rsi_min=75.0, max_hold_days=5),
    "e20": replace(_E, name="rsi90_t5", exit_rsi_min=90.0, max_hold_days=5),
}

VARIANTS.update(EXIT_VARIANTS)

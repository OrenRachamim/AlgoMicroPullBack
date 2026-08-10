"""Portfolio-level daily backtester for the micro-pullback strategy.

Execution model (no lookahead):
- Setups are computed on bar t (close data); entries happen on bar t+1.
- Protective stop and profit target are checked intraday (gap-aware: if the
  open is already through the level, the fill is at the open).
- Strength exit (short RSI overbought) and the time stop execute at that
  bar's close (market-on-close order).
- Exits are processed before entries each day, freeing position slots.
- Fills pay commission + slippage on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import (
    StrategyParams,
    compute_setup,
    entry_price_for,
    initial_stop_target,
)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    max_positions: int = 8
    commission_pct: float = 0.0005   # 5 bps per side
    slippage_pct: float = 0.0005     # 5 bps per side
    rank_by: str = "rsi"             # "rsi": deepest dip first; "momentum": strongest ret63 first


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    ret_pct: float
    bars_held: int
    exit_reason: str


class _Pos:
    __slots__ = ("ticker", "entry_date", "entry_price", "shares", "stop", "target",
                 "bars_held", "peak_high", "peak_close", "entry_atr")

    def __init__(self, ticker, entry_date, entry_price, shares, stop, target, entry_atr):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.stop = stop
        self.target = target
        self.bars_held = 0
        self.peak_high = entry_price
        self.peak_close = entry_price
        self.entry_atr = entry_atr


def run_backtest(data: dict[str, pd.DataFrame], params: StrategyParams,
                 config: BacktestConfig | None = None,
                 start: str | None = None, end: str | None = None,
                 regime: pd.Series | None = None,
                 earnings: dict[str, set] | None = None):
    """Run the strategy over a {ticker: indicator-enriched DataFrame} universe.

    `regime`: optional boolean Series indexed by date (e.g. SPY > SMA200).
    New entries are blocked on days where the regime is False (evaluated on
    the previous bar's value — no lookahead). Exits are always allowed.

    `earnings`: optional {ticker: set of earnings-risk trading days}. When
    params.avoid_earnings is on, entries are skipped on a ticker's risk days
    and open positions are closed at the last close before a risk day.
    (In live trading, use a forward-looking earnings calendar the same way.)

    Returns (trades: list[Trade], equity_curve: pd.Series, stats: dict).
    """
    cfg = config or BacktestConfig()
    if regime is not None:
        # use yesterday's regime state for today's entries
        regime = regime.shift(1).fillna(False)

    # Precompute setups and align calendars.
    setups: dict[str, pd.Series] = {}
    frames: dict[str, pd.DataFrame] = {}
    all_dates = set()
    for t, df in data.items():
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        if len(df) < 130:
            continue
        frames[t] = df
        setups[t] = compute_setup(df, params)
        all_dates.update(df.index)
    calendar = pd.DatetimeIndex(sorted(all_dates))

    cash = cfg.initial_capital
    positions: dict[str, _Pos] = {}
    trades: list[Trade] = []
    equity_hist: list[tuple[pd.Timestamp, float]] = []
    entry_cost = 1.0 + cfg.commission_pct + cfg.slippage_pct
    exit_cost = 1.0 - cfg.commission_pct - cfg.slippage_pct

    # Fast lookups: integer positions per ticker calendar.
    locs = {t: {d: i for i, d in enumerate(df.index)} for t, df in frames.items()}

    for day in calendar:
        # ---------- exits ----------
        for t in list(positions.keys()):
            pos = positions[t]
            df = frames[t]
            i = locs[t].get(day)
            if i is None:
                continue
            row = df.iloc[i]
            pos.bars_held += 1
            exit_price, reason = None, None

            o, h, l, c = row["open"], row["high"], row["low"], row["close"]

            # effective protective level: initial stop, trailing stops and
            # breakeven all collapse into "highest active stop". Trails are
            # ratcheted from the *previous* bar's peaks (no same-bar lookahead).
            eff_stop = pos.stop if np.isfinite(pos.stop) else -np.inf
            if params.trail_atr_mult > 0:
                atr_now = row["atr14"] if np.isfinite(row["atr14"]) else pos.entry_atr
                eff_stop = max(eff_stop, pos.peak_high - params.trail_atr_mult * atr_now)
            if params.trail_pct > 0:
                eff_stop = max(eff_stop, pos.peak_close * (1.0 - params.trail_pct))
            if params.breakeven_atr > 0 and np.isfinite(pos.entry_atr):
                if pos.peak_high >= pos.entry_price + params.breakeven_atr * pos.entry_atr:
                    eff_stop = max(eff_stop, pos.entry_price)

            # gap-aware stop / target, stop takes priority on the same bar
            if np.isfinite(eff_stop) and eff_stop > -np.inf and (o <= eff_stop or l <= eff_stop):
                exit_price = o if o <= eff_stop else eff_stop
                reason = "stop" if eff_stop == pos.stop else "trail_stop"
            elif np.isfinite(pos.target) and (o >= pos.target or h >= pos.target):
                exit_price = o if o >= pos.target else pos.target
                reason = "target"
            elif params.stop_close_pct > 0 and c <= pos.entry_price * (1.0 - params.stop_close_pct):
                exit_price, reason = c, "close_stop"
            elif params.avoid_earnings and earnings is not None and i + 1 < len(df) \
                    and df.index[i + 1] in earnings.get(t, ()):
                exit_price, reason = c, "earnings"
            elif params.exit_rsi_min > 0 and row[params.exit_rsi_col] >= params.exit_rsi_min:
                exit_price, reason = c, "rsi_strength"
            elif params.exit_below_ema10 and np.isfinite(row["ema10"]) and c < row["ema10"]:
                exit_price, reason = c, "below_ema10"
            elif pos.bars_held >= params.max_hold_days:
                exit_price, reason = c, "time"

            # update peaks for the next bar's trail calculations
            pos.peak_high = max(pos.peak_high, h)
            pos.peak_close = max(pos.peak_close, c)

            if exit_price is not None:
                proceeds = pos.shares * exit_price * exit_cost
                cash += proceeds
                cost_basis = pos.shares * pos.entry_price * entry_cost
                pnl = proceeds - cost_basis
                trades.append(Trade(
                    ticker=t, entry_date=pos.entry_date, exit_date=day,
                    entry_price=pos.entry_price, exit_price=float(exit_price),
                    shares=pos.shares, pnl=float(pnl),
                    ret_pct=float(proceeds / cost_basis - 1.0),
                    bars_held=pos.bars_held, exit_reason=reason,
                ))
                del positions[t]

        # ---------- entries ----------
        slots = cfg.max_positions - len(positions)
        if regime is not None and day in regime.index and not bool(regime.loc[day]):
            slots = 0
        if slots > 0:
            candidates = []
            for t, df in frames.items():
                if t in positions:
                    continue
                i = locs[t].get(day)
                if i is None or i == 0:
                    continue
                if params.avoid_earnings and earnings is not None \
                        and day in earnings.get(t, ()):
                    continue
                prev_i = i - 1
                if not setups[t].iloc[prev_i]:
                    continue
                prev = df.iloc[prev_i]
                row = df.iloc[i]
                fill = entry_price_for(row, float(prev["high"]), params)
                if fill is None:
                    continue
                rank_key = prev[params.rsi_entry_col] if cfg.rank_by == "rsi" else -prev["ret63"]
                candidates.append((rank_key, t, fill, float(prev["atr14"])))
            candidates.sort(key=lambda x: x[0])

            # equal-weight sizing on current equity
            equity_now = cash + sum(
                pos.shares * frames[t2].iloc[locs[t2][day]]["close"]
                for t2, pos in positions.items() if day in locs[t2]
            )
            alloc = equity_now / cfg.max_positions

            for _, t, fill, atr_value in candidates[:slots]:
                cost_per_share = fill * entry_cost
                budget = min(alloc, cash)
                shares = np.floor(budget / cost_per_share)
                if shares < 1:
                    continue
                cash -= shares * cost_per_share
                stop, target = initial_stop_target(fill, atr_value, params)
                pos = _Pos(t, day, fill, float(shares), stop, target, atr_value)
                # seed peaks with the entry day's own extremes
                row = frames[t].iloc[locs[t][day]]
                pos.peak_high = max(fill, float(row["high"]))
                pos.peak_close = max(fill, float(row["close"]))
                positions[t] = pos

        # ---------- mark to market ----------
        mtm = cash
        for t, pos in positions.items():
            i = locs[t].get(day)
            px = frames[t].iloc[i]["close"] if i is not None else pos.entry_price
            mtm += pos.shares * px
        equity_hist.append((day, mtm))

    equity = pd.Series(dict(equity_hist)).sort_index()
    stats = compute_stats(trades, equity, cfg.initial_capital)
    return trades, equity, stats


def compute_stats(trades: list[Trade], equity: pd.Series, initial_capital: float) -> dict:
    stats: dict = {"n_trades": len(trades)}
    if equity.empty:
        return stats
    total_ret = equity.iloc[-1] / initial_capital - 1.0
    n_days = len(equity)
    years = n_days / 252.0
    daily_ret = equity.pct_change().dropna()
    dd = equity / equity.cummax() - 1.0

    stats.update({
        "total_return_pct": round(100 * total_ret, 2),
        "cagr_pct": round(100 * ((equity.iloc[-1] / initial_capital) ** (1 / max(years, 1e-9)) - 1), 2) if years > 0 else np.nan,
        "max_drawdown_pct": round(100 * dd.min(), 2),
        "sharpe": round(float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()), 2) if daily_ret.std() > 0 else 0.0,
        "exposure_days": n_days,
    })
    if trades:
        rets = np.array([t.ret_pct for t in trades])
        wins, losses = rets[rets > 0], rets[rets <= 0]
        gross_win = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = -sum(t.pnl for t in trades if t.pnl <= 0)
        stats.update({
            "win_rate_pct": round(100 * len(wins) / len(rets), 1),
            "avg_trade_pct": round(100 * rets.mean(), 3),
            "avg_win_pct": round(100 * wins.mean(), 3) if len(wins) else 0.0,
            "avg_loss_pct": round(100 * losses.mean(), 3) if len(losses) else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else np.inf,
            "avg_hold_days": round(float(np.mean([t.bars_held for t in trades])), 1),
        })
        reasons = pd.Series([t.exit_reason for t in trades]).value_counts().to_dict()
        stats["exit_reasons"] = reasons
    return stats

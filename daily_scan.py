#!/usr/bin/env python3
"""Daily scanner: stocks matching the official SMA20-retest pattern TODAY.

Run after the US market close. For every stock in the broad universe it checks
whether today completed a retest setup (touch of SMA20 inside a fresh
breakout window, structure intact) and, if so, emits a BUY-STOP candidate for
tomorrow: trigger = today's high * 1.001. Candidates are ranked by 63-day
momentum; the SPY>SMA200 regime and the earnings filter are applied exactly
as in the backtest.

Delivery goes through the owner's existing Val Town relay
(orenbot/telegram-notify): one POST sends the message to Telegram and, via
Val Town's std/email, the same text as an email to the account owner's
address. The Telegram credentials live only in the val's environment.
Auth: set VALTOWN_NOTIFY_SECRET (the RETEST_TOKEN value configured in the
val). Without it the scan still runs and prints the list.

Usage:
  python daily_scan.py                 # scan as of the latest close
  python daily_scan.py --date 2026-10-28   # historical dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import pandas as pd
import requests

from micro_pullback.data import fetch_daily, load_universe
from micro_pullback.earnings import load_earnings_risk
from micro_pullback.indicators import add_indicators
from micro_pullback.strategy import compute_setup
from micro_pullback.universe import get_broad_universe
from micro_pullback.variants import VARIANTS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Val Town relay (orenbot/telegram-notify): POST {text, secret, email_subject}
# -> Telegram message + email to the val owner's address.
RELAY_URL = "https://orenbot--a8852f06893911f1b1a21607ee4eb77e.web.val.run"

STRATEGY_SUMMARY_HE = (
    "האסטרטגיה בקצרה: מניה שחצתה לאחרונה מעל כל הממוצעים הנעים אחרי חצי שנה "
    "מתחתיהם, חזרה לגעת ב-SMA20 כשהמבנה (SMA100/200) שלם. קונים בפריצת השיא "
    "של יום הנגיעה, מוכרים בסגירת היום ה-10. בלי סטופ מחיר — ההגנה היא גודל "
    "פוזיציה (20% מההון, עד 5 פוזיציות) וסטופ-זמן. ‏Backtest ‏2018–2026: ‏53% "
    "הצלחה, ‏+0.21% לעסקה."
)


def build_candidates(as_of: str | None = None, universe_limit: int = 600):
    tickers = get_broad_universe(limit=universe_limit)
    today = dt.date.fromisoformat(as_of) if as_of else dt.date.today()
    fetch_start = (today - dt.timedelta(days=700)).isoformat()
    fetch_end = (today + dt.timedelta(days=1)).isoformat()
    raw = load_universe(tickers, fetch_start, fetch_end, verbose=False)

    spy = fetch_daily("SPY", (today - dt.timedelta(days=500)).isoformat(), fetch_end)
    spy_ok = bool((spy["close"] > spy["close"].rolling(200, min_periods=200).mean()).iloc[-1])

    params = VARIANTS["final"]
    candidates = []
    scan_date = None
    if spy_ok:
        calendar = pd.DatetimeIndex(sorted(set().union(*[df.index for df in raw.values()])))
        earnings = load_earnings_risk(list(raw.keys()), calendar, verbose=False)
        for t, df0 in raw.items():
            df = add_indicators(df0)
            if as_of:
                df = df[df.index <= pd.Timestamp(as_of)]
            if len(df) < 330:
                continue
            setup = compute_setup(df, params)
            if not bool(setup.iloc[-1]):
                continue
            d = df.index[-1]
            scan_date = scan_date or d
            if d != scan_date:  # stale data for this ticker
                continue
            # earnings: entry day (next trading day) or the day after must be clear
            risk = earnings.get(t, set())
            pos = calendar.searchsorted(d)
            nxt = calendar[pos + 1: pos + 3]
            if any(x in risk for x in nxt):
                continue
            row = df.iloc[-1]
            candidates.append({
                "ticker": t,
                "close": round(float(row["close"]), 2),
                "trigger": round(float(row["high"]) * 1.001, 2),
                "sma20": round(float(row["sma20"]), 2),
                "ret63_pct": round(100 * float(row["ret63"]), 1),
                "exit_plan": "close of 10th trading day",
                "tradingview": f"https://www.tradingview.com/chart/?symbol={t}",
            })
        candidates.sort(key=lambda c: -c["ret63_pct"])
    return {
        "scan_date": str(scan_date.date()) if scan_date is not None else str(today),
        "spy_regime_ok": spy_ok,
        "candidates": candidates,
    }


def format_message(res: dict) -> str:
    lines = [f"📊 סריקת ריטסט יומית — {res['scan_date']}", ""]
    if not res["spy_regime_ok"]:
        lines.append("⛔ SPY מתחת ל-SMA200 — אין כניסות חדשות היום.")
    elif not res["candidates"]:
        lines.append("אין מניות שעונות על הדפוס היום.")
    else:
        lines.append(f"{len(res['candidates'])} מועמדות (קנייה מחר בפריצת הטריגר):")
        for i, c in enumerate(res["candidates"][:10], 1):
            lines.append(f"{i}. {c['ticker']}: טריגר {c['trigger']}$ "
                         f"(סגירה {c['close']}$, SMA20 {c['sma20']}$, "
                         f"מומנטום {c['ret63_pct']:+.0f}%)")
            lines.append(f"   📈 https://www.tradingview.com/chart/?symbol={c['ticker']}")
        if len(res["candidates"]) > 10:
            lines.append(f"...ועוד {len(res['candidates']) - 10}")
        lines.append("")
        lines.append("יציאה: סגירת היום ה-10. ללא סטופ. עד 5 פוזיציות, 20% הון כ\"א.")
    lines += ["", "—", STRATEGY_SUMMARY_HE]
    return "\n".join(lines)


def send_via_relay(text: str, subject: str, log: dict | None = None) -> dict:
    """Send through the Val Town relay: Telegram + email + blob archive in one call."""
    secret = os.getenv("VALTOWN_NOTIFY_SECRET")
    if not secret:
        return {"telegram": False, "email": False, "logged": False,
                "note": "no VALTOWN_NOTIFY_SECRET"}
    payload = {"text": text, "secret": secret, "email_subject": subject}
    if log is not None:
        payload["log_key"] = f"retest-scan-{log.get('scan_date', 'unknown')}"
        payload["log_json"] = log
    r = requests.post(RELAY_URL, json=payload, timeout=60)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code != 200 or not body.get("ok"):
        raise RuntimeError(f"relay failed ({r.status_code}): {body}")
    return {"telegram": True, "email": bool(body.get("emailed")),
            "logged": bool(body.get("logged")),
            "note": body.get("email_error") or body.get("log_error") or ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="scan as-of this date (dry-run)")
    ap.add_argument("--no-send", action="store_true")
    args = ap.parse_args()

    res = build_candidates(as_of=args.date)
    text = format_message(res)
    print(text)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"scan_{res['scan_date']}.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    if not args.no_send:
        subject = (f"סריקת ריטסט {res['scan_date']}: "
                   f"{len(res['candidates'])} מועמדות")
        st = send_via_relay(text, subject, log=res)
        print(f"\n[delivery] telegram: {'sent' if st['telegram'] else 'skipped'}, "
              f"email: {'sent' if st['email'] else 'skipped'}, "
              f"archive: {'saved' if st.get('logged') else 'skipped'}"
              + (f" ({st['note']})" if st.get("note") else ""))


if __name__ == "__main__":
    main()

"""Earnings dates from SEC EDGAR (free, full history).

An earnings announcement is almost always accompanied by an 8-K filing with
Item 2.02 ("Results of Operations and Financial Condition"); its filing date
is the announcement day (or the next morning for after-hours releases filed
after the EDGAR 5:30pm ET cutoff). We therefore treat a window of trading
days around each filing date as "earnings risk days":

    [D - buffer_before, ..., D + buffer_after]   (default 1 before, 1 after)

The backtester blocks new entries on risk days and force-exits positions at
the close of the last trading day before a risk day. A live system should do
exactly the same using a forward-looking calendar (e.g. the same SEC feed,
NASDAQ's calendar API, or the broker's data) instead of historical filings.

Results are cached under data_cache/earnings/<TICKER>.json.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

from .data import CACHE_DIR

EARNINGS_CACHE = os.path.join(CACHE_DIR, "earnings")
_UA = {"User-Agent": "AlgoMicroPullBack research (contact: orenrachamim@gmail.com)"}
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_cik_map: dict[str, int] | None = None


def _load_cik_map(session: requests.Session) -> dict[str, int]:
    global _cik_map
    if _cik_map is not None:
        return _cik_map
    os.makedirs(EARNINGS_CACHE, exist_ok=True)
    path = os.path.join(EARNINGS_CACHE, "_cik_map.json")
    if os.path.exists(path):
        with open(path) as f:
            _cik_map = json.load(f)
        return _cik_map
    r = session.get(_TICKER_MAP_URL, headers=_UA, timeout=30)
    r.raise_for_status()
    _cik_map = {v["ticker"].upper(): int(v["cik_str"]) for v in r.json().values()}
    with open(path, "w") as f:
        json.dump(_cik_map, f)
    return _cik_map


def fetch_earnings_dates(ticker: str, session: requests.Session | None = None,
                         pause: float = 0.15) -> list[str]:
    """Return sorted list of ISO earnings-8-K filing dates for `ticker`.

    Empty list when the ticker is unknown to EDGAR or has no 8-K 2.02 filings
    (e.g. foreign issuers filing 6-K) — callers should treat that as
    "no earnings info" and may choose to trade without the filter.
    """
    os.makedirs(EARNINGS_CACHE, exist_ok=True)
    path = os.path.join(EARNINGS_CACHE, f"{ticker.upper()}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    sess = session or requests.Session()
    cik = _load_cik_map(sess).get(ticker.upper())
    dates: list[str] = []
    if cik is not None:
        try:
            r = sess.get(_SUBMISSIONS_URL.format(cik=cik), headers=_UA, timeout=30)
            r.raise_for_status()
            recent = r.json()["filings"]["recent"]
            for form, date, items in zip(recent["form"], recent["filingDate"],
                                         recent["items"]):
                if form == "8-K" and items and "2.02" in items:
                    dates.append(date)
            dates = sorted(set(dates))
        except (requests.RequestException, KeyError, ValueError):
            dates = []
    with open(path, "w") as f:
        json.dump(dates, f)
    time.sleep(pause)  # SEC fair-access limit is 10 req/s; stay well under
    return dates


def risk_days(earnings_dates: list[str], calendar: pd.DatetimeIndex,
              buffer_before: int = 1, buffer_after: int = 1) -> set[pd.Timestamp]:
    """Map filing dates to a set of risk trading-days on `calendar`.

    For each filing date D, the window covers the trading days from
    `buffer_before` days before D through `buffer_after` days after D
    (positions on the calendar; D snaps to the first trading day >= D).
    """
    out: set[pd.Timestamp] = set()
    if len(calendar) == 0:
        return out
    for d in earnings_dates:
        ts = pd.Timestamp(d)
        pos = calendar.searchsorted(ts)  # first trading day >= filing date
        lo = max(0, pos - buffer_before)
        hi = min(len(calendar), pos + buffer_after + 1)
        out.update(calendar[lo:hi])
    return out


def load_earnings_risk(tickers: list[str], calendar: pd.DatetimeIndex,
                       buffer_before: int = 1, buffer_after: int = 1,
                       verbose: bool = True) -> dict[str, set[pd.Timestamp]]:
    """Fetch earnings dates for all tickers and map them to risk-day sets."""
    sess = requests.Session()
    out: dict[str, set[pd.Timestamp]] = {}
    n_empty = 0
    for i, t in enumerate(tickers):
        dates = fetch_earnings_dates(t, session=sess)
        out[t] = risk_days(dates, calendar, buffer_before, buffer_after)
        if not dates:
            n_empty += 1
        if verbose and (i + 1) % 40 == 0:
            print(f"  earnings {i + 1}/{len(tickers)}")
    if verbose and n_empty:
        print(f"  note: {n_empty} tickers with no 8-K 2.02 history (filter inactive for them)")
    return out

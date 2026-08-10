"""Daily OHLCV data loader.

Fetches daily bars from the Yahoo Finance chart API using plain `requests`
(works behind TLS-terminating proxies where curl_cffi/yfinance fails) and
caches each ticker as a CSV under `data_cache/` so repeated backtests do not
re-download.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def _to_epoch(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _cache_path(ticker: str, start: str, end: str) -> str:
    safe = ticker.replace("^", "_").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{start}_{end}.csv")


def fetch_daily(ticker: str, start: str, end: str, session: requests.Session | None = None,
                max_retries: int = 4, pause: float = 0.4) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns open/high/low/close/volume.

    Prices are split/dividend-adjusted (adjclose scaling applied to OHLC).
    Returns an empty DataFrame when the ticker cannot be fetched.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(ticker, start, end)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    sess = session or requests
    params = {
        "period1": _to_epoch(start),
        "period2": _to_epoch(end),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    data = None
    for attempt in range(max_retries):
        try:
            r = sess.get(_CHART_URL.format(ticker=ticker), params=params,
                         headers=_HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except requests.RequestException:
            time.sleep(2 ** attempt)
    if data is None:
        return pd.DataFrame()

    try:
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adjclose = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError):
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adjclose": adjclose,
            "volume": quote["volume"],
        },
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None),
    )
    df.index.name = "date"
    df = df.dropna(subset=["close"])

    # Scale OHLC by the adjustment factor so the whole series is adjusted.
    factor = df["adjclose"] / df["close"]
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * factor
    df = df.drop(columns=["adjclose"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df.to_csv(path)
    time.sleep(pause)  # be polite to the API
    return df


def load_universe(tickers: list[str], start: str, end: str, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch all tickers, returning {ticker: DataFrame}; skips failed/short series."""
    out: dict[str, pd.DataFrame] = {}
    sess = requests.Session()
    for i, t in enumerate(tickers):
        df = fetch_daily(t, start, end, session=sess)
        if len(df) >= 120:  # need enough history for SMA/indicators
            out[t] = df
        elif verbose:
            print(f"  skip {t}: only {len(df)} rows")
        if verbose and (i + 1) % 25 == 0:
            print(f"  loaded {i + 1}/{len(tickers)} tickers")
    return out

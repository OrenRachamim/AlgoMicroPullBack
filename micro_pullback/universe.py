"""Stock universes for backtesting.

Static lists of liquid, large/mid-cap US stocks (survivorship bias applies, as
with any static list — noted in the README).
"""

# Small universe for quick tests / first backtest pass.
SMALL_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "AVGO", "CRM", "NFLX", "COST", "JPM", "UNH", "V", "LLY",
    "XOM", "HD", "MRK", "ADBE",
]

# Nasdaq-100 constituents (2025 membership; static list — survivorship bias
# applies). Foreign private issuers (6-K filers) have no 8-K earnings history,
# so the earnings filter is inactive for them.
NASDAQ_UNIVERSE = sorted(set(SMALL_UNIVERSE + [
    "GOOG", "PEP", "CSCO", "TMUS", "INTC", "INTU", "QCOM", "TXN", "AMAT",
    "CMCSA", "HON", "AMGN", "ISRG", "BKNG", "VRTX", "ADP", "SBUX", "GILD",
    "MU", "ADI", "PANW", "REGN", "LRCX", "MDLZ", "KLAC", "SNPS", "CDNS",
    "MELI", "CRWD", "MAR", "PYPL", "ORLY", "CSX", "ABNB", "MRVL", "NXPI",
    "ROP", "FTNT", "WDAY", "ADSK", "PCAR", "DXCM", "CHTR", "MNST", "KDP",
    "AEP", "ROST", "PAYX", "CPRT", "ODFL", "FAST", "KHC", "GEHC", "DDOG",
    "IDXX", "EA", "VRSK", "EXC", "CTSH", "XEL", "CSGP", "BKR", "ON", "TTWO",
    "ZS", "FANG", "DLTR", "WBD", "MDB", "CDW", "BIIB", "SMCI", "CEG", "TEAM",
    "APP", "PLTR", "AXON", "LULU", "MCHP", "TER", "SWKS", "EBAY", "ANSS",
]))

# Extended universe: ~S&P 100 + liquid growth names.
EXTENDED_UNIVERSE = sorted(set(SMALL_UNIVERSE + [
    "ABBV", "ABT", "ACN", "AIG", "AMAT", "AMGN", "AMT", "AXP", "BA", "BAC",
    "BK", "BKNG", "BLK", "BMY", "C", "CAT", "CB", "CI", "CL", "CMCSA",
    "COF", "COP", "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "EMR",
    "ETN", "FDX", "GD", "GE", "GILD", "GM", "GS", "HON", "IBM", "INTC",
    "INTU", "ISRG", "JNJ", "KO", "LIN", "LMT", "LOW", "LRCX", "MA", "MCD",
    "MDLZ", "MDT", "MET", "MMC", "MMM", "MO", "MS", "MU", "NEE", "NKE",
    "NOW", "ORCL", "PEP", "PFE", "PG", "PGR", "PLTR", "PM", "PANW", "QCOM",
    "RTX", "SBUX", "SCHW", "SHOP", "SO", "SPGI", "T", "TGT", "TJX", "TMO",
    "TMUS", "TXN", "UBER", "UNP", "UPS", "USB", "VRTX", "VZ", "WFC", "WMT",
    "ANET", "APP", "CRWD", "DDOG", "DELL", "KLAC", "MRVL", "NET", "SMCI", "SNOW",
]))

# Full universe: S&P 100 extension + Nasdaq-100 (~176 liquid US names).
FULL_UNIVERSE = sorted(set(EXTENDED_UNIVERSE) | set(NASDAQ_UNIVERSE))


def get_broad_universe(limit: int = 600, refresh: bool = False) -> list[str]:
    """Dynamic broad universe: top-`limit` US mega/large-cap stocks by market
    cap from the NASDAQ screener API, cached to data_cache/broad_universe.json.

    Note: this is the CURRENT membership — survivorship bias applies to
    backtests, as with the static lists.
    """
    import json
    import os
    import re

    import requests

    from .data import CACHE_DIR

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "broad_universe.json")
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            return json.load(f)[:limit]

    r = requests.get(
        "https://api.nasdaq.com/api/screener/stocks",
        params={"tableonly": "true", "limit": "1500", "marketcap": "mega|large"},
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
                 "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json()["data"]["table"]["rows"]

    def cap(row):
        try:
            return float(row["marketCap"].replace(",", ""))
        except (ValueError, AttributeError):
            return 0.0

    rows.sort(key=cap, reverse=True)
    # plain common-stock symbols only (skip units/warrants/preferreds like BRK/A, ABC^W)
    tickers = [row["symbol"].strip() for row in rows
               if re.fullmatch(r"[A-Z]{1,5}", row["symbol"].strip())]
    with open(path, "w") as f:
        json.dump(tickers, f)
    return tickers[:limit]
